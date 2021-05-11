"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
import pyopencl as cl
from functools import partial

from pytools.obj_array import obj_array_vectorize
import pickle

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten, unflatten
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.fluid import split_conserved
from mirgecom.simutil import (
    sim_checkpoint,
    generate_and_distribute_mesh
)
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.initializers import PlanarDiscontinuity
from mirgecom.transport import SimpleTransport
from mirgecom.eos import PyrometheusMixture
import cantera
import pyrometheus as pyro

@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         snapshot_pattern="flame1d-{step:06d}-{rank:04d}.pkl",
         ):
    """Drive the Y0 example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))


    dim = 2
    order = 1
    vel_burned = np.zeros(shape=(dim,))
    vel_unburned = np.zeros(shape=(dim,))

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.
    from mirgecom.mechanisms import get_mechanism_cti
    # uiuc C2H4
    #mech_cti = get_mechanism_cti("uiuc")
    # sanDiego H2
    mech_cti = get_mechanism_cti("sanDiego")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 1500.0
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    # H2
    stoich_ratio = 0.5
    #C2H4
    #stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    # C2H4
    #i_fu = cantera_soln.species_index("C2H4")
    # H2
    i_fu = cantera_soln.species_index("H2")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0
    pres_unburned = one_atm

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY
    can_p = cantera_soln.P
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.


    # now find the conditions for the burned gas
    cantera_soln.equilibrate('TP')
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    casename = "flame1d"
    pyrometheus_mechanism = pyro.get_thermochem_class(cantera_soln)(actx.np)

    # C2H4
    mu = 1.e-5
    kappa = 1.6e-5  # Pr = mu*rho/alpha = 0.75
    # H2
    mu = 1.e-5
    kappa = mu*0.08988/0.75  # Pr = mu*rho/alpha = 0.75

    species_diffusivity = 1.e-5 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa, species_diffusivity=species_diffusivity)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned, transport_model=transport_model)
    species_names = pyrometheus_mechanism.species_names

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned state (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned state (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    flame_start_loc = 0.05
    flame_speed = 1000

    # use the burned conditions with a lower temperature
    bulk_init = PlanarDiscontinuity(dim=dim, disc_location=flame_start_loc, sigma=0.01, nspecies=nspecies,
                              temperature_left=temp_ignition, temperature_right=temp_unburned,
                              pressure_left=pres_burned, pressure_right=pres_unburned,
                              velocity_left=vel_burned, velocity_right=vel_unburned,
                              species_mass_left=y_burned, species_mass_right=y_unburned)

    char_len = 0.001
    box_ll = (0.0, 0.0)
    box_ur = (0.25, 0.01)
    num_elements = (int((box_ur[0]-box_ll[0])/char_len),
                        int((box_ur[1]-box_ll[1])/char_len))

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur, 
      n=num_elements, mesh_type="X",
      boundary_tag_to_face={
          "Inflow":["-x"],
          "Outflow":["+x"],
          "Wall":["+y","-y"]
          }
      )
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(actx, local_mesh, order=order, mpi_communicator=comm)
    nodes = thaw(actx, discr.nodes())

    # for Discontinuity initial conditions
    state = bulk_init(t=0., x_vec=nodes, eos=eos)
    # for uniform background initial condition
    #current_state = bulk_init(nodes, eos=eos)

    visualizer = make_visualizer(discr, order)

    with open(snapshot_pattern.format(step=0, rank=rank), "wb") as f:
        pickle.dump({
            "local_mesh": local_mesh,
            "state": obj_array_vectorize(actx.to_numpy, flatten(state)),
            "t": 0.,
            "step": 0,
            "global_nelements": global_nelements,
            "num_parts": nparts,
            }, f)

    cv = split_conserved(dim, state)
    reaction_rates = eos.get_production_rates(cv)
    viz_fields = [("reaction_rates", reaction_rates)]

    sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                   q=state, vizname=casename, nviz=0,
                   comm=comm, overwrite=True, viz_fields=viz_fields)

    exit()


if __name__ == "__main__":
    #freeze_support()
    main()

# vim: foldmethod=marker
