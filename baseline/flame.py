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
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial
import math

from pytools.obj_array import obj_array_vectorize
import pickle

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten, unflatten
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.euler import inviscid_operator, split_conserved
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    check_step,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
    rk4_step, 
    lsrk54_step, 
    lsrk144_step, 
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedBoundary,
    AdiabaticSlipBoundary,
    DummyBoundary
)
from mirgecom.initializers import (
    Lump,
    Uniform,
    MixtureDiscontinuity,
    MixtureInitializer
)
from mirgecom.eos import PyrometheusMixture
import cantera
import pyrometheus as pyro

from logpyle import IntervalTimer

from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (initialize_logmgr,
    logmgr_add_many_discretization_quantities, logmgr_add_cl_device_info)
logger = logging.getLogger(__name__)


#def generate_mesh():
    #char_len = 0.001
    #box_ll = (0.0, 0.0)
    #box_ur = (0.25, 0.01)
    #num_elements = (int((box_ur[0]-box_ll[0])/char_len),
                        #int((box_ur[1]-box_ll[1])/char_len))
#
    #from meshmode.mesh.generation import generate_regular_rect_mesh
    #mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur, n=num_elements,
      #boundary_tag_to_face={
          #"Inflow":["-x"],
          #"Outflow":["+x"],
          #"Wall":["+y","-y"]
          #}
      #)
    ##print("%d elements" % elements)
#
    #return mesh
#

@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         snapshot_pattern="y0euler-{step:06d}-{rank:04d}.pkl",
         restart_step=None, use_profiling=False, use_logmgr=False):
    """Drive the Y0 example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    """logging and profiling"""
    logmgr = initialize_logmgr(use_logmgr, filename="y0euler.sqlite",
        mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    #nviz = 500
    #nrestart = 500
    nviz = 10
    nrestart = 250
    #current_dt = 5.0e-8 # stable with euler
    current_dt = 5.0e-8 # stable with rk4
    #current_dt = 4e-7 # stable with lrsrk144
    t_final = 5.e-1

    dim = 2
    order = 1
    exittol = .09
    #t_final = 0.001
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    nstatus = 10000000000
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    vel_burned = np.zeros(shape=(dim,))
    vel_unburned = np.zeros(shape=(dim,))

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.
    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 2000.0
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index("C2H4")
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
    eos = PyrometheusMixture(pyrometheus_mechanism, tguess=temp_unburned)
    species_names = pyrometheus_mechanism.species_names

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned state (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned state (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    # use the burned conditions with a lower temperature
    bulk_init = MixtureDiscontinuity(dim=dim, x0=0.05, sigma=0.01, nspecies=nspecies,
                              tl=temp_ignition, tr=temp_unburned,
                              pl=pres_burned, pr=pres_unburned,
                              ul=vel_burned, ur=vel_unburned,
                              yl=y_burned, yr=y_unburned)

    inflow_init = MixtureInitializer(dim=dim, nspecies=nspecies, pressure=pres_burned, 
                                     temperature=temp_ignition, massfractions= y_burned,
                                     velocity=vel_burned)
    outflow_init = MixtureInitializer(dim=dim, nspecies=nspecies, pressure=pres_unburned, 
                                     temperature=temp_unburned, massfractions= y_unburned,
                                     velocity=vel_unburned)

    inflow = PrescribedBoundary(inflow_init)
    outflow = PrescribedBoundary(outflow_init)
    wall = AdiabaticSlipBoundary()
    dummy = DummyBoundary()

    from grudge import sym
    boundaries = {sym.DTAG_BOUNDARY("Inflow"): inflow,
                  sym.DTAG_BOUNDARY("Outflow"): outflow,
                  sym.DTAG_BOUNDARY("Wall"): wall}

    if restart_step is None:
        char_len = 0.001
        box_ll = (0.0, 0.0)
        box_ur = (0.25, 0.01)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len),
                            int((box_ur[1]-box_ll[1])/char_len))
    
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=box_ll, b=box_ur, n=num_elements,
          boundary_tag_to_face={
              "Inflow":["-x"],
              "Outflow":["+x"],
              "Wall":["+y","-y"]
              }
          )
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
        local_nelements = local_mesh.nelements

    else:  # Restart
        with open(snapshot_pattern.format(step=restart_step, rank=rank), "rb") as f:
            restart_data = pickle.load(f)

        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    if restart_step is None:
        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_state = bulk_init(t=0., x_vec=nodes, eos=eos)
        # for uniform background initial condition
        #current_state = bulk_init(nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        current_state = unflatten(
            actx, discr.discr_from_dd("vol"),
            obj_array_vectorize(actx.from_numpy, restart_data["state"]))

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
            extract_vars_for_logging, units_for_logging)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", "t_log.max",
                            "min_pressure", "max_pressure",
                            "min_temperature", "max_temperature"])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
    #    initname = initializer.__class__.__name__
    initname = "flame1d"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    timestepper = rk4_step
    #timestepper = lsrk54_step
    #timestepper = lsrk144_step
    #timestepper = euler_step

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    
    def my_rhs(t, state):
        # check for some troublesome output types
        inf_exists = not np.isfinite(discr.norm(state, np.inf))
        if inf_exists:
            if rank == 0:
                logging.info("Non-finite values detected in simulation, exiting...")
            # dump right now
            sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              q=state, vizname=casename,
                              step=999999999, t=t, dt=current_dt,
                              nviz=1, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm, vis_timer=vis_timer,
                              overwrite=True,s0=s0_sc,kappa=kappa_sc)
            exit()

        cv = split_conserved(dim=dim, q=state)
        return ( inviscid_operator(discr, q=state, t=t, boundaries=boundaries, eos=eos)
                 + eos.get_species_source_terms(cv))

    def my_checkpoint(step, t, dt, state):

        write_restart = (check_step(step, nrestart)
                         if step != restart_step else False)
        if write_restart is True:
            with open(snapshot_pattern.format(step=step, rank=rank), "wb") as f:
                pickle.dump({
                    "local_mesh": local_mesh,
                    "state": obj_array_vectorize(actx.to_numpy, flatten(state)),
                    "t": t,
                    "step": step,
                    "global_nelements": global_nelements,
                    "num_parts": nparts,
                    }, f)

        cv = split_conserved(dim, state)
        reaction_rates = eos.get_production_rates(cv)
        viz_fields = [("reaction_rates", reaction_rates)]
        
        return sim_checkpoint(discr=discr, visualizer=visualizer, eos=eos,
                              q=state, vizname=casename,
                              step=step, t=t, dt=dt, nstatus=nstatus,
                              nviz=nviz, exittol=exittol,
                              constant_cfl=constant_cfl, comm=comm, vis_timer=vis_timer,
                              overwrite=True, viz_fields=viz_fields)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      checkpoint=my_checkpoint,
                      get_timestep=get_timestep, state=current_state,
                      t_final=t_final, t=current_t, istep=current_step,
                      logmgr=logmgr,eos=eos,dim=dim)

    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_checkpoint(current_step, t=current_t,
                  dt=(current_t - checkpoint_t),
                  state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    use_logging = True
    use_profiling = False

    # crude command line interface
    # get the restart interval from the command line
    print(f"Running {sys.argv[0]}\n")
    nargs = len(sys.argv)
    if nargs > 1:
        restart_step = int(sys.argv[1])
        print(f"Restarting from step {restart_step}")
        main(restart_step=restart_step,use_profiling=use_profiling,use_logmgr=use_logging)
    else:
        print(f"Starting from step 0")
        main(use_profiling=use_profiling,use_logmgr=use_logging)

# vim: foldmethod=marker
