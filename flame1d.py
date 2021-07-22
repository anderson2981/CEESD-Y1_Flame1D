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
import yaml
import logging
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from meshmode.dof_array import thaw
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local
)
from mirgecom.restart import (
    write_restart_file
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
    PrescribedViscousBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.initializers import (
    PlanarDiscontinuity,
    MixtureInitializer
)
from mirgecom.transport import SimpleTransport
from mirgecom.eos import PyrometheusMixture
import cantera
import pyrometheus as pyro

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr, logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)
logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, casename="flame1d",
         user_input_file=None, restart_file=None, use_profiling=False,
         use_logmgr=False, use_lazy_eval=False):
    """Drive the 1D Flame example."""

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()
    if use_profiling:
        if use_lazy_eval:
            raise RuntimeError("Cannot run lazy with profiling.")
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        if use_lazy_eval:
            from meshmode.array_context import PytatoPyOpenCLArrayContext
            actx = PytatoPyOpenCLArrayContext(queue)
        else:
            actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # default input values that will be read from input (if they exist)
    nviz = 100
    nrestart = 100
    nhealth = 100
    nstatus = 1
    current_dt = 1e-9
    t_final = 1.e-3
    order = 1
    integrator = "rk4"
    fuel = "C2H4"

    if user_input_file:
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            input_data = None
        input_data = comm.bcast(input_data, root=0)

        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            fuel = input_data["fuel"]
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if(integrator not in allowed_integrators):
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    allowed_fuels = ["H2", "C2H4"]
    if(fuel not in allowed_fuels):
        error_message = "Invalid fuel selection: {}".format(fuel)
        raise RuntimeError(error_message)

    if rank == 0:
        print("#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tTime integration {integrator}")
        print(f"\tFuel: {fuel}")
        print("#### Simluation control data: ####")

    dim = 2
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    current_step = 0

    vel_burned = np.zeros(shape=(dim,))
    vel_unburned = np.zeros(shape=(dim,))

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.

    from mirgecom.mechanisms import get_mechanism_cti
    if fuel == "C2H4":
        mech_cti = get_mechanism_cti("uiuc")
    elif fuel == "H2":
        # mech_cti = get_mechanism_cti("sanDiego")
        mech_cti = get_mechanism_cti("sanDiego_trans")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 1500.0
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    if fuel == "C2H4":
        stoich_ratio = 3.0
    if fuel == "H2":
        stoich_ratio = 0.5
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index(fuel)
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

    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # now find the conditions for the burned gas
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    pyrometheus_mechanism = pyro.get_thermochem_class(cantera_soln)(actx.np)

    kappa = 1.6e-5  # Pr = mu*rho/alpha = 0.75
    mu = 1.e-5
    species_diffusivity = 1.e-5 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=species_diffusivity)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned,
                             transport_model=transport_model)
    species_names = pyrometheus_mechanism.species_names

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    flame_start_loc = 0.10

    # use the burned conditions with a lower temperature
    bulk_init = PlanarDiscontinuity(dim=dim,
                                    disc_location=flame_start_loc,
                                    sigma=0.0005,
                                    nspecies=nspecies,
                                    temperature_right=temp_ignition,
                                    temperature_left=temp_unburned,
                                    pressure_right=pres_burned,
                                    pressure_left=pres_unburned,
                                    velocity_right=vel_burned,
                                    velocity_left=vel_unburned,
                                    species_mass_right=y_burned,
                                    species_mass_left=y_unburned)
    inflow_init = MixtureInitializer(dim=dim,
                                     nspecies=nspecies,
                                     pressure=pres_burned,
                                     temperature=temp_ignition,
                                     massfractions=y_burned,
                                     velocity=vel_burned)
    outflow_init = MixtureInitializer(dim=dim,
                                      nspecies=nspecies,
                                      pressure=pres_unburned,
                                      temperature=temp_unburned,
                                      massfractions=y_unburned,
                                      velocity=vel_unburned)

    def symmetry(nodes, eos, cv=None, **kwargs):
        dim = len(nodes)

        if cv is not None:
            mass = cv.mass
            momentum = cv.momentum
            momentum[1] = -1.0 * momentum[1]
            energy = cv.energy
            species_mass = cv.species_mass
            return make_conserved(dim=dim,
                                  mass=mass,
                                  momentum=momentum,
                                  energy=energy,
                                  species_mass=species_mass)

    def dummy(nodes, eos, cv=None, **kwargs):
        dim = len(nodes)

        if cv is not None:
            mass = cv.mass
            momentum = cv.momentum
            energy = cv.energy
            species_mass = cv.species_mass
            return make_conserved(dim=dim,
                                  mass=mass,
                                  momentum=momentum,
                                  energy=energy,
                                  species_mass=species_mass)

    inflow = PrescribedViscousBoundary(q_func=inflow_init)
    outflow = PrescribedViscousBoundary(q_func=outflow_init)
    wall_symmetry = PrescribedViscousBoundary(q_func=symmetry)

    boundaries = {DTAG_BOUNDARY("Inflow"): inflow,
                  DTAG_BOUNDARY("Outflow"): outflow,
                  #DTAG_BOUNDARY("Wall"): wall}
                  #DTAG_BOUNDARY("Wall"): wall_dummy}
                  DTAG_BOUNDARY("Wall"): wall_symmetry}

    restart_step = None
    if restart_file is None:
        char_len = 0.0001
        box_ll = (0.0, 0.0)
        box_ur = (0.2, 0.00125)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len),
                            int((box_ur[1]-box_ll[1])/char_len))

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                boundary_tag_to_face={
                                    "Inflow": ["+x"],
                                    "Outflow": ["-x"],
                                    "Wall": ["+y", "-y"]})
        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")
    discr = EagerDGDiscretization(actx,
                                  local_mesh,
                                  order=order,
                                  mpi_communicator=comm)
    nodes = thaw(actx, discr.nodes())

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_state = bulk_init(x_vec=nodes, eos=eos, time=0.)
        # for uniform background initial condition
        #current_state = bulk_init(nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            restart_state = restart_data["state"]
            current_state = connection(restart_state)
        else:
            current_state = restart_data["state"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
        current_state = restart_data["state"]

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
            extract_vars_for_logging, units_for_logging)
        logmgr_set_time(logmgr, current_step, current_t)
        logmgr.add_quantity(log_cfl, interval=nstatus)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("cfl.max", "cfl = {value:1.4f}\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
            ("max_temperature",    "{value:7g})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(discr)

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

    def my_write_viz(step, t, dt, state, dv=None,
                     reaction_rates=None, ts_field=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if reaction_rates is None:
            reaction_rates = eos.get_production_rates(state)
        if ts_field is None:
            ts_field, cfl, dt = my_get_timestep(t, dt, state)
        viz_fields = [("CV_rho", state.mass),
                      ("CV_rhoU", state.momentum[0]),
                      ("CV_rhoV", state.momentum[1]),
                      ("CV_rhoE", state.energy),
                      ("DV", dv),
                      ("reaction_rates", reaction_rates),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.species_mass[i]/state.mass)
            for i in range(nspecies))
        write_visfile(discr, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(dv):
        health_error = False
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if check_range_local(discr, "vol", dv.pressure, 1, 2e6):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation.")

        return health_error

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            from mirgecom.viscous import get_viscous_timestep
            ts_field = current_cfl * get_viscous_timestep(discr, eos=eos, cv=state)
            from grudge.op import nodal_min
            dt = nodal_min(discr, "vol", ts_field)
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            ts_field = get_viscous_cfl(discr, eos=eos, dt=dt, cv=state)
            from grudge.op import nodal_max
            cfl = nodal_max(discr, "vol", ts_field)

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            dt = get_sim_timestep(discr, state, t, dt, current_cfl, eos,
                                  t_final, constant_cfl)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(dv), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                my_write_viz(step=step, t=t, dt=dt, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        return (
            ns_operator(discr, cv=state, t=t, boundaries=boundaries, eos=eos) +
            eos.get_species_source_terms(cv=state)
        )

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, eos, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state, dt=current_dt,
                      t_final=t_final, t=current_t, istep=current_step)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    final_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                current_cfl, eos, t_final, constant_cfl)
    my_write_viz(step=current_step, t=current_t, dt=final_dt, state=current_state,
                 dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "flame1d"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_file=restart_file, user_input_file=input_file,
         use_profiling=args.profile, use_lazy_eval=args.lazy, use_logmgr=args.log)
