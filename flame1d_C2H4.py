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
import sys
import numpy as np
import pyopencl as cl
#import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze

#from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import BoundaryDomainTag#, VolumeDomainTag
from grudge.dof_desc import DOFDesc, as_dofdesc, DISCR_TAG_BASE, VolumeDomainTag
from grudge.dof_desc import DD_VOLUME_ALL

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
    check_range_local,
    global_reduce,
    force_evaluation
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.fluid import make_conserved, species_mass_fraction_gradient

from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    logmgr_add_device_memory_usage,
    set_sim_state
)
from pytools.obj_array import make_obj_array

#######################################################################################

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger = logging.getLogger()
#root_logger.addHandler(h1)
#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.INFO, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""
    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None, x_thickness=None, y_thickness=None, amplitude):
        r"""Initialize the sponge parameters.
        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.
        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge = xpos*0.0

        if (self._x_max is not None):
          x0 = (self._x_max - self._x_thickness)
          dx = +((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.greater(xpos, x0),
                  actx.np.where(actx.np.greater(xpos, self._x_max),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
                  0.0
          )

        if (self._x_min is not None):
          x0 = (self._x_min + self._x_thickness)
          dx = -((xpos - x0)/self._x_thickness)
          sponge = sponge + self._amplitude * actx.np.where(
              actx.np.less(xpos, x0),
                  actx.np.where(actx.np.less(xpos, self._x_min),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
              0.0
          )

        return sponge


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # ~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # default i/o frequencies
    nviz = 5000
    nrestart = 25000 
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    #integrator = "euler"
    current_dt = 2.0e-9
    t_final = 4.00e-3

    niter = int(t_final/current_dt)
    
    # discretization and model control
    order = 3

######################################################

    use_overintegration = False

    local_dt = False
    constant_cfl = True
    current_cfl = 0.4

    dim = 2
    current_t = 0
    current_step = 0

##########################################################################################3

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)
        
    force_eval = True
    if integrator == "compiled_lsrk45":
        timestepper = _compiled_stepper_wrapper
        force_eval = False
    if integrator == "euler":
        timestepper = euler_step

    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if (constant_cfl == False):
          print(f"\tcurrent_dt = {current_dt}")
          print(f"\tt_final = {t_final}")
        else:
          print(f"\tconstant_cfl = {constant_cfl}")
          print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\tniter = {niter}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")

##########################################################################

    # {{{  Set up initial state using Cantera
    # Use Cantera for initialization
    mechanism_file = "/home/tulio/Desktop/postdoc/1D_code/myDGcode/1D_flame/MirgeCOM/uiuc_mod"
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_burned = 2413.4068590583524

    y_unburned = [0.06372925403106741,0.21806608724095825,9.520238462595688e-17,0.0,9.286921396822578e-15,4.0794962163368895e-14,0.7182046587279243]
    y_burned = [0.0,0.012786559046723626,0.16456300628256262,0.022470711798054044,0.08192751277168275,0.0,0.7182522101009768]
                

    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm
    pres_unburned = one_atm
    pres_burned = one_atm

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = \
        get_pyrometheus_wrapper_class_from_cantera(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=1350.0)

    species_names = pyrometheus_mechanism.species_names

    # }}}    
    
    # {{{ Initialize transport model

    from mirgecom.transport import TransportModel
    from meshmode.dof_array import DOFArray
    class MixtureAveragedTransport(TransportModel):
    
        def __init__(self, pyrometheus_mech, factor, alpha=0.0, diff_switch=1.0e-7):
            self._pyro_mech = pyrometheus_mech
            self._alpha = alpha
            self._factor = factor
            self._diff_switch = diff_switch
    
        def viscosity(self, cv, dv, eos = None):
            return self._factor*self._pyro_mech.get_mixture_viscosity_mixavg(
                    dv.temperature, cv.species_mass_fractions)
    
        def bulk_viscosity(self, cv, dv, eos = None):
            return self._alpha*self.viscosity(cv, dv)
    
        def volume_viscosity(self, cv, dv, eos = None):
            return (self._alpha - 2.0/3.0)*self.viscosity(cv, dv)
    
        def thermal_conductivity(self, cv, dv, eos = None):
            return self._factor*(self._pyro_mech.get_mixture_thermal_conductivity_mixavg(
                dv.temperature, cv.species_mass_fractions,))
    
        def species_diffusivity(self, cv, dv, eos = None):
            return self._factor*(
                self._pyro_mech.get_species_mass_diffusivities_mixavg(
                    dv.temperature, dv.pressure, cv.species_mass_fractions,
                    self._diff_switch)
            )

    # }}}    

    transport_model = MixtureAveragedTransport(pyrometheus_mechanism, factor=1.0)
    
    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned:")
    print(f"T = {temp_unburned}")
    print(f"Y = {y_unburned}")
    print(f" ")
    print(f"Burned:")
    print(f"T = {temp_burned}")
    print(f"Y = {y_burned}")
    print(f" ")

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

###############################################################################

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, pressure, temperature, dd=None):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i], 
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        cv = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

        # make a new CV with the limited variables
        return cv

#    def _get_temperature_update(cv, temperature):
#        y = cv.species_mass_fractions
#        e = gas_model.eos.internal_energy(cv) / cv.mass
#        return actx.np.abs(
#            pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y))

#    get_temperature_update = actx.compile(_get_temperature_update)

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed,
            limiter_func=_limit_fluid_cv
        )

    get_fluid_state = actx.compile(_get_fluid_state)
    
##############################################################################

#    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{nparts}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)
                               
    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")
            ])

        logmgr_add_device_memory_usage(logmgr, queue)
        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

##############################################################################

    restart_step = None
    if restart_file is None:  
        sys.exit()
    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order,
                                             mpi_communicator=comm)

    nodes = actx.thaw(dcoll.nodes())

    dd_vol = DD_VOLUME_ALL

    zeros = nodes[0]*0.0
    ones = nodes[0]*0.0 + 1.0

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, dcoll)
    h_min = vol_min(length_scales)
    h_max = vol_max(length_scales)

    if rank == 0:
        print("----- Discretization info ----")
        print(f"Dcoll: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    for i in range(nparts):
        if rank == i:
            print(f"{rank=},{local_nelements=},{global_nelements=}")
        comm.Barrier()

##############################################################################

    if restart_file is None:
        sys.exit()
    else:
        if local_dt:
            current_t = restart_data["step"]
        else:
            current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            sys.exit()
        else:
            current_cv = restart_data["cv"]
            tseed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

##############################################################################

    # initialize the sponge field
    sponge_x_thickness = 0.065
    sponge_amp = 10000.0

    xMaxLoc = +0.100
    xMinLoc = -0.100
        
    sponge_init = InitSponge(x_max=xMaxLoc,
                             x_min=xMinLoc,
                             x_thickness=sponge_x_thickness,
                             amplitude=sponge_amp)

    sponge_sigma = sponge_init(x_vec=nodes)
    
#    ref_cv = ref_state(x_vec=nodes, eos=eos, time=0.)
    ref_cv = 1.0*current_cv

#################################################################

    vel_burned = 5.3242521589756695
    vel_unburned = 0.6542240968866195

    mass_burned = 0.143747172793783
    mass_unburned = 1.1698532681009575

#    def _flow_bnd(state_minus, nodes, eos, side):
##        pressure = one_atm + 0*nodes[0]
#        if side == 'burned':
#            velocity = make_obj_array([vel_burned + 0*nodes[0], 0*nodes[0]])
#            mass = mass_burned + 0*nodes[0]
##            temperature = temp_burned + 0*nodes[0]
#            temperature = eos.temperature(state_minus.cv,
#                                          temperature_seed=temp_burned)
#            y = make_obj_array([y_burned[i] + 0*nodes[0]
#                            for i in range(nspecies)])
#        if side == 'unburned':
#            velocity = make_obj_array([vel_unburned + 0*nodes[0], 0*nodes[0]])
#            mass = mass_unburned + 0*nodes[0]
##            temperature = temp_unburned + 0*nodes[0]
#            temperature = eos.temperature(state_minus.cv,
#                                          temperature_seed=temp_unburned)
#            y = make_obj_array([y_unburned[i] + 0*nodes[0]
#                            for i in range(nspecies)])

##        mass = eos.get_density(pressure, temperature, y)
#        specmass = mass * y
#        mom = mass * velocity
#        internal_energy = eos.get_internal_energy(temperature=temperature,
#                                                   species_mass_fractions=y)
#        kinetic_energy = 0.5 * np.dot(velocity, velocity)
#        energy = mass * (internal_energy + kinetic_energy)

#        return make_conserved(dim=2, mass=mass, energy=energy,
#                              momentum=mom, species_mass=specmass)

#    inflow_btag = DTAG_BOUNDARY("inlet")
#    inflow_bnd_discr = discr.discr_from_dd(inflow_btag)
#    inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
#    bnd_smoothness = force_evaluation(actx, inflow_nodes[0]*0.0)
#    def _inflow_bnd_state_func(discr, btag, gas_model, state_minus,
#                               **kwargs):
#        inflow_bnd_cond = ( #force_evaluation(actx,
#                          _flow_bnd(state_minus, inflow_nodes,
#                                           eos, 'unburned'))
#        inflow_state = make_fluid_state(inflow_bnd_cond, gas_model,
#                                        temp_unburned, bnd_smoothness)
##        inflow_state = force_evaluation(actx, inflow_state)
#        return inflow_state

#    outflow_btag = DTAG_BOUNDARY("outlet")
#    outflow_bnd_discr = discr.discr_from_dd(outflow_btag)
#    outflow_nodes = actx.thaw(outflow_bnd_discr.nodes())
#    bnd_smoothness = force_evaluation(actx, outflow_nodes[0]*0.0)
#    def _outflow_bnd_state_func(discr, btag, gas_model, state_minus,
#                                **kwargs):
#        outflow_bnd_cond = ( #force_evaluation(actx,
#                           _flow_bnd(state_minus, outflow_nodes,
#                                           eos, 'burned'))
#        outflow_state = make_fluid_state(outflow_bnd_cond, gas_model,
#                                         temp_burned, bnd_smoothness)
##        outflow_state = force_evaluation(actx, outflow_state)
#        return outflow_state


#    inflow_boundary = PrescribedFluidBoundary(boundary_state_func=_inflow_bnd_state_func)
##    outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_outflow_bnd_state_func)
#    outflow_boundary = OutflowBoundary(boundary_pressure=101325.0)

#    boundaries = {DTAG_BOUNDARY("inlet"): inflow_boundary,
#                  DTAG_BOUNDARY("outlet"): outflow_boundary               
#                  }

    from mirgecom.fluid import make_conserved
    from pytools.obj_array import make_obj_array
    def _outflow_bnd(dcoll, dd_bdry, state_minus, nodes, eos):

        nhat = actx.thaw(dcoll.normal(dd_bdry))

        rhoref = mass_burned
        u_ref = vel_burned
        v_ref = 0.0
        pref = 101325.0
        R = eos.gas_const(state_minus.cv)
        temp_ref = pref/(rhoref*R)
   
        rtilde = state_minus.cv.mass - rhoref
        utilde = state_minus.velocity[0] - u_ref
        vtilde = state_minus.velocity[1] - v_ref
        ptilde = state_minus.dv.pressure - pref

        un_tilde = +utilde*nhat[0] + vtilde*nhat[1]
        ut_tilde = -utilde*nhat[1] + vtilde*nhat[0]
    
        a = eos.gamma(state_minus.cv,temp_ref)*pref/rhoref

        c1 = -rtilde*a**2 + ptilde
        c2 = rhoref*a*vtilde
        c3 = rhoref*a*utilde + ptilde
        c4 = 0.0
        r_tilde_bnd = 1.0/a**2*(-c1 + 0.5*c3 + 0.5*c4)
        un_tilde_bnd = 1.0/(2.0*rhoref*a)*(c3 - c4)
        ut_tilde_bnd = 1.0/(rhoref*a)*c2
        p_tilde_bnd = 1.0/2.0*(c3 + c4)

        mass = r_tilde_bnd + rhoref        

        det = 1.0/(nhat[0]**2 + nhat[1]**2)
        u_x = u_ref + det*( nhat[0]*un_tilde_bnd - nhat[1]*ut_tilde_bnd )
        u_y = v_ref + det*( nhat[1]*un_tilde_bnd + nhat[0]*ut_tilde_bnd )

        p = p_tilde_bnd + pref

        temperature = p/(R*mass)

        kin_energy = 0.5*mass*(u_x**2 + u_y**2)
        int_energy = mass*eos.get_internal_energy(
            temperature, state_minus.cv.species_mass_fractions)
        energy = kin_energy + int_energy

        mom = make_obj_array([u_x*mass, u_y*mass])

        specmass = mass*state_minus.cv.species_mass_fractions

        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


    def _inflow_bnd(dcoll, dd_bdry, state_minus, nodes, eos):

        nhat = -actx.thaw(dcoll.normal(dd_bdry))

        rhoref = mass_unburned
        u_ref = vel_unburned
        v_ref = 0.0
        pref = 101325.0
        R = eos.gas_const(state_minus.cv)
        temp_ref = pref/(rhoref*R)
    
        rtilde = state_minus.cv.mass - rhoref
        utilde = state_minus.velocity[0] - u_ref
        vtilde = state_minus.velocity[1] - v_ref
        ptilde = state_minus.dv.pressure - pref

        un_tilde = +utilde*nhat[0] + vtilde*nhat[1]
        ut_tilde = -utilde*nhat[1] + vtilde*nhat[0]
    
        a = eos.gamma(state_minus.cv,temp_ref)*pref/rhoref

        c4 = (-rhoref*a*un_tilde + ptilde)
        r_tilde_bnd = 1.0/(2.0*a**2)*c4
        un_tilde_bnd = -1.0/(2.0*rhoref*a)*c4
        ut_tilde_bnd = 0.0
        p_tilde_bnd = 1.0/2.0*c4

        mass = r_tilde_bnd + rhoref        

        det = 1.0/(nhat[0]**2 + nhat[1]**2)
        u_x = u_ref + det*( nhat[0]*un_tilde_bnd - nhat[1]*ut_tilde_bnd )
        u_y = v_ref + det*( nhat[1]*un_tilde_bnd + nhat[0]*ut_tilde_bnd )

        p = p_tilde_bnd + pref

        temperature = p/(R*mass)

        kin_energy = 0.5*mass*(u_x**2 + u_y**2)
        int_energy = mass*eos.get_internal_energy(
            temperature, state_minus.cv.species_mass_fractions)
        energy = kin_energy + int_energy

        mom = make_obj_array([u_x*mass, u_y*mass])

        specmass = mass*state_minus.cv.species_mass_fractions

        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


    def _inflow_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        inflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        inflow_nodes = actx.thaw(inflow_bnd_discr.nodes())
        inflow_bnd_cond = _inflow_bnd(dcoll, dd_bdry, state_minus, nodes, eos)
        return make_fluid_state(cv=inflow_bnd_cond, gas_model=gas_model,
                                temperature_seed=300.0)

    def _outflow_bnd_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        outflow_bnd_discr = dcoll.discr_from_dd(dd_bdry)
        outflow_nodes = actx.thaw(outflow_bnd_discr.nodes())
        outflow_bnd_cond = _outflow_bnd(dcoll, dd_bdry, state_minus, nodes, eos)
        return make_fluid_state(cv=outflow_bnd_cond, gas_model=gas_model,
                                temperature_seed=temp_burned)

    outflow_bnd = PrescribedFluidBoundary(boundary_state_func=_outflow_bnd_state_func)
    inflow_bnd = PrescribedFluidBoundary(boundary_state_func=_inflow_bnd_state_func)

    boundaries = {BoundaryDomainTag("inlet"): inflow_bnd,
                  BoundaryDomainTag("outlet"): outflow_bnd}

####################################################################################

    visualizer = make_visualizer(dcoll)

    initname = "flame1D"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     t_initial=current_t,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

##################################################################

    import grudge.op as op
    from meshmode.dof_array import DOFArray
    def get_production_rates(cv, temperature):
        return make_obj_array([eos.get_production_rates(cv, temperature)])
    compute_production_rates = actx.compile(get_production_rates)

    def my_write_viz(step, t, state):

        reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
#                      ("reaction_rates", reaction_rates),
                      ("sponge", sponge_sigma),
                      ]
                      
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.cv.species_mass_fractions[i])
                for i in range(nspecies))

#        if (state.tv.viscosity is not None):
#            viz_fields.extend([
#                ("TV_viscosity", state.tv.viscosity),
#                ("TV_thermal_conductivity", state.tv.thermal_conductivity),
#                ("TV_"+species_names[0], state.tv.species_diffusivity[0]),
#                ("TV_"+species_names[1], state.tv.species_diffusivity[1]),
#                ("TV_"+species_names[2], state.tv.species_diffusivity[2]),
#                ("TV_"+species_names[3], state.tv.species_diffusivity[3]),
#                ("TV_"+species_names[4], state.tv.species_diffusivity[4]),
#                ("TV_"+species_names[5], state.tv.species_diffusivity[5]),
#                ("TV_"+species_names[6], state.tv.species_diffusivity[6]),
#                ])  

        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

##################################################################

    def my_health_check(cv, dv):
        health_error = False
        pressure = force_evaluation(actx, dv.pressure)
        temperature = force_evaluation(actx, dv.temperature)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        return health_error

####################################################################################

    import os
    from mirgecom.fluid import velocity_gradient, species_mass_fraction_gradient    
    def my_pre_step(step, t, dt, state):
        
        if logmgr:
            logmgr.tick_before()

        cv, tseed = state
        cv = force_evaluation(actx, cv)
        tseed = force_evaluation(actx, tseed)

        fluid_state = get_fluid_state(cv, tseed)

        if local_dt:
            t = force_evaluation(actx, t)
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                gas_model, constant_cfl=constant_cfl, local_dt=local_dt)
            dt = force_evaluation(actx, actx.np.minimum(dt, current_dt))
        else:
            if constant_cfl:
                dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                      t_final, constant_cfl, local_dt)
             
        try:
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=fluid_state.cv, tseed=tseed)

            if do_viz:            
                ns_rhs, grad_cv, grad_t = \
                    ns_operator(dcoll, state=fluid_state, time=t,
                                boundaries=boundaries, gas_model=gas_model,
                                return_gradients=True,
                                quadrature_tag=quadrature_tag)
                
                my_write_viz(step=step, t=t, state=fluid_state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=fluid_state)
            raise

        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt


    def my_rhs(t, state):
        cv, tseed = state

        smoothness = cv.mass*0.0

        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=tseed,
            limiter_func=_limit_fluid_cv
        )

        ns_rhs, grad_cv, grad_t = (
            ns_operator(dcoll, state=fluid_state,
                        time=t, boundaries=boundaries,
                        gas_model=gas_model,
                        return_gradients=True,
                        quadrature_tag=quadrature_tag)
        )

        chem_rhs = (
            eos.get_species_source_terms(fluid_state.cv, fluid_state.temperature)
        )

        sponge_rhs = sponge_func(cv=fluid_state.cv, cv_ref=ref_cv, sigma=sponge_sigma)

        cv_rhs = ns_rhs + chem_rhs + sponge_rhs

        return make_obj_array([cv_rhs, fluid_state.temperature])


    def my_post_step(step, t, dt, state):
        if logmgr:
            if local_dt:    
                set_dt(logmgr, 1.0)
            else:
                set_dt(logmgr, dt)            
            logmgr.tick_after()

        return state, dt

##############################################################################

    current_state = get_fluid_state(current_cv, tseed)
    current_state = force_evaluation(actx, current_state)

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=make_obj_array([current_state.cv, tseed]),
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    my_write_viz(step=current_step, t=current_t, #dt=current_dt,
                 state=current_state)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

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
    casename = "flame1D"
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

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile, casename=casename,
         lazy=args.lazy, rst_filename=restart_file)
