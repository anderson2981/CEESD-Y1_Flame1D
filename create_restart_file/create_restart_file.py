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
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze

#from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
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
    check_range_local,
    global_reduce
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
#    rk4_step,
#    lsrk54_step,
#    lsrk144_step,
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    SymmetryBoundary,
    PrescribedFluidBoundary
)
from mirgecom.fluid import make_conserved, species_mass_fraction_gradient
#from mirgecom.transport import SimpleTransport, PowerLawTransport
#from mirgecom.viscous import get_viscous_timestep, get_viscous_cfl, diffusive_flux
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)
from pytools.obj_array import make_obj_array

from mirgecom.limiter import bound_preserving_limiter

#######################################################################################


class InterpolateCanteraFile:

    def __init__(
            self, *, dim=2):
                   
        self._dim = dim

    def __call__(self, x_vec, eos, *, time=0.0):

        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        x = x_vec[0]
        actx = x.array_context

        from meshmode.dof_array import DOFArray        
        import numpy as np
        from numpy import genfromtxt
#        import matplotlib.pyplot as plt
        from pytools.obj_array import make_obj_array
        from scipy.interpolate import CubicSpline
        
        data_x = actx.to_numpy(x_vec[0][0])

        data_rho = actx.to_numpy(x_vec[0][0])*0.0
        data_ru =  data_rho*0.0
        data_rv =  data_rho*0.0
        data_y0 =  data_rho*0.0
        data_y1 =  data_rho*0.0
        data_y2 =  data_rho*0.0
        data_y3 =  data_rho*0.0
        data_y4 =  data_rho*0.0
        data_y5 =  data_rho*0.0
        data_y6 =  data_rho*0.0
        data_y7 =  data_rho*0.0
        data_y8 =  data_rho*0.0
        pres = data_rho*0.0
        temp = data_rho*0.0

        data_ct = genfromtxt('adiabatic_flame_sandiego.csv',
                             skip_header=1,delimiter=",")

        #align flame with the boundaries of the domain refined region
        data_ct[:,0] = data_ct[:,0] - 0.028 + 0.0025

        cs = CubicSpline(data_ct[:,0], data_ct[:,3], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_rho[ii,ij] = cs(data_x[ii,ij])
         
        cs = CubicSpline(data_ct[:,0], data_ct[:,1], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_ru[ii,ij] = cs(data_x[ii,ij])

        data_rv[:,:] = 0.0  

        cs = CubicSpline(data_ct[:,0], data_ct[:,2], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            temp[ii,ij] = cs(data_x[ii,ij])
        
#        plt.plot(data_ct[:,0], data_ct[:,1])
#        plt.plot(data_x[:,0],data_ru[:,0])
#        plt.show()
#            
#        import sys
#        sys.exit()

        cs = CubicSpline(data_ct[:,0], data_ct[:,4], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y0[ii,ij] = cs(data_x[ii,ij])

        cs = CubicSpline(data_ct[:,0], data_ct[:,5], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y1[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,6], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y2[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,7], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y3[ii,ij] = cs(data_x[ii,ij])

        cs = CubicSpline(data_ct[:,0], data_ct[:,8], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y4[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,9], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y5[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,10], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y6[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,11], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y7[ii,ij] = cs(data_x[ii,ij])
            
        cs = CubicSpline(data_ct[:,0], data_ct[:,12], extrapolate=False)
        for ii in range(0,data_x.shape[0]):
          for ij in range(0,data_x.shape[1]):
            data_y8[ii,ij] = cs(data_x[ii,ij])


#        cte = 101325 + data_ct[-1,3]*data_ct[-1,1]**2
#        unbd_pres = cte - data_ct[0,3]*data_ct[0,1]**2
#        burn_pres = 101325.0
#        print(unbd_pres, burn_pres)
#        pres_aux = cte - data_ct[:,3]*data_ct[:,1]**2
#        cs = CubicSpline(data_ct[:,0], pres_aux, extrapolate=False)
#        for ii in range(0,data_x.shape[0]):
#          for ij in range(0,data_x.shape[1]):
#            pres[ii,ij] = cs(data_x[ii,ij])

#        plt.plot(data_ct[:,0], pres_aux)
#        plt.show()
#            
#        import sys
#        sys.exit()

        x_min = 0.00# - 0.028 + 0.0025
        x_max = 0.08 - 0.028 + 0.0025

        u_x = DOFArray(actx, data=(actx.from_numpy(np.array(data_ru)), ))
        u_x = actx.np.where(actx.np.less(x,-x_min), data_ct[0,1], u_x)
        u_x = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,1], u_x)

        u_y = DOFArray(actx, data=(actx.from_numpy(np.array(data_rv)), ))

        temperature = DOFArray(actx, data=(actx.from_numpy(np.array(temp)), ))
        temperature = actx.np.where(actx.np.less(x,-x_min), data_ct[0,2], temperature)
        temperature = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,2], temperature)

        mass = DOFArray(actx, data=(actx.from_numpy(np.array(data_rho)), ))
        mass = actx.np.where(actx.np.less(x,-x_min), data_ct[0,3], mass)
        mass = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,3], mass)

#        pressure = DOFArray(actx, data=(actx.from_numpy(np.array(pres)), ))
#        pressure = actx.np.where(actx.np.less(x,-x_min), unbd_pres, pressure)
#        pressure = actx.np.where(actx.np.less(x,x_max), pressure, burn_pres)
        
        y0 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y0)), ))
        y0 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 4], y0)
        y0 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 4], y0)
        y1 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y1)), ))
        y1 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 5], y1)
        y1 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 5], y1)
        y2 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y2)), ))
        y2 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 6], y2)
        y2 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 6], y2)
        y3 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y3)), ))
        y3 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 7], y3)
        y3 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 7], y3)
        y4 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y4)), ))
        y4 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 8], y4)
        y4 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 8], y4)
        y5 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y5)), ))
        y5 = actx.np.where(actx.np.less(x,-x_min), data_ct[0, 9], y5)
        y5 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1, 9], y5)
        y6 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y6)), )) 
        y6 = actx.np.where(actx.np.less(x,-x_min), data_ct[0,10], y6) 
        y6 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,10], y6) 
        y7 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y7)), ))
        y7 = actx.np.where(actx.np.less(x,-x_min), data_ct[0,11], y7)
        y7 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,11], y7)
        y8 = DOFArray(actx, data=(actx.from_numpy(np.array(data_y8)), )) 
        y8 = actx.np.where(actx.np.less(x,-x_min), data_ct[0,12], y8) 
        y8 = actx.np.where(actx.np.greater(x,x_max), data_ct[-1,12], y8) 
        
        velocity = make_obj_array([u_x, u_y])
        species = make_obj_array([y0, y1, y2, y3, y4, y5, y6, y7, y8])

        y = make_obj_array([
            actx.np.where(actx.np.less(species[i], 1e-6), 0.0, species[i])
                   for i in range(9)])

        #pressure = mass*0.0 + 101325.0

#        mass = eos.get_density(pressure, temperature,
#                               species_mass_fractions=y)

        internal_energy = eos.get_internal_energy(temperature=temperature,
                                                  species_mass_fractions=y)
        kinetic_energy = 0.5*np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        momentum = mass*velocity

        specmass = mass*y

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=specmass)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


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
    
    casename="flame_1D_25um_sandiego_p=2"

    rst_path = "./"
    viz_path = "./"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"    

    # default i/o frequencies
    nviz = 2000
    nrestart = 2000 
    nhealth = 10
    nstatus = 100

    # default timestepping control
    integrator = "euler"
    current_dt = 1.0e-9
    t_final = 1.e-3

    # discretization and model control
    order = 2
    fuel = "C2H4"

##########################################################################################3

    if integrator == "euler":
        timestepper = euler_step

    dim = 2
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    current_step = 0

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
#    mechanism_file = "BFER_2S"
    mechanism_file = "sandiego"
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0

    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyrometheus_mechanism = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned)
    species_names = pyrometheus_mechanism.species_names
    gas_model = GasModel(eos=eos)

    flow_init = InterpolateCanteraFile(dim=dim)

##########################################################################################3

    restart_step = None
    if restart_file is None:

        xx = np.loadtxt('x_025.dat')
        yy = np.loadtxt('y_025.dat')
        coords = tuple((xx,yy))

        from meshmode.mesh.generation import generate_box_mesh
        generate_mesh = partial(generate_box_mesh,
                                axis_coords=coords,
                                periodic=(False, True),
                                boundary_tag_to_face={
                                   # "wall": ["-y","+y"],
                                    "inlet": ["-x"],
                                    "outlet": ["+x"]})

        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order,
                                             mpi_communicator=comm)
    nodes = actx.thaw(dcoll.nodes())
    
    quadrature_tag = None

#####################################################################################

    temperature_seed = temp_unburned

    if restart_file is None:
        if rank == 0:
            print("Initializing soln.")
            logging.info("Initializing soln.")
        current_cv = flow_init(x_vec=nodes, eos=eos, time=0.)

    current_state = make_fluid_state(current_cv, gas_model, temperature_seed)
    temperature_seed = current_state.temperature

    ##################################################

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s\n")])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

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

##################################################################

    def get_production_rates(cv, temperature):
        return make_obj_array([eos.get_production_rates(cv, temperature)])
    compute_production_rates = actx.compile(get_production_rates)

    def my_write_viz(step, t, state):
        reaction_rates, = compute_production_rates(state.cv, state.temperature)
        viz_fields = [("CV_rho", state.cv.mass),
                      ("CV_rhoU", state.cv.momentum),
                      ("CV_rhoE", state.cv.energy),
                      ("DV_P", state.pressure),
                      ("DV_T", state.temperature),
                      ("DV_U", state.velocity[0]),
                      ("DV_V", state.velocity[1]),
                      ("reaction_rates", reaction_rates)]

        #~ species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], state.cv.species_mass_fractions[i]) 
                                for i in range(nspecies))

        print('Writing solution file...')
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)
                      
        return

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

            print('Writing restart file...')
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

        return

##############################################################################

#    limiting_function = bound_preserving_limiter
#    def limiter(cv,dv=None,temp=None):

#        spec_lim = make_obj_array([
#            limiting_function(discr, cv.species_mass_fractions[i],
#                              modify_average=True)
#                   for i in range(nspecies)])

#        kin_energy = 0.5*np.dot(cv.velocity,cv.velocity)
#        int_energy = cv.energy - cv.mass*kin_energy
#        
#        energy_lim = cv.mass*(
#            gas_model.eos.get_internal_energy(temp, species_mass_fractions=spec_lim)
#            + kin_energy
#        )

#        return make_conserved(dim=dim, mass=cv.mass, energy=energy_lim,
#                       momentum=cv.momentum, species_mass=cv.mass*spec_lim )

    def limit_species_source(cv, pressure, temperature, 
                             species_enthalpies=None):
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                                     mmin=0.0, mmax=1.0, modify_average=True)
            for i in range(nspecies)
        ])

        aux = cv.mass*0.0
        for i in range(0,nspecies):
          aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        mass_lim = eos.get_density(pressure=pressure, temperature=temperature,
                                   species_mass_fractions=spec_lim)

        energy_lim = mass_lim*(
            gas_model.eos.get_internal_energy(temperature,
                                              species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        cv_limited = make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
                                    momentum=mass_lim*cv.velocity,
                                    species_mass=mass_lim*spec_lim)

        return cv_limited

##############################################################################

    # update temperature value
    non_limited_state = make_fluid_state(current_cv, gas_model, temperature_seed)

    # apply limiter and reevaluate energy
    limited_cv = limit_species_source(non_limited_state.cv,
                                      non_limited_state.dv.pressure,
                                      non_limited_state.dv.temperature)

    # get new fluid_state with limited species and respective energy
    fluid_state = make_fluid_state(limited_cv, gas_model, temperature_seed)

    my_write_restart(step=0, t=0.0, cv=fluid_state.cv,
                     tseed=fluid_state.temperature)

    my_write_viz(step=0, t=0.0, state=fluid_state)


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
         use_profiling=args.profile,
         lazy=args.lazy, rst_filename=restart_file)
