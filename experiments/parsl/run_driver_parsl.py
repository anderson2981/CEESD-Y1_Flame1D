import parsl
import os
import os.path
from os import path
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
from parsl.data_provider.files import File
#parsl.set_stream_logger() # <-- log everything to stdout

# a configuration to run on local threads
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

local_threads = Config(
    executors=[
        ThreadPoolExecutor(max_threads=8, label='local_threads')
    ]
)

# a configuration to run locally with pilot jobs
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor
from parsl.providers import LSFProvider
from parsl.providers import SlurmProvider
from parsl.launchers import JsrunLauncher
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface

local_htex = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_Local",
            worker_debug=True,
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy=None,
)

lassen_htex = Config(
    executors=[
        HighThroughputExecutor(
            label="Lassen_HTEX",
            working_dir='/p/gpfs1/manders/',
            #address=address_by_interface('ib0'),   # assumes Parsl is running on a login node
            address='lassen.llnl.gov',   # assumes Parsl is running on a login node
            worker_port_range=(50000, 55000),
            worker_debug=True,
            provider=LSFProvider(
                launcher=JsrunLauncher(overrides='-g 1 -a 1'),
                walltime="00:20:00",
                nodes_per_block=1,
                init_blocks=1,
                max_blocks=1,
                worker_init=(
                             'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'
                             'export PYOPENCL_CTX="port:tesla"\n'
                            ),
                project='uiuc',
                cmd_timeout=600
            ),
        )
    ],
    strategy=None,
)

quartz_htex = Config(
    executors=[
        HighThroughputExecutor(
            label="Quartz_HTEX",
            working_dir='/p/lscratchh/manders/work/CEESD/parsl_remote/flame1d',
            address='quartz.llnl.gov',   # assumes Parsl is running on a login node
            worker_port_range=(50000, 55000),
            worker_debug=True,
            provider=SlurmProvider(
                launcher=SrunLauncher(),
                walltime="00:20:00",
                nodes_per_block=1,
                init_blocks=1,
                max_blocks=1,
                worker_init=(
                             'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'
                             'export PYOPENCL_CTX="port:pthread\n"'
                            ),
                account='uiuc',
                partition='quartz',
                cmd_timeout=600
            ),
        )
    ],
    strategy=None,
)

print(parsl.__version__)
parsl.set_stream_logger()

parsl.clear()
#parsl.load(local_threads)
#parsl.load(local_htex)
parsl.load(lassen_htex)

# build a string that loads conda correctly
def load_conda():
   return(
            'CONDA_BASE=$(conda info --base)\n'
            'source ${CONDA_BASE}/etc/profile.d/conda.sh\n'
            'conda deactivate\n'
            'conda activate mirgeDriver.flame1d\n'
            #'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'    
            #'export PYOPENCL_CTX="port:tesla"\n'
         )

#@bash_app
#def init_flame1d(execution_string='', stdout='flame1d_init.stdout', stderr='flame1d_init.stderr', outputs=[]):
    #return(execution_string)
#
#@bash_app
#def run_flame1d(execution_string='', stdout='flame1d_run.stdout', stderr='flame1d_run.stderr', inputs=[], outputs=[]):
    #return(execution_string)

@bash_app
def run_mirge(execution_string='', stdout='run.stdout', stderr='run.stderr', inputs=[], outputs=[]):
    return(execution_string)

# first run is just an init, on lassen
init_restart_file = File(os.path.join(os.getcwd(), 'flame1d-000000-0000.pkl'))
intro_str = 'echo "Running flame1d_init"\n'
conda_str = load_conda()
execution_str = 'python -u -m mpi4py flame_init.py\n'
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_init.stdout'
stderr_str = 'flame1d_init.stdout'
flame_init = run_mirge(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, outputs=[init_restart_file])

#print(flame_init.outputs)
print('Done: {}'.format(flame_init.done()))
print('Result: {}'.format(flame_init.result()))
print('Done: {}'.format(flame_init.done()))

# second run is a restart from the init, on lassen
run_restart_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005-0000.pkl'))
run_viz_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005.pvtu'))
input_file = File(os.path.join(os.getcwd(), 'run1_params.yaml'))
casename='flame1d_run'
intro_str = 'echo "Running flame1d_run"\n'
execution_str = ('python -u -m mpi4py flame_run.py -r {restart_file} -c {casename} -i {input_file}\n'.
                    format(restart_file=init_restart_file, casename=casename, input_file=input_file)
                )
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_run1.stdout'
stderr_str = 'flame1d_run1.stdout'
flame_run = run_mirge(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, inputs=[input_file, init_restart_file], outputs=[run_viz_file, run_restart_file])

# I get error messages if I do this, even though it finishes correctly...
print(flame_run.outputs[0])
print('Done: {}'.format(flame_run.done()))
print('Result: {}'.format(flame_run.outputs[0].result()))
print('Done: {}'.format(flame_run.done()))

## third run is a restart from the result of the second run, but on quartz
run2_restart_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005-0000.pkl'))
run2_viz_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005.pvtu'))
input_file2 = File(os.path.join(os.getcwd(), 'run2_params.yaml'))
casename='flame1d_run2'
intro_str = 'echo "Running flame1d_run2"\n'
execution_str = ('python -u -m mpi4py flame_run.py -r {restart_file} -c {casename} -i {input_file}\n'.
                    format(restart_file=run_restart_file, casename=casename, input_file=input_file2)
                )
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_run2.stdout'
stderr_str = 'flame1d_run2.stdout'
flame_run2 = run_mirge(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, inputs=[input_file2, run_restart_file], outputs=[run2_viz_file, run2_restart_file])

# I get error messages if I do this, even though it finishes correctly...
print(flame_run.outputs[0])
print('Done: {}'.format(flame_run2.done()))
print('Result: {}'.format(flame_run2.outputs[0].result()))
print('Done: {}'.format(flame_run2.done()))
