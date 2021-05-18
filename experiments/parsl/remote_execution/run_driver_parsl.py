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


local_executor = ThreadPoolExecutor(max_threads=8, label='local_threads')
local_thread_config = Config(
    executors=[local_executor]
)

# a configuration to run locally with pilot jobs
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.channels import SSHInteractiveLoginChannel
from parsl.executors import HighThroughputExecutor
from parsl.providers import LSFProvider
from parsl.providers import SlurmProvider
from parsl.launchers import JsrunLauncher
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface
from parsl.addresses import address_by_hostname
from parsl.data_provider.data_manager import default_staging
from parsl.data_provider.http import HTTPSeparateTaskStaging

lassen_ssh = SSHInteractiveLoginChannel(
    hostname="lassen.llnl.gov",
    username="manders",
    script_dir="/p/gpfs1/manders/parsl"
)

tmp = address_by_hostname()
print(f'address {tmp}')

lassen_executor= HighThroughputExecutor(
    label='lassen_htex',
    working_dir='/p/gpfs1/manders/',
    #address=address_by_interface('ib0'),   # assumes Parsl is running on a login node
    #address='lassen.llnl.gov',   # assumes Parsl is running on a login node
    address='10.194.85.172',
    #address=address_by_hostname(),   # 
    worker_port_range=(50000, 55000),
    worker_debug=True,
    #storage_access=[HTTPSeparateTaskStaging()],
    provider=LSFProvider(
        channel=lassen_ssh,
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

lassen_config = Config(
    executors=[lassen_executor],
    strategy=None,
)

distributed_remote_config = Config(
    executors=[local_executor, lassen_executor],
    strategy=None,
)

print(parsl.__version__)
parsl.set_stream_logger()

parsl.clear()
#parsl.load(local_config)
parsl.load(distributed_remote_config)
#parsl.load(m1_htex)
#parsl.load(lassen_config)

# build a string that loads conda correctly
def load_conda():
   return(
            'CONDA_BASE=$(conda info --base)\n'
            'source ${CONDA_BASE}/etc/profile.d/conda.sh\n'
            'conda deactivate\n'
            'conda activate mirgeDriver.flame1d\n'
         )

@bash_app(executors=['local_threads'])
def run_mirge_init(execution_string='', stdout='run.stdout', stderr='run.stderr', inputs=[], outputs=[]):
    return(execution_string)

@bash_app(executors=['lassen_htex'])
def run_mirge_sim(execution_string='', stdout='run.stdout', stderr='run.stderr', inputs=[], outputs=[]):
    return(execution_string)

# first run is just an init, locally
init_restart_file = File(os.path.join(os.getcwd(), 'flame1d-000000-0000.pkl'))
intro_str = 'echo "Running flame1d_init"\n'
conda_str = load_conda()
execution_str = 'python -u -m mpi4py flame_init.py\n'
ex_str = intro_str+conda_str+execution_str
stdout_str = 'flame1d_init.stdout'
stderr_str = 'flame1d_init.stdout'
#flame_init = run_mirge_init(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, outputs=[init_restart_file])

#print(flame_init.outputs)
#print('Done: {}'.format(flame_init.done()))
#print('Result: {}'.format(flame_init.result()))
#print('Done: {}'.format(flame_init.done()))

# second run is a restart from the init, on lassen
#run_restart_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005-0000.pkl'))
#run_viz_file = File(os.path.join(os.getcwd(), 'flame1d_run-000005.pvtu'))
#input_file = File(os.path.join(os.getcwd(), 'run1_params.yaml'))
#casename='flame1d_run'
#intro_str = 'echo "Running flame1d_run"\n'
#execution_str = ('python -u -m mpi4py flame_run.py -r {restart_file} -c {casename} -i {input_file}\n'.
                    #format(restart_file=init_restart_file, casename=casename, input_file=input_file)
                #)
#ex_str = intro_str+conda_str+execution_str
#stdout_str = 'flame1d_run1.stdout'
#stderr_str = 'flame1d_run1.stdout'
#flame_run = run_mirge_sim(execution_string=ex_str, stdout=stdout_str, stderr=stderr_str, inputs=[input_file, init_restart_file], outputs=[run_viz_file, run_restart_file])
#
## I get error messages if I do this, even though it finishes correctly...
#print(flame_run.outputs[0])
#print('Done: {}'.format(flame_run.done()))
#print('Result: {}'.format(flame_run.outputs[0].result()))
#print('Done: {}'.format(flame_run.done()))
#
