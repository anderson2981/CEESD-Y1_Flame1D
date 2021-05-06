import parsl
import os
import os.path
from os import path
from parsl.app.app import python_app, bash_app
#from parsl.configs.local_threads import config
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

print(parsl.__version__)

parsl.clear()
#parsl.load(local_threads)
parsl.load(local_htex)

@bash_app
def init_flame1d(stdout='flame1d_init.stdout', stderr='flame1d_init.stderr', outputs=[]):
    return 'mpirun -n 1 python -u -m mpi4py flame_init.py'

@bash_app
def run_flame1d(stdout='flame1d_run.stdout', stderr='flame1d_run.stderr', inputs=[], outputs=[]):
    return 'mpirun -n 1 python -u -m mpi4py flame_run.py -r {inputs[0]} -c flame1d_run'

init_restart_file = File(os.path.join(os.getcwd(), 'flame1d-000000-0000.pkl'))
flame_init = init_flame1d(outputs=[init_restart_file])

print(flame_init.outputs)
print('Done: {}'.format(flame_init.done()))
print('Result: {}'.format(flame_init.result()))
#print('stdout: {}'.format(flame_init.stdout))
#print(f"this? {flame_init.outputs[0].result()}")
#path.exists(flame_init.outputs[0].result())
print('Done: {}'.format(flame_init.done()))

1/0

run_restart_file = File(os.path.join(os.getcwd(), 'flame1d-000005-0000.pkl'))
run_viz_file = File(os.path.join(os.getcwd(), 'flame1d-000005.pvtu'))
flame_run = run_flame1d(inputs=[init_restart_file], outputs=[run_viz_file, run_restart_file])

print(flame_run.outputs)
print('Done: {}'.format(flame_run.done()))
print('Result: {}'.format(flame_run.result()))
print('Done: {}'.format(flame_run.done()))
