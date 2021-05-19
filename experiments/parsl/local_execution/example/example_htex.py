import parsl
import os
import os.path
from os import path
from parsl.app.app import python_app, bash_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.data_provider.files import File
from parsl.configs.local_threads import config

from parsl.executors.threads import ThreadPoolExecutor

    
    

def main():

    local_threads = Config(
        executors=[
            ThreadPoolExecutor(max_threads=8, label='local_threads')
        ]
    )
    
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
                    worker_init=(
                                 'export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES\n'
                                ),
                ),
            )
        ],
        strategy=None,
    )

    parsl.clear()
    #parsl.load(local_threads)
    parsl.load(local_htex)

    @bash_app
    def generate(outputs=[]):
        return "echo $(( RANDOM )) &> {}".format(outputs[0])
    
    @bash_app
    def concat(inputs=[], outputs=[]):
        return "cat {0} > {1}".format(" ".join(i.filepath for i in inputs), outputs[0])
    
    @python_app
    def total(inputs=[]):
        total = 0
        with open(inputs[0], 'r') as f:
            for l in f:
                total += int(l)
        return total
    
    # Create 5 files with semi-random numbers
    print(f'Getting started?')
    output_files = []
    for i in range (5):
         output_files.append(generate(outputs=[File(os.path.join(os.getcwd(), 'random-%s.txt' % i))]))
    
    # Concatenate the files into a single file
    print(f'before concat')
    cc = concat(inputs=[i.outputs[0] for i in output_files],
                outputs=[File(os.path.join(os.getcwd(), 'combined.txt'))])
    
    # Calculate the sum of the random numbers
    total = total(inputs=[cc.outputs[0]])
    
    print (total.result())

if __name__ == '__main__':
    main()
