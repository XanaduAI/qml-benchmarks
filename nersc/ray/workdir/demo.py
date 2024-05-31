
'''
https://docs.ray.io/en/latest/ray-core/tasks.html#ray-remote-functions
https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html
'''

import numpy as np
import ray
import time

from datetime import datetime

def print_elapsed(t1, t2):
    print("%.6f s" % ((t2 - t1).microseconds / 1e6))


# Assume this Ray node has 16 CPUs and 16G memory.
ray.init()


@ray.remote(num_cpus=4)
def process(file):
    # Actual work is reading the file and process the data.
    # Assume it needs to use 2G memory.
    task_id = ray.get_runtime_context().get_task_id()
    print(file, "on", task_id)
    for i in range(int(4e2)):
        np.cos(np.random.rand(100,100,100))
    #with open(file, 'w') as fp:
    #    fp.write('done')
    return file

print('starting list')

NUM_FILES = 16
result_refs = []
for i in range(NUM_FILES):
    # By default, process task will use 1 CPU resource and no other resources.
    # This means 16 tasks can run concurrently
    # and will OOM since 32G memory is needed while the node only has 16G.
    result_refs.append(process.remote(f"{i}.csv"))

print('list done')
time.sleep(3)
print('starting ray.get()')

t_start = datetime.now()
res = ray.get(result_refs)
t_end = datetime.now()
print_elapsed(t_start, t_end)

print(res)
