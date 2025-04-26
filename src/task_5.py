'''
For this task, we are asked to implement static-scheduling 
parallel processing of the Jacobi floorplan solver
Tasks:
  - Divide a subset of N floorplans evenly across P worker processes
  - Measure elapsed time for each P
  - Compute and plot speed-ups
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from os.path import join
from simulate import load_data, jacobi


LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4

# Number of floorplans
N = 100

# Worker counts to test (must be <= available CPU cores)
# We will be using the XeonGold6126
WORKER_COUNTS = [1, 2, 4, 8, 16]


def worker(chunk_ids):
    """
    Process a static chunk of building IDs sequentially.
    Only runs the Jacobi solver (no output to minimize overhead).
    """
    for bid in chunk_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi(u0, mask, MAX_ITER, ABS_TOL)


def time_for_workers(worker_count, building_ids):
    """
    Start worker_count processes, each handling an equal slice of building_ids.
    Returns the elapsed wall-clock time.
    """
    chunks = np.array_split(building_ids, worker_count)
    processes = []
    t0 = time.time()
    for chunk in chunks:
        p = Process(target=worker, args=(chunk.tolist(),))
        p.start()
        processes.append(p)
    for p in processes:
        p.join() # wait for all processes to finish
    return time.time() - t0


if __name__ == '__main__':
    # load and slice building IDs to select only N floorplans
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()
    building_ids = all_ids[:N]

    # run experiments
    elapsed_times = []
    for p in WORKER_COUNTS:
        elapsed = time_for_workers(p, building_ids)
        print(f"Workers={p}, Time={elapsed:.2f}s")
        elapsed_times.append(elapsed)

    # compute speed-ups relative to serial run
    speedups = [elapsed_times[0] / t for t in elapsed_times]

    # plot speed-ups
    plt.figure(figsize=(6, 4))
    plt.plot(WORKER_COUNTS, speedups, marker='o')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speed-up (T₁ / Tₚ)')
    plt.title('Static Scheduling Speed-up')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/speedup_static.png', dpi=300)
