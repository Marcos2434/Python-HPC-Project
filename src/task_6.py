import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from os.path import join
from simulate import load_data, jacobi

LOAD_DIR     = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER     = 20_000
ABS_TOL      = 1e-4
N            = 100

# Worker counts to test (must be <= available CPU cores)
# We will be using the XeonGold6126
WORKERS      = [1,2,4,8,16]

# the per‐plan worker function
def solve(bid):
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi(u0, mask, MAX_ITER, ABS_TOL)
    return None

def time_for_workers(p, ids):
    """
    Use a Pool with chunksize=1 so tasks are dynamically scheduled.
    """
    t0 = time.time()
    with Pool(p) as pool:
        # imap_unordered gives tasks one‐by‐one
        for _ in pool.imap_unordered(solve, ids, chunksize=1): pass
    return time.time() - t0

if __name__ == '__main__':
    # load the first N IDs
    with open(join(LOAD_DIR,'building_ids.txt')) as f:
        all_ids = f.read().splitlines()
    ids = all_ids[:N]

    times_dyn = []
    for p in WORKERS:
        dt = time_for_workers(p, ids)
        print(f"[dynamic] p={p:2d} → {dt:.2f}s")
        times_dyn.append(dt)

    # compute speed‐ups
    speed_dyn = [times_dyn[0]/t for t in times_dyn]

    plt.figure(figsize=(6,4))
    plt.plot(WORKERS, speed_dyn, 'o-', label='dynamic')
    plt.xlabel('Workers')
    plt.ylabel('Speed-up')
    plt.title('Dynamic Scheduling speedup')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/speedup_dynamic.png', dpi=300)
    plt.show()
