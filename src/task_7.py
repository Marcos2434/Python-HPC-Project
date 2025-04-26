import sys
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from matplotlib.ticker import MaxNLocator

from os.path import join

# In case we're not running with the profiler
try:
    profile
except NameError:
    profile = lambda fn: fn

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

# Run jacobi iterations for each floor plan
MAX_ITER = 20_000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

# Numba-optimized Jacobi using explicit loops for cache locality
@jit(nopython=True)
def jacobi(u, interior_mask, max_iter, atol):
    u_new = u.copy()
    rows, cols = u.shape
    # mask dims are (rows-2, cols-2)
    for it in range(max_iter):
        max_delta = 0.0
        # iterate only interior points 1..rows-2, 1..cols-2
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # check if the point is interior
                if interior_mask[i-1, j-1]:
                    val = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    diff = abs(u[i, j] - val)
                    if diff > max_delta: max_delta = diff
                    u_new[i, j] = val
                else:
                    # keep boundary or outside values unchanged
                    u_new[i, j] = u[i, j]
        # check convergence
        if max_delta < atol: break

        u, u_new = u_new, u
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    
    # Test
    # # Load data
    # with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    #     building_ids = f.read().splitlines()

    # # number of buildings to load, taken as the first argument in the cl
    # if len(sys.argv) < 2: N = 1
    # else: N = int(sys.argv[1])
    
    # building_ids = building_ids[:N]

    # # Load floor plans
    # all_u0 = np.empty((N, 514, 514))
    # all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    # for i, bid in enumerate(building_ids):
    #     u0, interior_mask = load_data(LOAD_DIR, bid)
    #     all_u0[i] = u0
    #     all_interior_mask[i] = interior_mask

    # all_u = np.empty_like(all_u0)
    # for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
    #     u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    #     all_u[i] = u

    # # Print summary statistics in CSV format
    # stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    # print('building_id, ' + ', '.join(stat_keys))  # CSV header
    # for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
    #     stats = summary_stats(u, interior_mask)
    #     print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        
    
    # number_of_fp = np.arange(1, 2+1)
    number_of_fp = np.arange(10, 20+1)
    
    elapsed_times = []
    
    for n in tqdm(number_of_fp):
        t_init = time()
        
        N = n

        # Load data
        with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
            building_ids = f.read().splitlines()

        building_ids = building_ids[:N]

        # Load floor plans
        all_u0 = np.empty((N, 514, 514))
        all_interior_mask = np.empty((N, 512, 512), dtype='bool')
        for i, bid in enumerate(building_ids):
            u0, interior_mask = load_data(LOAD_DIR, bid)
            all_u0[i] = u0
            all_interior_mask[i] = interior_mask

        all_u = np.empty_like(all_u0)
        for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
            u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
            all_u[i] = u

        # Print summary statistics in CSV format
        stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
        for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
            stats = summary_stats(u, interior_mask)
        
        t_elapsed = time() - t_init
        elapsed_times.append(t_elapsed)

    plt.figure(figsize=(8, 6))
    plt.plot(list(number_of_fp), elapsed_times, marker='o', label='Elapsed Time')
    plt.title('Computation Time vs. Number of Floor Plans')
    plt.xlabel('Number of Floor Plans')
    plt.ylabel('Elapsed Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
    plt.tight_layout()
    plt.savefig('plots/task7.png', dpi=300)
    plt.show()