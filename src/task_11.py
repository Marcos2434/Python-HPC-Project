from os.path import join
import sys
import pandas as pd

import cupy as cp
from numba import cuda

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

# Run jacobi iterations for each floor plan
MAX_ITER = 20_000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u, u_new = cp.copy(u), cp.empty_like(u[:, 1:-1, 1:-1])

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[:, 1:-1, :-2] + u[:, 1:-1, 2:] + u[:, :-2, 1:-1] + u[:, 2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        u[:, 1:-1, 1:-1][interior_mask] = u_new_interior
    return u

@cuda.jit
def jacobi_kernel(u, interior_mask, u_new):
    j, i = cuda.grid(2)
    n = cuda.blockIdx.z  # floor/building index

    if j >= 1 and j < u.shape[1] - 1 and i >= 1 and i < u.shape[2] - 1:
        if interior_mask[n, j-1, i-1]:
            u_new[n, j, i] = 0.25 * (
                u[n, j, i - 1] + u[n, j, i + 1] +
                u[n, j - 1, i] + u[n, j + 1, i]
            )

def jacobi_gpu(u, interior_mask, max_iter):
    d_u = cuda.to_device(u)
    d_mask = cuda.to_device(interior_mask)
    d_u_new = cuda.to_device(d_u)

    threads_per_block = (16, 16)
    blocks_per_grid_x = ((u.shape[2] + (threads_per_block[0] - 1)) // threads_per_block[0],
                         (u.shape[1] + (threads_per_block[1] - 1)) // threads_per_block[1], 
                         u.shape[0])
    
    for i in range(max_iter):
        jacobi_kernel[blocks_per_grid_x, threads_per_block](d_u, d_mask, d_u_new)
        cuda.synchronize()
        d_u, d_u_new = d_u_new, d_u
    return d_u.copy_to_host()

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # number of buildings to load, taken as the first argument in the cl
    if len(sys.argv) < 2: N = 1
    else: N = int(sys.argv[1])
    
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    all_u = cp.empty_like(all_u0)
    all_u = jacobi_gpu(all_u0, all_interior_mask, MAX_ITER)#, ABS_TOL)

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(cp.array(u), interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))