import sys
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from simulate import load_data, jacobi, summary_stats
from matplotlib.ticker import MaxNLocator
from time import time
from tqdm import tqdm

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'


# Run jacobi iterations for each floor plan
MAX_ITER = 20_000
ABS_TOL = 1e-4

if __name__ == '__main__':
    number_of_fp = np.arange(10, 20)
    
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
plt.savefig('plots/task2.png', dpi=300)
plt.show()