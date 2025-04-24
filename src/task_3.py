from simulate import load_data, jacobi, summary_stats, LOAD_DIR, MAX_ITER, ABS_TOL
from utils import plot_floorplans

# list the IDs to visualize
with open(f'{LOAD_DIR}/building_ids.txt') as f:
    all_ids = f.read().splitlines()
sample_ids = [ all_ids[i] for i in (0, len(all_ids)//2, len(all_ids)-1) ] # some arbitrary ids, here: (first, middle, last)

results = {}
for bid in sample_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    u_final = jacobi(u0, mask, MAX_ITER, ABS_TOL)
    results[bid] = (u0, u_final, mask)

plot_floorplans(results)