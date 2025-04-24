from simulate import load_data, jacobi, summary_stats, LOAD_DIR, MAX_ITER, ABS_TOL
import matplotlib.pyplot as plt

def plot_floorplans(results):
    """
    Display each floorplan's initial condition beside its converged temperature field.
    """
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 4*n),
                            constrained_layout=True)

    for row, (bid, (u0, u_final, mask)) in enumerate(results.items()):
        # initial condition
        ax0 = axes[row, 0]
        
        # https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/
        im0 = ax0.imshow(u0[1:-1,1:-1], 
                        cmap='coolwarm', 
                        interpolation='nearest',
                        origin='lower')
        ax0.set_title(f'{bid} - initial')
        fig.colorbar(im0, ax=ax0, 
                    fraction=0.046, 
                    pad=0.04,
                    label='Temperature (°C)')

        # final steady state
        ax1 = axes[row, 1]
        
        # https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/
        im1 = ax1.imshow(u_final[1:-1,1:-1], 
                        cmap='inferno', 
                        interpolation='nearest',
                        origin='lower')
        ax1.set_title(f'{bid} - steady state')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04,
                    label='Temperature (°C)')

    plt.suptitle('Wall-Heating Simulation Results', fontsize=16)
    plt.savefig('plots/florrplans')