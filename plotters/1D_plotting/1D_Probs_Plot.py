import numpy as np
import matplotlib.pyplot as plt

def plot_1d_probability(prob_file, t_target=100):
    """
    Loads the aggregated 1D probability file and plots probability vs. x-position 
    at the chosen time t_target.
    Expects the file to contain:
      - "times": array of shape (num_times,)
      - Keys "W=0.0", "W=0.2", etc., each mapping to an array of shape (num_times, Nx)
    """
    data = np.load(prob_file, allow_pickle=True)
    if "times" not in data:
        print("Error: 'times' not found in the file.")
        return
    times = data["times"]
    if t_target not in times:
        print(f"Error: t_target={t_target} not in times: {times}")
        return
    t_index = np.where(times == t_target)[0][0]
    
    plt.figure(figsize=(9,6))
    w_keys = [k for k in data.keys() if k.startswith("W=")]

    # Define colors (consistent with previous plots)
    colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    lines = ["solid", "dashed", (0, (3, 1, 1, 1, 1, 1)), "dotted", "dashdot", (5, (10, 3))]

    # Inside your loop over w_keys, modify the plotting code like this:

    for i, w_str in enumerate(sorted(w_keys, key=lambda x: float(x[2:]))):
        arr = data[w_str]
        if arr is None or arr.ndim != 2:
            print(f"Skipping {w_str}: unexpected shape {np.shape(arr)}")
            continue
        Nx = arr.shape[1]
        dist = arr[t_index]  # shape (Nx,)

        # Set x-axis: if Nx == 2*t_target+1 then use [-t_target,...,t_target]
        if Nx == 2*t_target + 1:
            x_axis = np.arange(-t_target, t_target + 1)
        else:
            x_axis = np.arange(Nx)

        # Mask odd x-axis points
        mask = (x_axis % 2 == 0)
        x_axis_masked = x_axis[mask]
        dist_masked = dist[mask]

        plt.plot(x_axis_masked, dist_masked, linestyle=lines[i], linewidth=3, 
                label=w_str, color=colors[i % len(colors)])

    # Axis labels and title
    plt.xlabel("x", fontsize=15)
    plt.ylabel("P(x)", fontsize=15)

    # Set x-axis limits
    plt.xlim(-80, 80)

    # Legend
    plt.legend(fontsize=15)

    # No gridlines
    plt.grid(False)

    # Format ticks: inward-facing on all four sides
    plt.tick_params(direction='in', length=6, width=1.2, labelsize=15, 
                    bottom=True, top=True, left=True, right=True)  # Major ticks on all sides

    # Set minor ticks
    plt.minorticks_on()
    plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                    bottom=True, top=True, left=True, right=True)  # Minor ticks on all sides

    # Tight layout
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    prob_file = r"aggregated_1D_probabilityFINAL.npz"
    plot_1d_probability(prob_file, t_target=100)
