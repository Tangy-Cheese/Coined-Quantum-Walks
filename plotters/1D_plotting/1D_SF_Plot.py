import numpy as np
import matplotlib.pyplot as plt

def plot_1d_fidelity(fid_file):
    """
    Loads the aggregated 1D fidelity file and plots fidelity vs. time.
    Expects:
      - "times": array of shape (num_times,)
      - Keys "W=0.0", "W=0.2", etc., each mapping to an array of shape (num_times,)
    """
    data = np.load(fid_file, allow_pickle=True)
    if "times" not in data:
        print("Error: 'times' not found.")
        return
    times = data["times"]
    w_keys = [k for k in data.keys() if k.startswith("W=")]
    
    # Define colors (matching previous aesthetic style)
    colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    lines = ["solid", "dashed", (0, (3, 1, 1, 1, 1, 1)), "dotted", "dashdot", (5, (10, 3))]


    plt.figure(figsize=(9,6))

    # Plot each disorder strength in sorted order
    for i, w_str in enumerate(sorted(w_keys, key=lambda x: float(x[2:]))):
        arr = data[w_str]
        if arr is None or arr.ndim != 1:
            print(f"Skipping {w_str}: unexpected shape {np.shape(arr)}")
            continue
        plt.plot(times, arr, linestyle=lines[i], linewidth=1.5, label=w_str, color=colors[i % len(colors)])

    # Axis labels and title
    plt.xlabel("t", fontsize=15)
    plt.ylabel("State Fidelity", fontsize=15)

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
    fid_file = rf"aggregated_1D_fidelityFINAL.npz"
    plot_1d_fidelity(fid_file)
