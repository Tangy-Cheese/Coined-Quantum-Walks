import numpy as np
import matplotlib.pyplot as plt

def plot_1d_entanglement(ent_file):
    """
    Loads the aggregated 1D entanglement file and plots entanglement entropy vs. time.
    Expects:
      - "times": array of shape (num_times,)
      - Keys "W=0.0", "W=0.2", etc., each mapping to an array of shape (num_times,)
    """
    data = np.load(ent_file, allow_pickle=True)
    if "times" not in data:
        print("Error: 'times' not found.")
        return
    times = data["times"]
    w_keys = [k for k in data.keys() if k.startswith("W=")]
    
    # Define colors (matching previous aesthetic style)
    colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    #lines = ["solid", "dashed", (0, (3, 1, 1, 1, 1, 1)), "dotted", "dashdot", (5, (10, 3))]

    plt.figure(figsize=(6,6))

    # Plot each disorder strength in sorted order
    for i, w_str in enumerate(sorted(w_keys, key=lambda x: float(x[2:]))):
        arr = data[w_str]
        if arr is None or arr.ndim != 1:
            print(f"Skipping {w_str}: unexpected shape {np.shape(arr)}")
            continue
        plt.plot(times, arr, linestyle="solid", linewidth=3, label=w_str, color=colors[i % len(colors)])

    # Axis labels and title
    plt.xlabel("t", fontsize=16)
    plt.ylabel("S(t)", fontsize=16)
    plt.xlim(50, 100)
    plt.ylim(1.1, 1.4)
    
    
    plt.title("(b)", fontsize=16)
    # Legend
    #plt.legend(fontsize=15)

    # No gridlines
    plt.grid(False)

    # Format ticks: inward-facing on all four sides
    plt.tick_params(direction='in', length=6, width=1.2, labelsize=16, 
                    bottom=True, top=True, left=True, right=True)  # Major ticks on all sides

    # Set minor ticks
    plt.minorticks_on()
    plt.tick_params(which='minor', direction='in', length=3, width=0.8, 
                    bottom=True, top=True, left=True, right=True)  # Minor ticks on all sides

    # Tight layout
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    ent_file = rf"aggregated_2D_entanglementFINAL.npz"
    plot_1d_entanglement(ent_file)
