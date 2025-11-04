import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_2d_probability(prob_file, W_target="W=0.0", t_target=100):
    """
    Loads the aggregated 2D probability file and plots a **2D heatmap** 
    of the probability distribution at a specified time t_target for a given disorder strength W_target.

    Expects:
      - "times": array of shape (num_times,)
      - Keys "W=0.0", etc., each mapping to an array of shape (num_times, Nx, Nx).
    
    Ensures:
      - Uniform **custom colormap** (white → green → yellow → orange → red).
      - **Consistent font sizes** across title, labels, and colorbar.
      - **Axis ticks every 20, labels every 40** for clarity.
      - **Scientific notation exponent in colorbar matches font size.**
      - **Restricted plot range to [-80, 80]** for both x and y.
    """
    data = np.load(prob_file, allow_pickle=True)
    
    # Check for necessary keys
    if "times" not in data:
        print("Error: 'times' not found in the file.")
        return
    times = data["times"]
    if t_target not in times:
        print(f"Error: t_target={t_target} not in times: {times}")
        return
    t_index = np.where(times == t_target)[0][0]
    if W_target not in data:
        print(f"Error: key {W_target} not found.")
        return
    
    # Extract 2D probability distribution for selected W and time
    prob_2d = data[W_target]  # shape (num_times, Nx, Nx)
    heatmap = prob_2d[t_index]
    Nx = heatmap.shape[0]

    # Compute global min/max across all W values for consistent colorbar scaling
    global_min = np.min([np.min(data[w][t_index]) for w in data.keys() if w.startswith("W=")])
    global_max = np.max([np.max(data[w][t_index]) for w in data.keys() if w.startswith("W=")])

    # Create a continuous colormap using smooth interpolation
    colors = ["white", "green", "yellow", "orange", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Set axis scale
    if Nx == 2*t_target + 1:
        axis = np.arange(-t_target, t_target + 1)
    else:
        axis = np.arange(Nx)

    # Create figure and plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap, extent=[axis[0], axis[-1], axis[0], axis[-1]], 
                    origin="lower", cmap=cmap, interpolation="nearest")

    # Restrict display range to [-80, 80] in both x and y
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)

    # Set font size
    font = 23

    # Set custom ticks: Markers every 20 but labels every 40
    ticks = np.arange(-80, 81, 20)
    tick_labels = [str(tick) if tick % 40 == 0 else "" for tick in ticks]
    plt.xticks(ticks, tick_labels, fontsize=font)
    plt.yticks(ticks, tick_labels, fontsize=font)

    # Add colorbar with consistent formatting
    cbar = plt.colorbar(im)
    cbar.set_label("P(x,y)", fontsize=font)
    cbar.ax.tick_params(labelsize=font)

    # Ensure scientific notation exponent uses the same font size
    cbar.formatter.set_powerlimits((0, 0))  # Force scientific notation when needed
    cbar.ax.yaxis.get_offset_text().set_fontsize(font)  # Set exponent font size

    # Labels, title, and tick formatting
    plt.xlabel("x", fontsize=font)
    plt.ylabel("y", fontsize=font)
    plt.title("(b)", fontsize=font)

    # Inward-facing ticks (matches previous plots)
    plt.tick_params(direction="in", length=6, width=1.2, labelsize=font, 
                    bottom=True, top=True, left=True, right=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prob_file = r"aggregated_2D_probabilityFINAL _0.3et_al.npz"
    plot_2d_probability(prob_file, W_target="W=0.3", t_target=100)
