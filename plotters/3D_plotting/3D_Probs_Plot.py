import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_z0_heatmap_custom(prob_file, W_target="W=0.0", t_target=100, prob_threshold=1e-5):
    """
    Loads the aggregated 3D probability file and plots a **2D heatmap** 
    of the probability distribution at the `z = 0` plane.

    - Filters points with probability < `prob_threshold` for clarity.
    - Uses the **original custom colormap** (white → green → yellow → orange → red).
    
    Arguments:
        prob_file (str): Path to the .npz file storing probability data.
        W_target (str): Disorder strength key (e.g., "W=0.2").
        t_target (int): Time step to visualize.
        prob_threshold (float): Minimum probability to include in the plot.
    """
    data = np.load(prob_file, allow_pickle=True)
    
    # Fetch disorder-specific time array
    times_key = f"{W_target}_times"
    if times_key not in data:
        print(f"Error: Expected key {times_key} not found. Available keys: {list(data.keys())}")
        return
    times = data[times_key]  # Get correct time steps

    # Check if t_target is present
    if t_target not in times:
        print(f"Error: t_target {t_target} not found in times array: {times}")
        return
    t_index = np.where(times == t_target)[0][0]  # Find index for requested time

    # Fetch probability distribution for given W_target
    if W_target not in data:
        print(f"Error: key {W_target} not found in data.")
        return
    prob_3d = data[W_target]  # shape (num_times, Nx, Nx, Nx)
    full_prob = prob_3d[t_index]  # Extract probability distribution at t_target

    # Get grid size and center coordinates
    Nx = full_prob.shape[0]
    c = Nx // 2  # Center coordinate
    axis_range = np.arange(-c, c + 1)  # X, Y range

    # Extract the slice where z = 0
    z_center_idx = c  # The center index corresponds to z = 0
    prob_z0_plane = full_prob[:, :, z_center_idx]  # Extract slice at z = 0

    # Create the original custom colormap
    colors = ["white", "royalblue","rebeccapurple","darkviolet", "red"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    # Create 2D heatmap plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(prob_z0_plane.T, extent=[-c, c, -c, c], origin="lower", cmap=custom_cmap, aspect="auto")
    
    # Set font size
    font = 23

    # Add custom colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("P(x, y, z=0)", fontsize=font)  # Same fontsize as labels
    cbar.ax.tick_params(labelsize=font)  # Match tick labels to fontsize

    # **Ensure scientific notation exponent uses the same font size**
    cbar.formatter.set_powerlimits((0, 0))  # Force scientific notation when needed
    cbar.ax.yaxis.get_offset_text().set_fontsize(font)  # Set exponent font size

    # Set custom ticks: markers at every 20, labels every 40
    ticks = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
    tick_labels = [str(tick) if tick % 40 == 0 else "" for tick in ticks]
    plt.xticks(ticks, tick_labels, fontsize=font)
    plt.yticks(ticks, tick_labels, fontsize=font)
    plt.subplots_adjust(bottom=0.15)
    # Labels and title
    plt.title("(c)", fontsize=font)
    plt.xlabel("x", fontsize=font)
    plt.ylabel("y", fontsize=font)
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)

    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    prob_file = "aggregated_3D_probabilityFINAL_Proper.npz"
    plot_z0_heatmap_custom(prob_file, W_target="W=1.0", t_target=100, prob_threshold=0.0)
