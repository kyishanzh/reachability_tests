import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution(target, bx, by, psi, thetas, max_points=2000, tick_step=np.pi/4):
    """
    Visualizes the robot's base reachability and the internal joint manifolds.
    """
    num_samples, num_joints = thetas.shape

    # Name joints
    theta_names = [r"$\theta_{}$".format(i+1) for i in range(num_joints)]

    # Subsample down to max_points if more are provided (to avoid plot exploding)
    if num_samples > max_points:
        idx = np.random.choice(num_samples, max_points, replace=False)
        bx, by, psi, thetas = bx[idx], by[idx], psi[idx], thetas[idx]
    
    # Plot base reachability
    fig1, ax_base = plt.subplots(figsize=(7, 7))
    ax_base.scatter(bx, by, s=2, alpha=0.3, c='royalblue', label="Base $(x,y)$")
    # Quiver plot for orientation (psi); we only plot a subset for readability.
    skip = max(1, len(bx) // 100)
    ax_base.quiver(bx[::skip], by[::skip], np.cos(psi[::skip]), np.sin(psi[::skip]), color='crimson', scale=25, width=0.005, alpha=0.7, label=r"Heading $\psi$")
    # Mark the target at the origin
    ax_base.plot(*target, 'gold', marker="*", ms=15, mec='k', label=f"Target: {target}")
    ax_base.set_aspect('equal')
    ax_base.set_xlabel("World X (meters)")
    ax_base.set_ylabel("World Y (meters)")
    ax_base.set_title(f"Base distribution mapping\nTotal samples: {num_samples}")
    ax_base.grid(True, linestyle='--', alpha=0.5)
    ax_base.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

    # Plot joint manifold matrix: a grid of num_joints x num_joints plots showing every joint's relationship to every other joint
    fig2, axes = plt.subplots(num_joints, num_joints, figsize=(4*num_joints, 4*num_joints))
    if num_joints == 1: axes = np.array([[axes]]) # consistency for 1-joint robots
    # Ticks setup for angular data (-pi to pi)
    pi_ticks = np.arange(-np.pi, np.pi + tick_step, tick_step)
    k = int(round(np.pi / tick_step))
    pi_labels = [r"$0$" if n==0 else (r"$\pi$" if n==k else (r"$-\pi$" if n==-k else rf"$\frac{{{n}\pi}}{{{k}}}$")) for n in np.round(pi_ticks / tick_step).astype(int)]

    for i in range(num_joints):
        for j in range(num_joints):
            ax = axes[i, j]
            # Labeling axes
            ax.set_xticks(pi_ticks); ax.set_xticklabels(pi_labels)
            ax.set_yticks(pi_ticks); ax.set_yticklabels(pi_labels)
            ax.set_xlabel(theta_names[j])
            ax.set_ylabel(theta_names[i])

            if i == j: # Show the marginal distribution of a single joint
                ax.hist(thetas[:, i], bins=50, density=True, color='teal', alpha=0.6, edgecolor='white')
                ax.set_title(f"Distribution of {theta_names[i]}")
                ax.set_xlim(-np.pi, np.pi)
                # Y-axis on histogram is density, not radians, so we reset y-labels here
                ax.set_ylabel("Probability density")
                ax.set_yticklabels([]) # Density values are less important than the shape
            elif i < j: # Upper triangle: scatter plot showing correlation between two joints]
                ax.scatter(thetas[:, j], thetas[:, i], color='purple', alpha=0.3, s=1)
                ax.set_title(f"{theta_names[i]} vs {theta_names[j]}")
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(-np.pi, np.pi)
                ax.grid(True, alpha=0.3)
            else: # i > j is the same plots but reversed
                ax.axis('off')
            

    fig2.suptitle("Joint space manifolds", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_binding_correlation(env, q_gt, q_model, target_h, angle_start_idx=2, ncols=3):
    """
    Plots base distance r vs each angle dimension in q (from angle_start_idx onward),
    with GT as density and model as scatter, arranged in a grid with ncols columns.
    """
    # 1. Compute Base Distance to Target (r)
    gt_base_pos = q_gt[:, :2]
    mod_base_pos = q_model[:, :2]
    target_pos = target_h[:2]
    r_gt = np.linalg.norm(gt_base_pos - target_pos, axis=1)
    r_mod = np.linalg.norm(mod_base_pos - target_pos, axis=1)

    # 2. Compute Arm Extension Proxy 
    gt_angles = q_gt[:, angle_start_idx:] # (N, A)
    mod_angles = q_model[:, angle_start_idx:]  # (N, A)
    A = gt_angles.shape[1]

    names = ["psi"] + [f"theta_{k+1}" for k in range(A-1)]

    nrows = int(np.ceil(A / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    for k in range(A):
        i, j = divmod(k, ncols)
        ax = axes[i, j]
        ext_gt = gt_angles[:, k]
        ext_mod = mod_angles[:, k]
        
        sns.kdeplot(x=r_gt, y=ext_gt, fill=True, cmap="Blues", alpha=0.5, levels=5, label="Ground truth density", ax=ax)
    
        # Scatter Model points on top
        ax.scatter(r_mod, ext_mod, s=5, c='crimson', alpha=0.5, label="Model samples")
        
        ax.set_xlabel("Base distance to target (m)")
        ax.set_ylabel(f"{names[k]} (rad)")
        ax.set_title(f"r vs {names[k]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    # turn off unused axes
    for k in range(A, nrows*ncols):
        i, j = divmod(k, ncols)
        axes[i, j].axis("off")

    fig.suptitle(f"Kinematic binding: Base radius vs angles\n(Target: {target_h})", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_density_difference(q_gt, q_model):
    """
    Plots P_model(x,y) - P_gt(x,y) to reveal mode collapse/dropping.
    bounds: [xmin, xmax, ymin, ymax]
    """
    # Bin the Base X/Y data
    nbins = 50
    xmin = min(np.min(q_gt[:, 0]), np.min(q_model[:, 0]))
    xmax = max(np.max(q_gt[:, 0]), np.max(q_model[:, 0]))
    ymin = min(np.min(q_gt[:, 1]), np.min(q_model[:, 1]))
    ymax = max(np.max(q_gt[:, 1]), np.max(q_model[:, 1]))
    bounds = [xmin, xmax, ymin, ymax]
    range_bins = [[bounds[0], bounds[1]], [bounds[2], bounds[3]]]
    
    # Compute 2D histograms (Probability Density)
    H_gt, xedges, yedges = np.histogram2d(q_gt[:,0], q_gt[:,1], bins=nbins, range=range_bins, density=True)
    H_mod, _, _ = np.histogram2d(q_model[:,0], q_model[:,1], bins=nbins, range=range_bins, density=True)
    
    # Difference Map
    diff = H_mod - H_gt
    
    fig, ax = plt.subplots(figsize=(7, 6))
    # Diverging colormap: Red = Model High, Blue = GT High
    im = ax.imshow(diff.T, origin='lower', extent=[bounds[0], bounds[1], bounds[2], bounds[3]], 
                   cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    
    plt.colorbar(im, label="Density difference (Model - GT)")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_title("Where is the model cheating?\n(Red = oversampled, Blue = ignored)")
    plt.show()

def plot_marginal_comparisons(q_gt, q_model, cols=2, nbins=50):
    """
    Plots marginal histograms for all dimensions of the state space.
    Arranges them in a 2 column for easy comparison.
    """
    assert q_gt.shape[1] == q_model.shape[1], "q_gt and q_model must have same number of dims"
    d_q = q_gt.shape[1]

    # Build labels
    dim_names = [r"Base position $x$", r"Base position $y$", r"Base heading $\psi$"]
    theta_start = 3 # true for all rotaryNLink environments TODO make general later for more environments
    n_thetas = max(0, d_q - theta_start)
    dim_names += [fr"Joint $\theta_{i}$" for i in range(1, n_thetas + 1)]

    # Figure layout
    rows = math.ceil(d_q / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 3.8*rows))
    axes = np.array(axes).reshape(-1) # flatten to be sure

    for i in range(d_q):
        ax = axes[i]
        
        # Determine shared bin range to ensure alignment
        data_min = min(q_gt[:, i].min(), q_model[:, i].min())
        data_max = max(q_gt[:, i].max(), q_model[:, i].max())
        # avoid degenerate bins if constant
        if np.isclose(data_min, data_max):
            data_min -= 1e-3
            data_max += 1e-3

        bins = np.linspace(data_min, data_max, 50)
        
        # Plot Ground Truth (Blue)
        ax.hist(q_gt[:, i], bins=bins, density=True, 
                color='royalblue', alpha=0.6, label='Ground Truth' if i == 0 else "")
        
        # Plot Model (Red)
        ax.hist(q_model[:, i], bins=bins, density=True, 
                color='crimson', alpha=0.5, label='Model' if i == 0 else "")
        
        # Formatting
        ax.set_title(dim_names[i])
        ax.set_yticks([]) # Hide density ticks for cleaner look
        ax.grid(True, alpha=0.3)

    # turn off unused axes
    for i in range(d_q, len(axes)):
        axes[i].axis("off")

    # Add a single legend to the figure
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Marginal Distribution Comparison (Model vs. Ground Truth)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Make room for suptitle
    plt.show()
