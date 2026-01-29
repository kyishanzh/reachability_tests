import numpy as np
import matplotlib.pyplot as plt

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
