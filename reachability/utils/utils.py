from __future__ import annotations
import numpy as np
import torch

def set_seed(seed: int) -> np.random.Generator:
    """Returns a numpy Generator seeded deterministically."""
    return np.random.default_rng(seed)

def print_results(name: str, results: dict) -> None:
    print(f"\n=== Results: {name} ===")
    for k, v in results.items():
        print(f"{k:30s} {v:.6f}" if isinstance(v, float) else f"{k:30s} {v}")

def normalize_relative(env, val: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Computes (val - anchor) / scale"""
    return (val - anchor) / env.get_robot_scale()

def revert_relative_to_worldscale(env, norm_val: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return norm_val * env.get_robot_scale() + anchor

def fourier_coord_feature(x: np.ndarray, B):
    x_proj = (2.0 * np.pi * x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

def align_target_to_samples(h_world: np.ndarray, num_total_samples: int) -> np.ndarray:
    """Broadcasts targets to match the number of samples. Assumes samples are generated in blocks per target."""
    num_targets = h_world.shape[0]
    if num_total_samples == num_targets:
        return h_world
    # 1. calculate how many samples (S) exist per target (B)
    S = num_total_samples // num_targets
    # 2. if we have multiple samples per target, repeat the anchors
    return np.repeat(h_world, S, axis=0)

def rotate_2d(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotates 2D vectors by specified angles (in radians).
    vectors: [N, 2], angles: [N, 1]
    Formula:    x' = x cos(theta) - y sin(theta)
                y' = x sin(theta) + y cos(theta)
    """
    x, y = vectors[:, 0:1], vectors[:, 1:2]
    c, s = np.cos(angles), np.sin(angles)
    x_rot = x * c - y * s
    y_rot = x * s + y * c
    return np.concatenate([x_rot, y_rot], axis=1)

def angles_to_cossin(angles: np.ndarray) -> np.ndarray:
    """angles: (num_samples, num_thetas) -> (num_samples, 2*num_thetas) as [cos1, sin1, cos2, sin2, ...]"""
    angles = np.asarray(angles)
    if angles.ndim == 1:
        angles = angles.reshape(-1, 1) # (N,) -> (N, 1)
    cos = np.cos(angles)
    sin = np.sin(angles)
    out = np.empty((angles.shape[0], angles.shape[1] * 2), dtype=cos.dtype)
    out[:, 0::2] = cos
    out[:, 1::2] = sin
    return out

def revert_to_angle(thetas, eps):
    """Crucial: This function only works if thetas = [cos(t1), sin(t1), cos(t2), sin(t2), ...]"""
    if thetas.shape[-1] % 2 != 0:
        raise RuntimeError("Odd number of cos(theta_i), sin(theta_i) features were inputted: must be even to de-featurize properly! Check code for errors.")
    pairs = thetas.reshape(thetas.shape[0], -1, 2) # (num_samples, num_thetas, 2)
    cos = pairs[:, :, 0]
    sin = pairs[:, :, 1]

    r = np.sqrt(cos * cos + sin * sin + eps)
    cos = cos / r
    sin = sin / r
    angle = np.arctan2(sin, cos)
    return angle

def wrap_to_2pi(theta: np.ndarray) -> np.ndarray: # TODO: deprecate in favor of wrap_to_pi
    """Wrap angles to [0, 2pi)"""
    return np.mod(theta, 2.0 * np.pi)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def sample_from_union(intervals, rng: np.random.Generator, size):
    intervals = np.array(intervals, dtype=float)
    lows = intervals[:, 0]
    highs = intervals[:, 1]
    lengths = highs - lows
    probabilities = lengths/lengths.sum()
    idx = rng.choice(len(intervals), size=size, p=probabilities)
    return rng.uniform(lows[idx], highs[idx])


def grad_global_norm(parameters) -> float:
    """Helpful debugging metric: exploding gradients -> huge norm, dead training -> ~0 norm"""
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5

def compute_bucketed_losses(losses: torch.Tensor, time_steps: torch.Tensor, total_steps: int, num_buckets: int, wandb_title: str) -> dict[str, float]:
    """Groups individual losses into buckets based on their diffusion timestep."""
    # create boundaries: e.g. [250, 500, 750] for 1000 steps and 4 buckets
    boundaries = torch.linspace(0, total_steps, num_buckets + 1).to(time_steps.device) # num_buckets + 1 because to split a range into N buckets, you need N + 1 boundary points (4 buckets over [0, 1000]: 0 | 250 | 500 | 750 | 1000)

    # identify which bucket each timestep belongs to
    bucket_indices = torch.bucketize(time_steps, boundaries[1:-1]) # 1-indexed (1 to num_buckets)

    metrics = {}
    for i in range(num_buckets):
        # print(f"losses.shape (inside of compute_bucketed_losses): {losses.shape}")
        mask = bucket_indices == i
        if mask.any():
            low = int(boundaries[i])
            high = int(boundaries[i+1]) - 1
            metrics[f"{wandb_title}/t_{low}_{high}"] = losses[mask].mean().item()
    return metrics

def compute_snr_stats(time_steps: torch.Tensor, alphas_cumprod: torch.Tensor, wandb_title: str) -> dict[str, float]:
    """SNR(t) = signal variance / noise variance = bar(alpha)_t / (1 - bar(alpha)_t)"""
    bar_alpha_t = alphas_cumprod.gather(-1, time_steps)
    # calculate SNR: alpha / (1 - alpha)
    snr = bar_alpha_t / (1 - bar_alpha_t)
    return {
        f"{wandb_title}/snr_mean": snr.mean().item(),
        f"{wandb_title}/snr_max": snr.max().item(),
        f"{wandb_title}/snr+min": snr.min().item()
    }

# DEPRECATED CODE
# def normalize_base_xy(env, x, y, basexy_norm_type: str = "bound") -> np.ndarray:
#     x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
#     if basexy_norm_type == "bound":
#         x_norm = 2 * (x - x_min)/(x_max - x_min) - 1
#         y_norm = 2 * (y - y_min)/(y_max - y_min) - 1
#     elif basexy_norm_type == "standardize":
#         x_center, y_center = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0
#         x_scale, y_scale = (x_max - x_min) / 4.0, (y_max - y_min) / 4.0
#         x_norm = (x - x_center) / x_scale # x should be [N, 1]
#         y_norm = (y - y_center) / y_scale
#     else:
#         raise ValueError(f"Unknown robot base (x,y) normalization method: {basexy_norm_type}")
#     return x_norm, y_norm

# def revert_bounded_to_worldscale(env, x, y): # will work for torch and numpy
#     x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
#     x_reverted = 0.5 * (x + 1)*(x_max - x_min) + x_min
#     y_reverted = 0.5 * (y + 1)*(y_max - y_min) + y_min
#     return x_reverted, y_reverted

# def revert_standardized_to_worldscale(env, x, y): # will work for torch and numpy
#     x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
#     x_center, y_center = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0
#     x_scale, y_scale = (x_max - x_min) / 4.0, (y_max - y_min) / 4.0
#     x_reverted = x * x_scale + x_center
#     y_reverted = y * y_scale + y_center
#     return x_reverted, y_reverted