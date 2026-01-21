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

def revert_to_angle(cos, sin, eps):
    r = np.sqrt(cos * cos + sin * sin + eps)
    cos = cos / r
    sin = sin / r
    angle = np.arctan2(sin, cos)
    return np.mod(angle, 2.0 * np.pi)

def wrap_to_2pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 2pi)"""
    return np.mod(theta, 2.0 * np.pi)

def grad_global_norm(parameters) -> float:
    """Helpful debugging metric: exploding gradients -> huge norm, dead training -> ~0 norm"""
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


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