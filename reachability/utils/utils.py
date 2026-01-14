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

def normalize_base_xy(env, x, y, basexy_norm_type: str = "bound") -> np.ndarray:
    x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
    if basexy_norm_type == "bound":
        x_norm = 2 * (x - x_min)/(x_max - x_min) - 1
        y_norm = 2 * (y - y_min)/(y_max - y_min) - 1
    elif basexy_norm_type == "standardize":
        x_center, y_center = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0
        x_scale, y_scale = (x_max - x_min) / 4.0, (y_max - y_min) / 4.0
        x_norm = (x - x_center) / x_scale # x should be [N, 1]
        y_norm = (y - y_center) / y_scale
    else:
        raise ValueError(f"Unknown robot base (x,y) normalization method: {basexy_norm_type}")
    return x_norm, y_norm

def h_to_hnorm(env, H: np.ndarray, basexy_norm_type: str = "bound") -> np.ndarray:
    hx_norm, hy_norm = normalize_base_xy(env, H[:, 0:1], H[:, 1:2], basexy_norm_type=basexy_norm_type)
    # print("hx_norm shape: ", hx_norm.shape)
    return np.concat([hx_norm, hy_norm], axis=1).astype(np.float32)

def q_to_qfeat(env, Q: np.ndarray, basexy_norm_type: str = "bound") -> np.ndarray:
    """SimpleEnv: Q: [N,3]=(x,y,theta) -> Q_feat: [N,4]=(x,y,cos, sin)
    RotaryLinkEnv: Q: [N,5]=(x,y,psi,theta1,theta2) -> Q_feat: [N,8]=(x,y,cos(psi),sin(psi),cos(theta1),sin(theta1),cos(theta2),sin(theta2))
    Two normalization methods for base (x,y) coordinate normalization:
    (1) "bound": Normalizing x, y to [-1, 1] using workspace bounds.
    (2) "standardize": Affine normalization that maps workspace bounds to a fixed "sigma-like" range"""

    Q = Q.astype(np.float32)
    x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
    # normalize base positions (x,y)
    x, y = Q[:, 0:1], Q[:, 1:2]
    x_norm, y_norm = normalize_base_xy(env, x, y, basexy_norm_type=basexy_norm_type)

    if env.name == 'Simple':
        return np.concatenate(
            [x_norm, y_norm, np.cos(Q[:, 2:3]), np.sin(Q[:, 2:3])],
            axis=1
        ).astype(np.float32)
    elif env.name == 'RotaryLink':
        return np.concatenate(
            [x_norm, y_norm,
            np.cos(Q[:, 2:3]), np.sin(Q[:, 2:3]),
            np.cos(Q[:, 3:4]), np.sin(Q[:, 3:4]),
            np.cos(Q[:, 4:5]), np.sin(Q[:, 4:5])],
            axis=1
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown environment, env.name = {env.name}")

def revert_to_angle(cos, sin, eps):
    r = np.sqrt(cos * cos + sin * sin + eps)
    cos = cos / r
    sin = sin / r
    angle = np.arctan2(sin, cos)
    return np.mod(angle, 2.0 * np.pi)

def revert_bounded_to_worldscale(env, x, y):
    x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
    x_reverted = 0.5 * (x + 1)*(x_max - x_min) + x_min
    y_reverted = 0.5 * (y + 1)*(y_max - y_min) + y_min
    return x_reverted, y_reverted

def revert_standardized_to_worldscale(env, x, y):
    x_min, x_max, y_min, y_max = env.workspace.hx_min, env.workspace.hx_max, env.workspace.hy_min, env.workspace.hy_max
    x_center, y_center = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0
    x_scale, y_scale = (x_max - x_min) / 4.0, (y_max - y_min) / 4.0
    x_reverted = x * x_scale + x_center
    y_reverted = y * y_scale + y_center
    return x_reverted, y_reverted
    
def qfeat_to_q(env, Q_feat: np.ndarray, eps: float = 1e-8, basexy_norm_type: str = "bound") -> np.ndarray:
    """SimpleEnv: Q_feat: [N,4]=(x,y,cos,sin) -> Q: [N,3]=(x,y,theta in [0,2pi))
    RotaryLinkEnv: Q_feat: [N,8]=(x,y,cos(psi),sin(psi),cos(theta1),sin(theta1),cos(theta2),sin(theta2)) -> Q: [N,5]=(x,y,psi,theta1,theta2)"""
    Q_feat = Q_feat.astype(np.float32)
    # revert normalization on base (x,y)
    if basexy_norm_type == "bound":
        x_reverted, y_reverted = revert_bounded_to_worldscale(env, Q_feat[:, 0:1], Q_feat[:, 1:2])
    elif basexy_norm_type == "standardize":
        x_reverted, y_reverted = revert_standardized_to_worldscale(env, Q_feat[:, 0:1], Q_feat[:, 1:2])
    else:
        raise ValueError(f"Unknown robot base (x,y) normalization method: {basexy_norm_type}")

    if env.name == 'Simple':
        theta = revert_to_angle(Q_feat[:, 2:3], Q_feat[:, 3:4], eps)
        return np.concatenate([x_reverted, y_reverted, theta], axis=1).astype(np.float32)
    elif env.name == "RotaryLink":
        psi = revert_to_angle(Q_feat[:, 2:3], Q_feat[:, 3:4], eps)
        theta1 = revert_to_angle(Q_feat[:, 4:5], Q_feat[:, 5:6], eps)
        theta2 = revert_to_angle(Q_feat[:, 6:7], Q_feat[:, 7:8], eps)
        return np.concatenate([x_reverted, y_reverted, psi, theta1, theta2], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown environment, env.name = {env.name}")


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