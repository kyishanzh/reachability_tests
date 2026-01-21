import torch
import numpy as np
from typing import Any
from reachability.utils.utils import *

def c_world_to_feat(
    env,
    c_world: np.ndarray,
    basexy_norm_type: str = "relative",
    add_fourier_feat: bool = False,
    fourier_B: Any = None
) -> np.ndarray:
    if basexy_norm_type == "relative":
        # in relative frame, the target is at (0, 0) by definition 1/R*(hx - hx, hy - hy) -> return [0, 0]
        return np.zeros_like(c_world).astype(np.float32)
    # DEPRECATED: basexy_norm_type == "relative" or "bound"
    # hx_norm, hy_norm = normalize_base_xy(env, c_world[:, 0:1], c_world[:, 1:2], basexy_norm_type=basexy_norm_type)
    # h_feat = np.concat([hx_norm, hy_norm], axis=1).astype(np.float32)
    # if add_fourier_feat:
    #     return fourier_coord_feature(h_feat, fourier_B).astype(np.float32)
    # return h_feat

def q_world_to_feat(env, q_world: np.ndarray, h_world: np.ndarray, basexy_norm_type: str = "relative") -> np.ndarray:
    """SimpleEnv: Q: [N,3]=(x,y,theta) -> Q_feat: [N,4]=(x,y,cos, sin)
    RotaryLinkEnv: Q: [N,5]=(x,y,psi,theta1,theta2) -> Q_feat: [N,8]=(x,y,cos(psi),sin(psi),cos(theta1),sin(theta1),cos(theta2),sin(theta2))"""

    q_world = q_world.astype(np.float32)

    # base positions (x,y)
    x, y = q_world[:, 0:1], q_world[:, 1:2]

    # relative transformation
    if basexy_norm_type == "relative":
        hx, hy = h_world[:, 0:1], h_world[:, 1:2]
        x_norm = normalize_relative(env, x, hx)
        y_norm = normalize_relative(env, y, hy)
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")

    if env.name == 'Simple':
        return np.concatenate(
            [x_norm, y_norm, np.cos(q_world[:, 2:3]), np.sin(q_world[:, 2:3])],
            axis=1
        ).astype(np.float32)
    elif env.name == 'RotaryLink':
        return np.concatenate(
            [x_norm, y_norm,
            np.cos(q_world[:, 2:3]), np.sin(q_world[:, 2:3]),
            np.cos(q_world[:, 3:4]), np.sin(q_world[:, 3:4]),
            np.cos(q_world[:, 4:5]), np.sin(q_world[:, 4:5])],
            axis=1
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown environment, env.name = {env.name}")

def q_feat_to_world(env, q_feat: np.ndarray, h_world: np.ndarray, eps: float = 1e-8, basexy_norm_type: str = "bound") -> np.ndarray:
    """SimpleEnv: Q_feat: [N,4]=(x,y,cos,sin) -> Q: [N,3]=(x,y,theta in [0,2pi))
    RotaryLinkEnv: Q_feat: [N,8]=(x,y,cos(psi),sin(psi),cos(theta1),sin(theta1),cos(theta2),sin(theta2)) -> Q: [N,5]=(x,y,psi,theta1,theta2)"""
    q_feat = q_feat.astype(np.float32)
    # revert normalization on base (x,y)
    x_feat, y_feat = q_feat[:, 0:1], q_feat[:, 1:2]

    if basexy_norm_type == "relative":
        scale = env.get_robot_scale()
        # 1. calculate how many samples (S) exist per target (B)
        num_total_samples = q_feat.shape[0]
        num_targets = h_world.shape[0]
        S = num_total_samples // num_targets
        hx, hy = h_world[:, 0:1], h_world[:, 1:2]
        # 2. if we have multiple samples per target, repeat the anchors
        if S > 1:
            # np.repeat turns [h1, h2] into [h1, h1, h2, h2] if S = 2
            hx = np.repeat(hx, S, axis=0)
            hy = np.repeat(hy, S, axis=0)
        x_reverted = revert_relative_to_worldscale(env, x_feat, hx)
        y_reverted = revert_relative_to_worldscale(env, y_feat, hy)
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")

    if env.name == 'Simple':
        theta = revert_to_angle(q_feat[:, 2:3], q_feat[:, 3:4], eps)
        return np.concatenate([x_reverted, y_reverted, theta], axis=1).astype(np.float32)
    elif env.name == "RotaryLink":
        psi = revert_to_angle(q_feat[:, 2:3], q_feat[:, 3:4], eps)
        theta1 = revert_to_angle(q_feat[:, 4:5], q_feat[:, 5:6], eps)
        theta2 = revert_to_angle(q_feat[:, 6:7], q_feat[:, 7:8], eps)
        return np.concatenate([x_reverted, y_reverted, wrap_to_2pi(psi), wrap_to_2pi(theta1), wrap_to_2pi(theta2)], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown environment, env.name = {env.name}")