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
    if basexy_norm_type == "relative" or basexy_norm_type == "canonical":
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
    r_pos, r_psi, r_joints = q_world[:, 0:2], q_world[:, 2:3], q_world[:, 3:]
    h_pos = h_world[:, 0:2]

    # relative transformation
    if basexy_norm_type == "relative":
        pos_feat = normalize_relative(env, r_pos, h_pos)
        psi_feat = r_psi
    elif basexy_norm_type == "canonical":
        h_phi = h_world[:, 2:3] # target heading
        scale = env.get_robot_scale()
        # translation: delta = (x - hx, y - hy)
        delta = r_pos - h_pos
        # rotation: R(-phi) * delta
        delta_canonical = rotate_2d(delta, -h_phi)
        # scaling: delta_tilde / R
        pos_feat = delta_canonical / scale
        psi_feat = r_psi - h_phi
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")

    all_angles = np.concatenate([psi_feat, r_joints], axis=1)
    angle_feats = angles_to_cossin(all_angles) # (num_samples, num_thetas)
    return np.concatenate([pos_feat, angle_feats], axis=1).astype(np.float32)
    
def q_feat_to_world(env, q_feat: np.ndarray, h_world: np.ndarray, eps: float = 1e-8, basexy_norm_type: str = "bound") -> np.ndarray:
    """SimpleEnv: Q_feat: [N,4]=(x,y,cos,sin) -> Q: [N,3]=(x,y,theta in [0,2pi))
    RotaryLinkEnv: Q_feat: [N,8]=(x,y,cos(psi),sin(psi),cos(theta1),sin(theta1),cos(theta2),sin(theta2)) -> Q: [N,5]=(x,y,psi,theta1,theta2)"""
    q_feat = q_feat.astype(np.float32)
    # 1. Align targets to samples -> if we have multiple samples per target, repeat the anchors
    h_aligned = align_target_to_samples(h_world, q_feat.shape[0])

    # 2. Revert normalization on base (x,y)
    pos_feat = q_feat[:, 0:2]
    angles_reverted = revert_to_angle(q_feat[:, 2:], eps)

    psi_reverted = angles_reverted[:, 0:1]
    joints_reverted = angles_reverted[:, 1:]
    
    # 3. Apply reversion strategy
    if basexy_norm_type == "relative":
        h_pos = h_aligned[:, 0:2]
        r_pos = revert_relative_to_worldscale(env, pos_feat, h_pos)
        r_psi = psi_reverted
    elif basexy_norm_type == "canonical":
        h_pos = h_aligned[:, 0:2]
        h_phi = h_aligned[:, 2:3]
        scale = env.get_robot_scale()
        # 1. Un-scale: Delta_tilde = feat * R
        delta_canonical = pos_feat * scale
        # 2. Un-rotate: Delta = R(+phi) * Delta_tilde
        delta = rotate_2d(delta_canonical, h_phi)
        # 3. Un-translate: x = Delta + hx
        r_pos = delta + h_pos
        # 4. Unshift angle: psi = psi_tilde + phi
        r_psi = psi_reverted + h_phi
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")

    return np.concatenate([r_pos, r_psi, joints_reverted], axis=1).astype(np.float32)
    