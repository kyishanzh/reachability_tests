from __future__ import annotations
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

# S = number samples
# read into the math in this file a lot more carefully

def hand_error(env, q_samples: np.ndarray, h_world: np.ndarray) -> np.ndarray:
    """
    ||FK(Q) - H||
    Computes decoupled position and orientation errors. 
    Returns dictionary with keys 'pos_err' (metersr) and 'ori_err' (radians)
    """
    B, S, _ = q_samples.shape
    world_coord = h_world.shape[-1]
    q_flattened = q_samples.reshape(B * S, -1)  # [B, S, dQ] -> [B * S, dQ]
    h_world_repeated = h_world[:, None, :]  # [B, 2] -> [B,1,2]

    # Compute FK -> [x, y, phi]
    hand = env.fk_hand(q_flattened).reshape(B, S, world_coord)  # [B * S, 2] -> [B, S, 2]
    
    # Hand position error
    pos_pred = hand[:, :, :2]
    pos_gt = h_world_repeated[:, :, :2]
    pos_err = np.linalg.norm(pos_pred - pos_gt, axis=-1)

    if env.name == "RotaryNLink":
        # Hand orientation error
        phi_pred = hand[:, :, 2]
        phi_gt = h_world_repeated[:, :, 2]
        diff = phi_pred - phi_gt
        ori_err = np.abs(np.arctan2(np.sin(diff), np.cos(diff))) # smallest angle difference: |atan2 sin(diff), np.cos(diff)|
        return {
            "pos_err": pos_err.astype(np.float32),
            "ori_err": ori_err.astype(np.float32)
        }
    return {"pos_err": pos_err.astype(np.float32), "ori_err": None}

def compute_validity_rates(env, q_samples: np.ndarray) -> dict:
    """
    Computes physical feasibility metrics:
    1. Self-collision rate (percet of samples colliding)
    2. Joint limit violation rate (percent of samples out of bounds)
    """
    B, S, d_q = q_samples.shape
    q_flat = q_samples.reshape(B * S, d_q)

    # 1. Self collision
    collisions = env.check_self_collision(q_flat)
    coll_rate = np.mean(collisions)

    # 2. Joint limits
    limit_violations = np.zeros(B * S, dtype=bool)
    if env.joint_limits is not None:
        thetas = q_flat[:, 3:]
        for i, limits in enumerate(env.joint_limits):
            intervals = np.array(limits, dtype=np.float32)
            if intervals.ndim == 1: # if provided single [min, max] instead of [[min, max]]
                intervals = intervals.reshape(1, 2)
            
            # A joint angle is valid if it falls inside *any* of the allowed intervals
            joint_is_valid = np.zeros(B * S, dtype=bool) # assume all invalid
            joint_vals = thetas[:, i]
            for (low, high) in intervals:
                in_this_interval = (joint_vals >= low) & (joint_vals <= high)
                joint_is_valid |= in_this_interval
            
            # If a sample is valid in 0 intervals, it's a violation
            limit_violations |= (~joint_is_valid)
    limit_rate = np.mean(limit_violations)

    return {
        "collision_rate": float(coll_rate),
        "limit_rate": float(limit_rate)
    }

def compute_scale_vect(env, d_q):
    w = env.workspace
    pos_scale = max(w.hx_max - w.hx_min, w.hy_max - w.hy_min)
    angle_scale = 2 * np.pi

    # Create scale vector [pos, pos, angle, angle...]
    scale = np.ones(d_q, dtype=np.float32)
    scale[0:2] = pos_scale
    scale[2:] = angle_scale
    return scale

def embed_q(env, q):
    w = env.workspace
    pos_scale = max(w.hx_max - w.hx_min, w.hy_max - w.hy_min)
    pos = q[..., :2] / pos_scale
    angles = q[..., 2:]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    angle_feats = np.concatenate([cos_a, sin_a], axis=-1) * 0.5 # so max distance becomes 1.0 -> balances weight with the position variables
    return np.concatenate([pos, angle_feats], axis=-1)

def compute_coverage_fidelity(env, q_pred: np.ndarray, q_gt: np.ndarray) -> tuple[float, float]:
    """
    Manifold metrics:
    - Coverage (recall): Mean dist from ground truth samples to nearest generated sample.
    - Fidelity (precision): Mean dist from generated samples to nearest ground truth sample.
    Input shapes: [num_samples, d_q] (for a single H)
    """
    # TODO: figure out why this scaling was problematic
    # Create scale vector [pos, pos, angle, angle...]
    # d_q = q_pred.shape[-1]
    # scale = compute_scale_vect(env, d_q)

    # # Normalize
    # q_pred_norm = q_pred / scale
    # q_gt_norm = q_gt / scale
    q_pred_emb = embed_q(env, q_pred)
    q_gt_emb = embed_q(env, q_gt)
    
    # Compute pairwise distance matrix (normalized) [num_samples_pred, num_samples_gt]
    dists = cdist(q_pred_emb, q_gt_emb, metric='euclidean')

    # Coverage: For every GT (col), find closest prediction (row)
    coverage = np.mean(np.min(dists, axis=0)) # min over rows (axis=0) -> [num_samples_gt] -> mean

    # Fidelity: For every prediction (row), find closest GT (col)
    fidelity = np.mean(np.min(dists, axis=1)) # min over cols (axis=1) -> [num_samples_pred] -> mean

    return float(coverage), float(fidelity)

def compute_emd_circular(angles_pred: np.ndarray, angles_gt: np.ndarray) -> float:
    """
    Wasserstein-1 distance on the circle S1.
    Note: Since optimal transport on a circle is hard (requires cutting), we use a robust approximation:
        { Try shifting one distribution by {0, 90, 180, 270} degrees and take the minimum linear Wasserstein distance. This prevents edge-cases where a distribution split across the 0/2pi cut looks "far apart" in linear space. }
    """
    # Standardize to [0, 2pi)
    a = np.mod(angles_pred, 2 * np.pi)
    b = np.mod(angles_gt, 2 * np.pi)

    # Compute EMD for different circular shifts of B to align the "cut"
    shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    scores = []

    for shift in shifts:
        # Shift b, wrap, then compute linear EMD
        b_shifted = np.mod(b + shift, 2 * np.pi)
        scores.append(wasserstein_distance(a, b_shifted))

    return float(min(scores))

def implied_angles(env, q_samples: np.ndarray, h_world: np.ndarray) -> np.ndarray:
    """Returns implied theta angles in [0, 2pi): shape [B, S]"""
    B, S, _ = q_samples.shape
    q_flattened = q_samples.reshape(B * S, -1)
    h_world_repeated = np.repeat(h_world, repeats=S, axis=0) # [B*S, 2]
    th = env.target_bearing_world(q_flattened, h_world_repeated).reshape(B, S)
    return th.astype(np.float32)

def kl_to_uniform(theta: np.ndarray, n_bins: int, eps: float = 1e-9) -> float:
    """theta: [M] angles in [0, 2pi)
    KL( p_hat || uniform )
    """
    hist, _ = np.histogram(theta, bins=n_bins, range=(0, 2 * np.pi), density=False)
    p = hist.astype(np.float64) + eps
    p = p / np.sum(p)
    u = np.ones_like(p) / p.shape[0]
    kl = float(np.sum(p * (np.log(p) - np.log(u))))
    return kl

def max_angle_gap(theta: np.ndarray) -> float:
    """theta: [M] angles in [0, 2pi)
    returns max gap on circle (radians)"""
    if theta.size == 0:
        return float("nan")
    th = np.sort(theta.astype(np.float64))
    diffs = np.diff(th)
    wrap_gap = (th[0] + 2.0 * np.pi) - th[-1]
    return float(max(np.max(diffs) if diffs.size else 0, wrap_gap))

def var_Q(q_samples: np.ndarray) -> float:
    """Q_samples: [S, dQ] for a fixed H"""
    return float(np.mean(np.var(q_samples.astype(np.float64), axis=0)))

def compute_mmd(env, samples_pred, samples_gt, sigmas=[0.01, 0.05, 0.1, 0.5, 1.0]):
    """MMD with multi-scale RBF kernel"""
    # x, y = torch.as_tensor(samples_pred), torch.as_tensor(samples_gt)

    # TODO: figure out why this scaling was problematic
    # scale = compute_scale_vect(env, samples_pred.shape[-1])
    # scale_tensor = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    # x = x / scale_tensor
    # y = y / scale_tensor
    x = torch.as_tensor(embed_q(env, samples_pred), dtype=torch.float32)
    y = torch.as_tensor(embed_q(env, samples_gt), dtype=torch.float32)

    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b).pow(2)
        return torch.exp(-dist / (2 * sigma**2))

    mmd_sq = 0
    for s in sigmas:
        k_xx = rbf_kernel(x, x, s).mean()
        k_yy = rbf_kernel(y, y, s).mean()
        k_xy = rbf_kernel(x, y, s).mean()
        mmd_sq += (k_xx + k_yy - 2 * k_xy)
    return torch.sqrt(torch.clamp(mmd_sq, min=0)).item()
