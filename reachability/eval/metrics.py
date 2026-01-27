from __future__ import annotations
import numpy as np
import torch
from scipy.stats import wasserstein_distance

# S = number samples
# read into the math in this file a lot more carefully

def hand_error(env, q_samples: np.ndarray, h_world: np.ndarray) -> np.ndarray:
    """||FK(Q) - H||"""
    B, S, _ = q_samples.shape
    q_flattened = q_samples.reshape(B * S, -1)  # [B, S, dQ] -> [B * S, dQ]
    hand = env.fk_hand(q_flattened).reshape(B, S, 2)  # [B * S, 2] -> [B, S, 2]
    h_world_repeated = h_world[:, None, :]  # [B, 2] -> [B,1,2]
    err = np.linalg.norm(hand - h_world_repeated, axis=-1)  #[B, S]
    return err.astype(np.float32)

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

def compute_mmd(samples_pred, samples_gt, sigmas=[0.1, 0.5, 1.0]):
    """MMD with multi-scale RBF kernel"""
    x, y = torch.as_tensor(samples_pred), torch.as_tensor(samples_gt)

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

def compute_emd_1d(angles_pred, angles_gt):
    """Wasserstein distance for circular distributions"""
    return wasserstein_distance(angles_pred, angles_gt)
