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

def q_to_qfeat_np(Q: np.ndarray) -> np.ndarray:
    """Q: [N,3]=(x,y,theta) -> Q_feat: [N,4]=(x,y,cos, sin)"""
    Q = Q.astype(np.float32)
    return np.concatenate(
        [Q[:, 0:1], Q[:, 1:2], np.cos(Q[:, 2:3]), np.sin(Q[:, 2:3])],
        axis=1
    ).astype(np.float32)

def qfeat_to_q_np(Q_feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Q_feat: [N,4]=(x,y,cos,sin) -> Q: [N,3]=(x,y,theta in [0,2pi))"""
    Q_feat = Q_feat.astype(np.float32)
    cos = Q_feat[:, 2:3]
    sin = Q_feat[:, 3:4]

    r = np.sqrt(cos * cos + sin * sin + eps)
    cos = cos / r
    sin = sin / r

    theta = np.arctan2(sin, cos)
    theta = np.mod(theta, 2.0 * np.pi)
    return np.concatenate([Q_feat[:, 0:1], Q_feat[:, 1:2], theta], axis=1).astype(np.float32)

def q_to_qfeat_torch(Q: torch.Tensor) -> torch.Tensor:
    """Q: [N, 3] = (x, y, theta) -> Q_feat: [N, 4] = (x, y, cos theta, sin theta)"""
    th = Q[:, 2:3]
    return torch.cat([Q[:, 0:1], Q[:, 1:2], torch.cos(th), torch.sin(th)], dim=1)

def qfeat_to_q_torch(Q_feat: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    """Q_feat: [N, 4] = (x, y, cos theta, sin theta) -> Q: [N, 3] = (x, y, theta)"""
    x = Q_feat[:, 0:1]
    y = Q_feat[:, 1:2]
    cos = Q_feat[:, 2:3]
    sin = Q_feat[:, 3:4]

    # normalizing
    r = torch.sqrt(cos * cos + sin * sin + eps)
    cos = cos / r
    sin = sin / r

    th = torch.atan2(sin, cos)
    th = torch.remainder(th, 2.0 * np.pi)
    return torch.cat([x, y, th], dim=1)
