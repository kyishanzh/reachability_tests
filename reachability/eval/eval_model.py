from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from reachability.eval.metrics import *

@dataclass
class EvalConfig:
    n_samples_per_h: int
    n_bins_theta: int # how finely we discretize angle space when estimatig distributions
    eps_hist: float # for KL div - prevents log(0)
    sampling_temperature: float = 0.0

def evaluate_model(env, model, h_world_test: np.ndarray, c_world_test: np.ndarray, cfg: EvalConfig, rng: np.random.Generator) -> dict:
    """Evaluate on test H points:
    - Draw S samples per h in h_world_test
    - Compute error/coverage stats"""
    S = cfg.n_samples_per_h
    q_samples = model.sample(h_world=h_world_test, c_world=c_world_test, n_samples=S, rng=rng, sampling_temperature=cfg.sampling_temperature)  # [B,S,d_q]
    
    err = hand_error(env, q_samples, h_world_test) # [B,S]
    err_mean = float(np.mean(err))
    err_median = float(np.median(err))
    err_p95 = float(np.quantile(err, 0.95))

    base_heading_angles = implied_angles(env, q_samples, h_world_test) # [B,S]

    # per-H metrics
    per_h_max_gap = np.array([max_angle_gap(base_heading_angles[i]) for i in range(base_heading_angles.shape[0])], dtype=np.float64)
    max_gap_mean = float(np.mean(per_h_max_gap))
    max_gap_p95 = float(np.quantile(per_h_max_gap, 0.95))

    # global KL to uniform over all angles
    th_all = base_heading_angles.reshape(-1)
    kl = kl_to_uniform(th_all, n_bins=cfg.n_bins_theta, eps=cfg.eps_hist)

    # "uses latent" proxy (works for any stochastic sampler): fix one H and check variance over samples -- TODO: think about this more!
    # H0 = H_test[0:1]
    # Q0 = model.sample(H0, n_samples=2000, rng=rng)[0]  # [S,3]
    # var = var_Q(Q0)

    # distributional comparison: compare generate joint configs vs. ground truth joint configs
    num_h = h_world_test.shape[0]
    mmd_scores = []
    for i in range(num_h):
        # samples for this specific H
        pred_cloud = q_samples[i] # shape [S, d_q]
        h_repeated = np.tile(h_world_test[i], (S, 1))
        # print("h repeated shape: ", h_repeated.shape)
        gt_cloud = env.sample_q(h_repeated, rng)
        # print("gt cloud shape = ", gt_cloud.shape)        
        score = compute_mmd(pred_cloud, gt_cloud)
        mmd_scores.append(score)
    avg_mmd = np.mean(mmd_scores)

    return {
        "hand_err/mean": err_mean,
        "hand_err/median": err_median,
        "hand_err/p95": err_p95,
        "coverage/max_gap_mean": max_gap_mean,
        "coverage/max_gap_p95": max_gap_p95,
        "coverage/kl_to_uniform": kl,
        "eval/theta_values": th_all.astype(np.float64),
        "eval/avg_mmd": avg_mmd
    }
