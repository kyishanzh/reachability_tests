from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from reachability.eval.metrics import *
from reachability.utils.plotting import *

@dataclass
class EvalConfig:
    n_samples_per_h: int
    n_bins_theta: int # how finely we discretize angle space when estimatig distributions
    eps_hist: float # for KL div - prevents log(0)
    sampling_temperature: float = 0.0

def evaluate_model(env, model, h_world_test: np.ndarray, c_world_test: np.ndarray, cfg: EvalConfig, rng: np.random.Generator, gt_samples_override = None) -> dict:
    if env.name == "Simple" or env.name == "RotaryLink":
        return evaluate_model_legacy(env, model, h_world_test, c_world_test, cfg, rng)
    elif env.name == "RotaryNLink":
        return evaluate_model_rotaryn(env, model, h_world_test, c_world_test, cfg, rng, gt_samples_override=gt_samples_override)
    else:
        raise RuntimeError(f"Unknown environment name encountered during evaluation (evaluate_model function): {env.name}")

def evaluate_model_legacy(env, model, h_world_test: np.ndarray, c_world_test: np.ndarray, cfg: EvalConfig, rng: np.random.Generator) -> dict:
    """Evaluate on test H points:
    - Draw S samples per h in h_world_test
    - Compute error/coverage stats"""
    S = cfg.n_samples_per_h
    q_samples = model.sample(h_world=h_world_test, c_world=c_world_test, n_samples=S, rng=rng, sampling_temperature=cfg.sampling_temperature)  # [B,S,d_q]
    
    errs = hand_error(env, q_samples, h_world_test) # [B,S]
    pos_err, ori_err = errs['pos_err'], errs['ori_err']
    pos_err_mean = float(np.mean(pos_err))
    pos_err_median = float(np.median(pos_err))
    pos_err_p95 = float(np.quantile(pos_err, 0.95))

    base_heading_angles = implied_angles(env, q_samples, h_world_test) # [B,S]

    # per-H metrics
    per_h_max_gap = np.array([max_angle_gap(base_heading_angles[i]) for i in range(base_heading_angles.shape[0])], dtype=np.float64)
    max_gap_mean = float(np.mean(per_h_max_gap))
    max_gap_p95 = float(np.quantile(per_h_max_gap, 0.95))

    # global KL to uniform over all angles
    th_all = base_heading_angles.reshape(-1)
    kl = kl_to_uniform(th_all, n_bins=cfg.n_bins_theta, eps=cfg.eps_hist)

    # distributional comparison: compare generate joint configs vs. ground truth joint configs
    num_h = h_world_test.shape[0]
    mmd_scores = []
    for i in range(num_h):
        # samples for this specific H
        pred_cloud = q_samples[i] # shape [S, d_q]
        h_repeated = np.tile(h_world_test[i], (S, 1))
        # print("h repeated shape: ", h_repeated.shape)
        gt_cloud = env.sample_q(h_repeated, np.random.default_rng(666)) # choose different seed
        # print("gt cloud shape = ", gt_cloud.shape)        
        score = compute_mmd(env, pred_cloud, gt_cloud)
        mmd_scores.append(score)
    avg_mmd = np.mean(mmd_scores)

    return {
        "hand_pos_err/mean": pos_err_mean,
        "hand_pos_err/median": pos_err_median,
        "hand_pos_err/p95": pos_err_p95,
        "coverage/max_gap_mean": max_gap_mean,
        "coverage/max_gap_p95": max_gap_p95,
        "coverage/kl_to_uniform": kl,
        "eval/theta_values": th_all.astype(np.float64),
        "eval/avg_mmd": avg_mmd
    }

def evaluate_model_rotaryn(env, model, h_world_test: np.ndarray, c_world_test: np.ndarray, cfg: EvalConfig, rng: np.random.Generator, gt_samples_override = None) -> dict:
    """Evaluate on test H points:
    - Draw S samples per h in h_world_test
    - Compute error/coverage stats"""
    S = cfg.n_samples_per_h
    B = h_world_test.shape[0]

    if gt_samples_override is None:
        # 1. Model sampling
        q_samples = model.sample(h_world=h_world_test, c_world=c_world_test, n_samples=S, rng=rng, sampling_temperature=cfg.sampling_temperature)  # [B,S,d_q]
    else:
        q_samples = gt_samples_override # Allow running eval function on ground truth samples
    
    # 2. Fidelity metrics (FK error)
    errs = hand_error(env, q_samples, h_world_test) # [B,S]
    pos_err, ori_err = errs['pos_err'], errs['ori_err']

    # 3. Feasibility metrics (Is the robot config valid?)
    validity_stats = compute_validity_rates(env, q_samples)

    # 4. Distributional & manifold metrics
    mmd_scores, emd_bearing_scores, cov_scores, fid_scores = [], [], [], []

    # Calculate implied bearings for model samples
    bearings_model = implied_angles(env, q_samples, h_world_test) # [B,S]

    for i in range(B):
        # Get model cloud (samples ~ model) for H[i]
        pred_cloud = q_samples[i] # shape [S, d_q]
        pred_bearing = bearings_model[i] # [S]
        
        # Get ground truth cloud for H[i]
        h_repeated = np.tile(h_world_test[i], (S, 1))
        gt_cloud = env.sample_q(h_repeated, np.random.default_rng(666)) # [S, d_q]

        # Joint space MMD -> distributional comparison: compare generate joint configs vs. ground truth joint configs
        score = compute_mmd(env, pred_cloud, gt_cloud)
        mmd_scores.append(score)

        # Compute bearing EMD (circular)
        gt_bearing = implied_angles(env, gt_cloud[None, :, :], h_world_test[i:i+1]).reshape(-1) # gt_cloud[None, :, :].shape = (1, S, d_q) -> need to have a batch in front because implied angles processes: B, S, _ = q_samples.shape
        emd_bearing_scores.append(compute_emd_circular(pred_bearing, gt_bearing))

        # Compute coverage & fidelity
        cov, fid = compute_coverage_fidelity(env, pred_cloud, gt_cloud)
        cov_scores.append(cov)
        fid_scores.append(fid)

    metrics = {
        # Fidelity
        "fidelity/pos_err_mean": float(np.mean(pos_err)),
        "fidelity/pos_err_median": float(np.median(pos_err)),
        "fidelity/pos_err_p95": float(np.quantile(pos_err, 0.95)),
        "hand_ori_err/mean": float(np.mean(ori_err)),
        "hand_ori_err/median": float(np.median(ori_err)),
        "hand_ori_err/p95": float(np.quantile(ori_err, 0.95)),
        # Feasibility
        "feasibility/collision_rate": validity_stats["collision_rate"],
        "feasibility/joint_limit_rate": validity_stats["limit_rate"],
        # Distribution
        "distribution/mean_mmd": float(np.mean(mmd_scores)),
        "distribution/mean_emd_bearing": float(np.mean(emd_bearing_scores)),
        # Manifold mode coverage
        "manifold/coverage_recall": float(np.mean(cov_scores)),
        "manifold/fidelity_precision": float(np.mean(fid_scores))
    }

    return metrics

def visualize_sampled_distribution(env, q_samples, h_world, q_gt):
    """
    Visualize the distribution of samples generated for one target pose h_world.
    q_samples: [num_samples, d_q]
    h_world: [d_h] s.t. q_samples ~ distribution(h_world)
    q_gt: [num_samples, d_q] (q_gt ~ gt_distribution(h_world))
    """
    bx = q_samples[:, 0]
    by = q_samples[:, 1]
    psi = q_samples[:, 2]
    thetas = q_samples[:, 3:]

    plot_data_distribution(h_world, bx, by, psi, thetas)
    plot_marginal_comparisons(q_gt, q_samples)
    plot_binding_correlation(env, q_gt, q_samples, h_world)
    plot_density_difference(q_gt, q_samples)
