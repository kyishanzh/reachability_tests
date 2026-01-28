# running from root (cd reachability_tests): python -m scripts.train_knn --config configs/simple_knn.yaml
# to save trained model, add tags --save --save_path "outputs/model_ckpts/nn_baseline/nnbaseline_1132026.pt" (etc.)
from __future__ import annotations
import argparse
import yaml
import numpy as np
from pathlib import Path

from reachability.utils.utils import set_seed, print_results
from reachability.envs.workspace import Workspace2D
from reachability.envs.simple import SimpleEnv
from reachability.envs.rotary_link import RotaryLinkEnv
from reachability.data.datasets import Dataset
from reachability.models.knn import KNNConditionalSampler, NNDeterministicLookup
from reachability.eval.eval_model import EvalConfig, evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_knn.yaml")
    ap.add_argument("--save", default=False, action="store_true")
    ap.add_argument("--save_path", default="../outputs/model_ckpts")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    rng = set_seed(int(cfg["seed"]))

    # env
    env_cfg = cfg["env"]
    ws = Workspace2D(**env_cfg["workspace"])
    if env_cfg['type'] == 'Simple':
        env = SimpleEnv(L=float(env_cfg["L"]), workspace=ws)
    elif env_cfg['type'] == 'RotaryLink':
        joint_limits=env_cfg['joint_limits']
        if joint_limits == "None":
            joint_limits = None
        env = RotaryLinkEnv(
            ws,
            link_lengths=env_cfg['link_lengths'],
            joint_limits=joint_limits,
            n_links=int(env_cfg['n_links']),
            base_pos_eps=float(env_cfg['base_pos_eps']),
            base_heading_stddev=float(env_cfg['base_heading_stddev'])
        )
    else:
        raise ValueError(f"Invalid environment type: {env_cfg['type']}")
    
    # data
    n_train = int(cfg["data"]["n_train"])
    n_test = int(cfg["data"]["n_test"])

    train_ds = Dataset.generate(env=env, n=n_train, rng=rng)
    test_ds = Dataset.generate(env=env, n=n_test, rng=rng)

    h_train, q_train = train_ds.h_world, train_ds.q_world
    h_test = test_ds.h_world

    # eval config
    ecfg = cfg["eval"]
    eval_cfg = EvalConfig(
        n_samples_per_h=int(ecfg["n_samples_per_h"]),
        n_bins_theta = int(ecfg["n_bins_theta"]),
        eps_hist=float(ecfg["eps_hist"])
    )

    mcfg = cfg["model"]

    # deterministic nearest neighbors baseline
    nn_model = NNDeterministicLookup(
        metric=str(mcfg.get("metric", "euclidean")),
        algorithm=str(mcfg.get("algorithm", "auto"))
    )
    nn_model.fit(h_train=h_train, q_train=q_train)
    nn_results = evaluate_model(env=env, model=nn_model, h_world_test=h_test, c_world_test=h_test, cfg=eval_cfg, rng=rng)
    print_results("NNDeterministicLookup", nn_results)

    if args.save:
        nn_model.save(args.save_path)
        save_path = Path(args.save_path)
        config_path = save_path.with_suffix(".yaml")
        with open(config_path, 'w') as file:
            yaml.dump(cfg, file)
        print(f"Saved nearest neighbors model to {args.save_path}, corresponding config to {config_path}")

    # stochastic kNN
    knn_model = KNNConditionalSampler(
        k=int(mcfg["k"]),
        sigma=float(mcfg["sigma"]),
        metric=str(mcfg.get("metric", "euclidean")),
        algorithm=str(mcfg.get("algorithm", "auto"))
    )
    knn_model.fit(h_train=h_train, q_train=q_train)
    knn_results = evaluate_model(env=env, model=knn_model, h_world_test=h_test, c_world_test=h_test, cfg=eval_cfg, rng=rng)
    print_results("KNNConditionalSampler", knn_results)

if __name__ == "__main__":
    main()
