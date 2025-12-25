# running from root (cd reachability_tests): python -m scripts.train_knn --config configs/simple_knn.yaml
from __future__ import annotations
import argparse
import yaml
import numpy as np

from reachability.utils.utils import set_seed, print_results
from reachability.envs.simple import SimpleEnv, Workspace2D
from reachability.data.datasets import SimpleDataset
from reachability.models.knn import KNNConditionalSampler, NNDeterministicLookup
from reachability.eval.eval_model import EvalConfig, evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_knn.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    rng = set_seed(int(cfg["seed"]))

    # env
    env_cfg = cfg["env"]
    ws = Workspace2D(**env_cfg["workspace"])
    env = SimpleEnv(L=float(env_cfg["L"]), workspace=ws)

    # data
    n_train = int(cfg["data"]["n_train"])
    n_test = int(cfg["data"]["n_test"])

    train_ds = SimpleDataset.generate(env=env, n=n_train, rng=rng)
    test_ds = SimpleDataset.generate(env=env, n=n_test, rng=rng)

    H_train, Q_train = train_ds.H, train_ds.Q
    H_test = test_ds.H

    # eval config
    ecfg = cfg["eval"]
    eval_cfg = EvalConfig(
        n_samples_per_H=int(ecfg["n_samples_per_H"]),
        n_bins_theta = int(ecfg["n_bins_theta"]),
        eps_hist=float(ecfg["eps_hist"])
    )

    mcfg = cfg["model"]

    # deterministic nearest neighbors baseline
    nn_model = NNDeterministicLookup(
        metric=str(mcfg.get("metric", "euclidean")),
        algorithm=str(mcfg.get("algorithm", "auto"))
    )
    nn_model.fit(H_train=H_train, Q_train=Q_train)
    nn_results = evaluate_model(env=env, model=nn_model, H_test=H_test, cfg=eval_cfg, rng=rng)
    print_results("NNDeterministicLookup", nn_results)

    # stochastic kNN
    knn_model = KNNConditionalSampler(
        k=int(mcfg["k"]),
        sigma=float(mcfg["sigma"]),
        metric=str(mcfg.get("metric", "euclidean")),
        algorithm=str(mcfg.get("algorithm", "auto"))
    )
    knn_model.fit(H_train=H_train, Q_train=Q_train)
    knn_results = evaluate_model(env=env, model=knn_model, H_test=H_test, cfg=eval_cfg, rng=rng)
    print_results("KNNConditionalSampler", knn_results)

if __name__ == "__main__":
    main()
