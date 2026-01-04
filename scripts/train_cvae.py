# running from root (reachability_tests/): python -m scripts.train_cvae --config configs/simple_cvae.yaml --wandb
from __future__ import annotations

import argparse
import yaml
import wandb

from reachability.utils.utils import set_seed, print_results
from reachability.envs.simple import SimpleEnv, Workspace2D
from reachability.data.datasets import SimpleDataset
from reachability.models.cvae import CVAEConditionalSampler
from reachability.eval.eval_model import EvalConfig, evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_cvae.yaml")
    ap.add_argument("--wandb", default=False, action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run = None
    if args.wandb:
        run = wandb.init(
            project="kyz-mit",
            name=cfg.get("run_name", "simple_cvae"),
            config=cfg
        )
    
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

    # model config
    mcfg = cfg["model"]

    # model
    model = CVAEConditionalSampler(
        z_dim=int(mcfg["z_dim"]),
        enc_hidden=tuple(mcfg["enc_hidden"]),
        dec_hidden=tuple(mcfg["dec_hidden"]),
        lr=float(mcfg["lr"]),
        batch_size=int(mcfg["batch_size"]),
        epochs=int(mcfg["epochs"]),
        beta=float(mcfg["beta"]),
        device=str(mcfg.get("device", "cpu")),
        seed=int(cfg["seed"]),
        wandb_run=run
    )
    model.fit(H_train=H_train, Q_train=Q_train)

    # eval config
    ecfg = cfg["eval"]
    eval_cfg = EvalConfig(
        n_samples_per_H=int(ecfg["n_samples_per_H"]),
        n_bins_theta = int(ecfg["n_bins_theta"]),
        eps_hist=float(ecfg["eps_hist"])
    )

    # eval
    results = evaluate_model(env=env, model=model, H_test=H_test, cfg=eval_cfg, rng=rng)
    print_results("CVAE", results)

    theta_values = results.pop("eval/theta_values", None)
    if args.wandb:
        wandb.log(
            {f"eval/{k.replace('/', '_')}": v for k, v in results.items()}
        )
        if theta_values is not None:
            wandb.log({"eval/theta_hist": wandb.Histogram(theta_values)})
        run.finish()

if __name__ == "__main__":
    main()