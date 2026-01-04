from __future__ import annotations

import argparse
import yaml
import wandb
from pathlib import Path

from reachability.utils.utils import set_seed, print_results
from reachability.envs.simple import SimpleEnv, Workspace2D
from reachability.data.datasets import SimpleDataset
from reachability.models.cinn import CINNConditionalSampler
from reachability.eval.eval_model import EvalConfig, evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_cinn.yaml")
    ap.add_argument("--wandb", default=False, action="store_true")
    ap.add_argument("--save", default=False, action="store_true")
    ap.add_argument("--save_path", default="../outputs/model_ckpts")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run = None
    if args.wandb:
        run = wandb.init(
            project="kyz-mit",
            name=cfg.get("run_name", "simple_cinn"),
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
    model = CINNConditionalSampler(
        n_blocks=int(mcfg["n_blocks"]),
        hidden=int(mcfg["hidden"]),
        clamp=float(mcfg.get("clamp", 2.0)),
        lr=float(mcfg["lr"]),
        batch_size=int(mcfg["batch_size"]),
        epochs=int(mcfg["epochs"]),
        device=str(mcfg.get("device", "cpu")),
        seed=int(cfg["seed"]),
        L=float(env.L),
        lambda_fk=float(mcfg.get("lambda_fk", 0.0)),
        wandb_run=run
    )
    model.fit(H_train=H_train, Q_train=Q_train)

    # save model
    if args.save:
        model.save(args.save_path)
        save_path = Path(args.save_path)
        config_path = save_path.with_suffix(".yaml")
        with open(config_path, 'w') as file:
            yaml.dump(cfg, file)
        print(f"Saved model to {args.save_path}, corresponding config to {config_path}")

    # eval config
    ecfg = cfg["eval"]
    eval_cfg = EvalConfig(
        n_samples_per_H=int(ecfg["n_samples_per_H"]),
        n_bins_theta = int(ecfg["n_bins_theta"]),
        eps_hist=float(ecfg["eps_hist"])
    )

    # eval
    results = evaluate_model(env=env, model=model, H_test=H_test, cfg=eval_cfg, rng=rng)
    print_results("CINN", results)

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
