# running from root (reachability_tests/): python -m scripts.train_cvae --config configs/simple_cvae.yaml --wandb
# to save trained model, add tags --save --save_path "outputs/model_ckpts/cvae/cvae_rotary_1142026.pt" (etc.)
from __future__ import annotations

import argparse
import yaml
import wandb
from pathlib import Path

from reachability.utils.utils import set_seed, print_results
from reachability.envs.simple import SimpleEnv, Workspace2D
from reachability.envs.rotary_link import RotaryLinkEnv
from reachability.data.datasets import Dataset
from reachability.models.cvae import CVAEConditionalSampler
from reachability.eval.eval_model import EvalConfig, evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_cvae.yaml")
    ap.add_argument("--wandb", default=False, action="store_true")
    ap.add_argument("--save", default=False, action="store_true")
    ap.add_argument("--save_path", default="../outputs/model_ckpts")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run = None
    if args.wandb:
        run = wandb.init(
            project=cfg.get("project_name", "simple_cvae"),
            config=cfg
        )
    
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

    H_train, Q_train = train_ds.H, train_ds.Q
    H_test = test_ds.H

    # model config
    mcfg = cfg["model"]

    # model
    model = CVAEConditionalSampler(
        env=env,
        dQ=Q_train.shape[1],
        z_dim=int(mcfg["z_dim"]),
        enc_hidden=tuple(mcfg["enc_hidden"]),
        dec_hidden=tuple(mcfg["dec_hidden"]),
        lr=float(mcfg["lr"]),
        batch_size=int(mcfg["batch_size"]),
        epochs=int(mcfg["epochs"]),
        beta=float(mcfg["beta"]),
        device=str(mcfg.get("device", "cpu")),
        seed=int(cfg["seed"]),
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
