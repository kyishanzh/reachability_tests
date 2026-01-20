# run from root reachability_tests/: python -m scripts.train_cinn --config configs/simple_cinn.yaml --wandb
# to save trained model, add tags --save --save_path "outputs/model_ckpts/cinn/cinn_rotary_1142026.pt" (etc.)
from __future__ import annotations

import argparse
import yaml
import wandb
from pathlib import Path

from reachability.utils.utils import set_seed, print_results
from reachability.envs.simple import SimpleEnv, Workspace2D
from reachability.envs.rotary_link import RotaryLinkEnv
from reachability.data.datasets import Dataset
from reachability.data.loaders import DataLoader
from reachability.models.cinn import CINNConditionalSampler
from reachability.eval.eval_model import EvalConfig, evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/simple_cinn.yaml")
    ap.add_argument("--wandb", default=False, action="store_true")
    ap.add_argument("--save", default=False, action="store_true")
    ap.add_argument("--save_path")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run = None
    if args.wandb:
        run = wandb.init(
            project=cfg.get("project_name", "simple_cinn"),
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

    train_full_ds = Dataset.generate(env=env, n=n_train, rng=rng)
    train_ds, val_ds = train_full_ds.split(split_ratio=0.9, rng=rng)
    test_ds = Dataset.generate(env=env, n=n_test, rng=rng)

    H_test = test_ds.H_raw

    # model config
    mcfg = cfg["model"]
    device = str(mcfg.get("device", "cpu"))
    
    # data preprocessing
    basexy_norm_type = mcfg.get("basexy_norm_type", "standardize")
    train_ds.preprocess(basexy_norm_type=basexy_norm_type)
    train_ds.to(device)
    val_ds.preprocess(basexy_norm_type=basexy_norm_type)
    val_ds.to(device)

    # create loaders
    batch_size = int(mcfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, rng=rng)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, rng=rng)

    # model
    model = CINNConditionalSampler(
        env=env,
        # model
        hidden_dim=int(mcfg["hidden_dim"]),
        dQ=train_full_ds.dQ,
        dQ_feat=train_loader.dataset.dQ_feat,
        dCond=train_loader.dataset.dH,
        num_blocks=int(mcfg["num_blocks"]),
        clamp=float(mcfg.get("clamp", 1.0)),
        basexy_norm_type=mcfg.get("basexy_norm_type", "standardize"),
        # training
        device=device,
        epochs=int(mcfg["epochs"]),
        seed=int(cfg["seed"]),
        lr=float(mcfg["lr"]),
        lr_milestones=mcfg.get("lr_milestones", [20, 40]),
        lr_gamma=mcfg.get("lr_gamma", 0.1),
        batch_size=batch_size,
        grad_clip=float(mcfg.get("grad_clip", 10.)),
        # optional constraint shaping
        lambda_fk=float(mcfg.get("lambda_fk", 0.0)),
        # wandb
        wandb_run=run
    )
    model.fit(train_loader=train_loader, val_loader=val_loader, val_frequency=10)

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
