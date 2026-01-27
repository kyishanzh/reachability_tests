# start a new sweep: (running from root reachability_tests/) python -m scripts.sweep --module scripts.train_cinn --config configs/rotarylink_cinn_v2.yaml --sweep_config configs/sweeps/cinn_sweep.yaml --project rotarylink_cinn_sweep --count 20
# ^ after starting, will print: Create sweep with ID: otprzynk
# joining existing sweep: start a new terminal, python -m scripts.sweep --module scripts.train_cinn --config configs/rotarylink_cinn_v2.yaml --sweep_config configs/sweeps/cinn_sweep.yaml --project rotarylink_cinn_sweep --count 20 --sweep_id otprzynk
import argparse
import yaml
import wandb
import sys
import importlib
import tempfile
import os
from pathlib import Path

def update_config(config, updates):
    """Helper to update nested dictionary with dot notation (e.g. "model.lr" -> dict["model"]["lr"])"""
    for key, value in updates.items():
        parts = key.split(".")
        sub_config = config
        for part in parts[:-1]:
            if part not in sub_config:
                sub_config[part] = {}
            sub_config = sub_config[part]
        sub_config[parts[-1]] = value
    return config

def run_agent(module_name, base_config_path, project_name):
    """Generic worker function called by wandb.agent"""
    def train_wrapper():
        with wandb.init() as run:
            # 1. Load the base config (standard .yaml)
            with open(base_config_path, "r") as f:
                full_cfg = yaml.safe_load(f)
            
            # 2. Override cfg with sweep parameters
            sweep_params = dict(run.config)
            full_cfg = update_config(full_cfg, sweep_params)

            # 3. Create a temp file for this trial's config
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(full_cfg, tmp)
                temp_config_path = tmp.name
            
            try:
                # 4. Import the target module (e.g. scripts.train_cinn)
                module = importlib.import_module(module_name) # assuming training scripts have a main() function

                # 5. Monkey-patch sys.argv so argparse in main() reads the temp config
                original_argv = sys.argv
                sys.argv = [
                    module_name,
                    "--config", temp_config_path,
                    "--wandb" # ensure the training script knows to log
                ]

                # 6. Run the training
                print(f"****** Starting trial: {module_name} ******")
                module.main()

            except Exception as e:
                print(f"‼️ Trial failed: {e}")
                raise e
            finally:
                # Cleanup
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
    return train_wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True, help="Python module to run (e.g., scripts.train_cinn)")
    parser.add_argument("--config", type=str, required=True, help="Path to base config.yaml")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing sweep ID (if joining a sweep)")
    parser.add_argument("--sweep_config", type=str, default=None, help="Path to sweep definition yaml (if starting new)")
    parser.add_argument("--count", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--project", type=str, default="reachability_sweeps", help="WandB Project Name")
    
    args = parser.parse_args()

    if args.sweep_id is None:
        if args.sweep_config is None:
            raise ValueError("Must provide either --sweep_id (to join a running sweep) or --sweep_config (to start a new sweep)")
            
        with open(args.sweep_config, "r") as f:
            sweep_conf = yaml.safe_load(f)
        
        sweep_id = wandb.sweep(sweep_conf, project=args.project)
        print(f"Created new sweep: sweep_id = {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: sweep_id = {sweep_id}")
    
    # Launch the agent
    worker_fn = run_agent(args.module, args.config, args.project)
    wandb.agent(sweep_id=sweep_id, function=worker_fn, count=args.count, project=args.project)

if __name__ == "__main__":
    main()
