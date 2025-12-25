import numpy as np

def set_seed(seed: int) -> np.random.Generator:
    """Returns a numpy Generator seeded deterministically."""
    return np.random.default_rng(seed)

def print_results(name: str, results: dict) -> None:
    print(f"\n=== Results: {name} ===")
    for k, v in results.items():
        print(f"{k:30s} {v:.6f}" if isinstance(v, float) else f"{k:30s} {v}")
    