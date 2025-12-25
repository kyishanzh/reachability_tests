from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Batch:
    H: np.ndarray # [B, dH]
    Q: np.ndarray # [B, dQ]