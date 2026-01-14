from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from reachability.types import Batch

@dataclass(frozen=True)
class Dataset:
    env: Any
    H: np.ndarray  # [N, 2]
    Q: np.ndarray  # [N, 3]

    @classmethod
    def generate(cls, env, n: int, rng: np.random.Generator):
        """H ~ p(H) then Q ~ p*(Q | H)"""
        H = env.sample_H(n, rng)
        Q = env.sample_Q_given_H_uniform(H, rng)
        return cls(env=env, H=H, Q=Q)

    def __len__(self) -> int:
        return int(self.H.shape[0])

    def batch(self, idcs: np.ndarray) -> Batch:
        """Returns a chunk of H and Q specified by idx"""
        return Batch(H=self.H[idcs], Q=self.Q[idcs])
