from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import NearestNeighbors
from reachability.models.base import ConditionalGenerativeModel

@dataclass
class NNDeterministicLookup(ConditionalGenerativeModel):
    """Simplest baseline (deterministic nearest neighbors):
    - For each query H, find the single nearest training Hi
    - Return its associated Qi"""
    metric: str = "euclidean"
    algorithm: str = "auto"

    _H_train: np.ndarray | None = None
    _Q_train: np.ndarray | None = None
    _nn: NearestNeighbors | None = None

    def fit(self, H_train: np.ndarray, Q_train: np.ndarray) -> None:
        self._H_train = H_train
        self._Q_train = Q_train

        self._nn = NearestNeighbors(
            n_neighbors=1,
            metric=self.metric,
            algorithm=self.algorithm #type:ignore
        )
        self._nn.fit(self._H_train)

    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self._nn is None or self._H_train is None or self._Q_train is None:
            raise RuntimeError("Call fit() before sample()")
        _, idcs = self._nn.kneighbors(H, return_distance=True) # idcs: [B,1]
        idcs = idcs[:, 0] # B
        Q_nn = self._Q_train[idcs] # [B, dQ]
        # repeat deterministically along the sample axis to match [B, n_samples, dQ]
        out = np.repeat(Q_nn[:, None, :], repeats=n_samples, axis=1).astype(np.float32)
        return out

@dataclass
class KNNConditionalSampler(ConditionalGenerativeModel):
    """Conditional empirical model:
    - Retrieve k nearest neighbors in H-space
    - Sample their associated Qs with weights exp(-||Hi - H||^2 / sigma^2)"""
    k: int = 50
    sigma: float = 0.35
    metric: str = "euclidean"
    algorithm: str = "auto"

    def fit(self, H_train: np.ndarray, Q_train: np.ndarray) -> None:
        self._H_train = H_train
        self._Q_train = Q_train
        self._nn = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.metric,
            algorithm=self.algorithm, #type:ignore
        )
        self._nn.fit(self._H_train)

    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self._nn is None or self._H_train is None or self._Q_train is None:
            raise RuntimeError("Call fit() before sample()")
        dists, idcs = self._nn.kneighbors(H, return_distance=True) #[B, k], [B, k]

        # weights: exp(-d^2/sigma^2)
        sigma2 = float(self.sigma) ** 2
        w = np.exp(-(dists ** 2) / max(sigma2, 1e-12)).astype(np.float32)
        w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12
        p = w / w_sum # [B,k]

        B, k = idcs.shape
        dQ = self._Q_train.shape[1]
        out = np.zeros((B, n_samples, dQ))

        # sample neighbor indices for each query independently
        for b in range(B):
            choices = rng.choice(k, size=(n_samples,), replace=True, p=p[b])
            neighbor_rows = idcs[b, choices] # [n_samples]
            out[b] = self._Q_train[neighbor_rows]
        return out
