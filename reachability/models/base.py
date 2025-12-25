from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ConditionalGenerativeModel(ABC):
    @abstractmethod
    def fit(self, H_train: np.ndarray, Q_train: np.ndarray) -> None:
        """H_train: [N, dH], Q_train: [N, dQ]"""
        ...

    @abstractmethod
    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Return Q_samples with shape [batch size B, n_samples, dQ]"""
        ...
