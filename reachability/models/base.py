from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ConditionalGenerativeModel(ABC):
    @abstractmethod
    def fit(self, train_loader, val_loader) -> None:
        """H_train: [N, dH], Q_train: [N, dQ]"""
        # knn child class of this breaks substitutability (has a different function signature) -- but all the models share this!
        ...

    @abstractmethod
    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float) -> np.ndarray:
        """Return Q_samples with shape [batch size B, n_samples, dQ]"""
        ...
