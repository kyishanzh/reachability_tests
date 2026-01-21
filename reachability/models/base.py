from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ConditionalGenerativeModel(ABC):
    @abstractmethod
    def fit(self, train_loader, val_loader) -> None:
        """h_train: [N, d_h], q_train: [N, d_q_feat]"""
        # knn child class of this breaks substitutability (has a different function signature) -- but all the models share this!
        ...

    @abstractmethod
    def sample(self, h_world: np.ndarray, c_world: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float) -> np.ndarray:
        """Return q_samples with shape [batch size B, n_samples, d_q]"""
        ...
