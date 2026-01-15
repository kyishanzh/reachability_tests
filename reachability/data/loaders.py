from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from reachability.data.datasets import Dataset

@dataclass
class DataLoader:
    dataset: Dataset
    batch_size: int
    shuffle: bool
    rng: np.random.Generator

    def __iter__(self):
        n = len(self.dataset)
        idcs = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idcs)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = idcs[start:end]
            yield self.dataset[batch_idx] # dataset.__getitem__

    def __len__(self) -> int:
        """Returns number of batches per epoch"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
