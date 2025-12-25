from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from reachability.types import Batch
from reachability.data.datasets import SimpleDataset

@dataclass
class SimpleDataLoader:
    dataset: SimpleDataset
    batch_size: int
    shuffle: bool
    rng: np.random.Generator

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            sl = idx[start : start + self.batch_size]
            yield self.dataset.batch(sl)

    def all(self) -> Batch:
        idx = np.arange(len(self.dataset))
        return self.dataset.batch(idx)
