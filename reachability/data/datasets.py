from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from reachability.utils.utils import q_to_qfeat, h_to_hnorm

@dataclass
class Dataset(TorchDataset):
    env: Any
    H_raw: np.ndarray  # [N, 2] Original world coords
    Q_raw: np.ndarray  # [N, Q_dim] Original angles

    # These will be populated after preprocessing
    H_feat: torch.Tensor | None = None # [N, 2] normalized coordinates
    Q_feat: torch.Tensor | None = None # [N, Q_feat] normalized coordinates + trig features
    device: str = "cpu"

    @property
    def dQ(self) -> int:
        return int(self.Q_raw.shape[1])
    
    @property
    def dH(self) -> int:
        return int(self.H_raw.shape[1])
    
    @property
    def dQ_feat(self):
        if self.Q_feat is None:
            raise RuntimeError("Dataset.dQ_feat referenced before self.Q_feat was populated (likely because downstream processes were run before Dataset.preprocess).")
        return int(self.Q_feat.shape[1])

    @property
    def dH_feat(self):
        if self.H_feat is None:
            raise RuntimeError("Dataset.dH_feat referenced before self.H_feat was populated (likely because downstream processes were run before Dataset.preprocess).")
        return int(self.H_feat.shape[1])

    @classmethod
    def generate(cls, env, n: int, rng: np.random.Generator):
        """H ~ p(H) then Q ~ p*(Q | H)"""
        H = env.sample_H(n, rng)
        Q = env.sample_Q_given_H_uniform(H, rng)
        return cls(env=env, H_raw=H, Q_raw=Q)

    def preprocess(
        self,
        basexy_norm_type: str = "bound",
        add_fourier_feat: bool = False,
        fourier_B: Any = None
    ) -> None:
        """Converts raw numpy arrays into normalized Torch features"""
        # 1. normalize H
        H_feat = h_to_hnorm(self.env, self.H_raw, basexy_norm_type=basexy_norm_type, add_fourier_feat=add_fourier_feat, fourier_B=fourier_B)
        print("H_feat shape: ", H_feat.shape)

        # 2. featurize Q
        Q_feat = q_to_qfeat(self.env, self.Q_raw, basexy_norm_type=basexy_norm_type)

        # 3. store as float tensors
        self.H_feat = torch.from_numpy(H_feat).to(torch.float32)
        self.Q_feat = torch.from_numpy(Q_feat).to(torch.float32)

    def to(self, device: str) -> Dataset:
        """Moves data to GPU/CPU once to speed up training."""
        self.device = device
        if self.H_feat is not None:
            self.H_feat = self.H_feat.to(device)
            self.Q_feat = self.Q_feat.to(device)
        return self

    def __len__(self) -> int:
        return int(self.H_raw.shape[0])

    def __getitem__(self, idx) -> dict[str, torch.Tensor | np.ndarray]:
        """Returns a single sample or batch slice."""
        if self.H_feat is None:
            raise RuntimeError("Call dataset.preprocess() before accessing data!")
        return {
            "H_feat": self.H_feat[idx],
            "Q_feat": self.Q_feat[idx],
            "H_raw": self.H_raw[idx]
        }

    def split(self, split_ratio: float = 0.8, rng: np.random.Generator = None) -> tuple[Dataset, Dataset]:
        """Splits this dataset into Train and Val sets."""
        n = len(self)
        n_train = int(n * split_ratio)
        indices = np.arange(n)
        if rng:
            rng.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_ds = Dataset(self.env, self.H_raw[train_idx], self.Q_raw[train_idx])
        val_ds = Dataset(self.env, self.H_raw[val_idx], self.Q_raw[val_idx])

        return train_ds, val_ds
