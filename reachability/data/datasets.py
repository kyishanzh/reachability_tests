from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

from reachability.models.features import q_world_to_feat, c_world_to_feat

@dataclass
class Dataset(TorchDataset):
    env: Any
    q_world: np.ndarray  # [N, d_q] Original angles
    h_world: np.ndarray  # [N, d_h] Original world coords
    c_world: np.ndarray # conditioning information

    # These will be populated after preprocessing
    c_feat: torch.Tensor | None = None # [N, d_c_feat] normalized coordinates
    q_feat: torch.Tensor | None = None # [N, d_q_feat] normalized coordinates + trig features

    device: str = "cpu"

    @property
    def d_q(self) -> int:
        return int(self.q_world.shape[1])
    
    @property
    def d_h(self) -> int:
        return int(self.h_world.shape[1])

    @property
    def d_c(self) -> int:
        return int(self.c_world.shape[1])
    
    @property
    def d_q_feat(self):
        if self.q_feat is None:
            raise RuntimeError("Dataset.d_q_feat referenced before self.q_feat was populated (likely because downstream processes were run before Dataset.preprocess).")
        return int(self.q_feat.shape[1])

    @property
    def d_c_feat(self):
        if self.c_feat is None:
            raise RuntimeError("Dataset.d_c_feat referenced before self.c_feat was populated (likely because downstream processes were run before Dataset.preprocess).")
        return int(self.c_feat.shape[1])

    @classmethod
    def generate(cls, env, n: int, rng: np.random.Generator):
        """h ~ p(h) then q ~ p*(q | h). Later, when adding conditioning variables (e.g. obstacles): q ~ p(q | c = [h, ...])"""
        h = env.sample_h(n, rng)
        q = env.sample_q_given_h_uniform(h, rng)
        return cls(env=env, q_world=q, h_world=h, c_world=h) # c_world = h for now, later add actual conditioning information

    def preprocess(
        self,
        basexy_norm_type: str = "relative",
        add_fourier_feat: bool = False,
        fourier_B: Any = None
    ) -> None:
        """Converts raw numpy arrays into normalized Torch features"""
        # 1. Normalize c
        c_feat = c_world_to_feat(self.env, self.c_world, basexy_norm_type=basexy_norm_type, add_fourier_feat=add_fourier_feat, fourier_B=fourier_B)
        print("c_feat shape: ", c_feat.shape)

        # 2. Featurize Q
        q_feat = q_world_to_feat(self.env, self.q_world, h_world=self.h_world, basexy_norm_type=basexy_norm_type)

        # 3. Store as float tensors
        self.c_feat = torch.from_numpy(c_feat).to(torch.float32)
        self.q_feat = torch.from_numpy(q_feat).to(torch.float32)

    def to(self, device: str) -> Dataset:
        """Moves data to GPU/CPU once to speed up training."""
        self.device = device
        if self.c_feat is not None:
            self.c_feat = self.c_feat.to(device)
            self.q_feat = self.q_feat.to(device)
        return self

    def __len__(self) -> int:
        return int(self.h_world.shape[0])

    def __getitem__(self, idx) -> dict[str, torch.Tensor | np.ndarray]:
        """Returns a single sample or batch slice."""
        if self.c_feat is None:
            raise RuntimeError("Call dataset.preprocess() before accessing data!")
        return {
            "c_feat": self.c_feat[idx],
            "q_feat": self.q_feat[idx],
            "h_world": self.h_world[idx]
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

        train_ds = Dataset(
            env=self.env,
            h_world=self.h_world[train_idx],
            q_world=self.q_world[train_idx],
            c_world=self.c_world[train_idx]
        )
        val_ds = Dataset(
            env=self.env,
            h_world=self.h_world[val_idx],
            q_world=self.q_world[val_idx],
            c_world=self.c_world[val_idx]
        )

        return train_ds, val_ds

class Normalizer(nn.Module): # TODO: read through and understand this code + decide if we want to normalize conditioning inputs too 
    """
    Simple standard scaler (z-score) module.
    x_norm = (x - mean) / std
    """
    def __init__(self, size: int, epsilon: float = 1e-5, device="cpu"):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size, device=device))
        self.register_buffer('std', torch.ones(size, device=device))
        self.epsilon = epsilon
        self.fitted = False # TODO: potentially get rid of this ? not using it in code

    def fit(self, data_tensor: torch.Tensor):
        """Compute stats from a large batch of data."""
        self.mean = torch.mean(data_tensor, dim=0)
        self.std = torch.clamp(torch.std(data_tensor, dim=0), min=self.epsilon) # TODO: read into why we do this
        self.fitted = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
        