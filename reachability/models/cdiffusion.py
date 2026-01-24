from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Any, List

import math
import wandb
import numpy as np
import torch
import torch.nn as nn

from diffusers import DDPMScheduler

from reachability.models.base import ConditionalGenerativeModel
from reachability.models.loss import fk_mse_from_qfeat_wrapper
from reachability.models.features import c_world_to_feat, q_feat_to_world, q_world_to_feat
from reachability.utils.utils import compute_bucketed_losses, compute_snr_stats

# TODO: ensure that data fed in (q_feat and c_feat) are normalized to [-1, 1]: diffusion models like this!

class SinusoidalPosEmb(nn.Module):
    """Positional encoding for tracking time step t"""
    # Read into how sinusoidal positional embeddings work
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device 
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module): # Consolidate with version of this class in cvae_helpers.py later
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(), # read literature explaining why SiLU works better than ReLU for diffusion
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class ResMLPDenoiser(nn.Module):
    def __init__(
        self,
        d_q_feat: int,
        d_c_feat: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        dropout: float = 0.0
    ):
        """Args:
        d_q_feat: dimension of the state (noisy input)
        d_c_feat: dimension of the conditioning vector
        hidden_dim: internal dimension of the MLP
        num_res_blocks: how deep the network is
        """
        super().__init__()
        self.d_q_feat = d_q_feat
        self.d_c_feat = d_c_feat

        # 1. time embedding: TODO figure out how to optimally do this part
        self.time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.Mish(),
            nn.Linear(self.time_dim * 2, self.time_dim)
        )

        # 2. input projection
        # concatenate input (q) + condition (c) + time embedding
        input_dim = d_q_feat + d_c_feat + self.time_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 3. residual net
        self.res_blocks = nn.ModuleList([ # can't use nn.Sequential because we want to manually control how the forward pass goes in diffusion model
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # 4. output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, d_q_feat) # prediction of noise (epsilon)

    def forward(self, x, t, c):
        # embed time
        t_emb = self.time_mlp(t)

        # concatenate everything
        x_input = torch.cat([x, c, t_emb], dim=-1)
        # print("x_input shape: ", x_input.shape)

        # project to hidden dim
        h = self.input_proj(x_input)

        # apply residual blocks
        for block in self.res_blocks:
            h = block(h)
        
        h = self.final_norm(h)
        return self.final_linear(h)

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

@dataclass
class DiffusionConditionalSampler(ConditionalGenerativeModel):
    """Wraps a ResMLPDenoiser into fit/sample API."""
    env: Any

    # model
    hidden_dim: int
    d_q: int
    d_q_feat: int
    d_c_feat: int
    num_blocks: int
    basexy_norm_type: str = "relative"

    # training
    device: str = "cpu"
    epochs: int = 100
    seed: int = 0
    lr: float = 1e-4
    batch_size: int = 256
    num_train_timesteps: int = 1000
    grad_clip: float = 1.0

    # inference
    num_inference_timesteps: int = 50

    # optional constraint shaping TODO: think about if we want to add this for diffusion architecture
    # lambda_fk: float = 0.0

    _model: ResMLPDenoiser | None = None
    wandb_run: object | None = None

    def __post_init__(self):
        self.noise_scheduler = DDPMScheduler( # TODO: look into custom implementations later
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False, # we already z-score normalize
            prediction_type="epsilon" # we train model to predict epsilon
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, val_frequency: int = 10) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"d_c_feat = {self.d_c_feat}, d_q_feat = {self.d_q_feat}")

        # initialize model
        self._model = ResMLPDenoiser(
            d_q_feat=self.d_q_feat,
            d_c_feat=self.d_c_feat,
            hidden_dim=self.hidden_dim,
            num_res_blocks=self.num_blocks
        ).to(self.device)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # normalize data that we noise (since noise added is N(0, 1), we need x + epsilon -> x should be within [-1, 1] ish range to have noise + actual input within the same scale)
        self.normalizer = Normalizer(self.d_q_feat, device=self.device)
        all_q_feats = []
        for batch in train_loader:
            all_q_feats.append(batch['q_feat'])
        full_data = torch.cat(all_q_feats).to(self.device)
        self.normalizer.fit(full_data)
        del full_data, all_q_feats
        print(f"normalizer mean: {self.normalizer.mean.cpu().numpy()} | normalizer std: {self.normalizer.std.cpu().numpy()}")
        # TODO find more elegant way to do this potentially ^
        
        self._model.train()
        nll_mean = []
        # training loop -- over all epochs
        for ep in range(self.epochs):
            total_loss = 0.0
            total_nll = 0.0
            total_fk = 0.0
            n_seen = 0
            
            # Train batch loop
            for batch in train_loader:
                c_feat_batch = batch['c_feat']
                q_feat_batch = batch['q_feat']
                # h_world_batch = torch.from_numpy(batch["h_world"]).to(self.device)

                # normalize input q_feat
                x_0 = self.normalizer.normalize(q_feat_batch)
                batch_size = x_0.shape[0]

                # 1. sample noise
                noise = torch.randn_like(x_0)
                # 2. sample timesteps to train on for each data point in batch
                time_steps = torch.randint(0, self.num_train_timesteps, (x_0.shape[0],)).long().to(self.device)
                # 3. noise the samples (forward diffusion)
                noisy_x = self.noise_scheduler.add_noise(x_0, noise, time_steps) # add_noise auto handles *noise^i given timestep i
                # 4. get the model prediction for epsilon
                noise_pred = self._model(noisy_x, time_steps, c_feat_batch)
                # 5. calculate the loss
                raw_losses = nn.functional.mse_loss(noise_pred, noise, reduction='none') # [batch_size, d_q_feat]
                per_sample_losses = raw_losses.mean(dim=1) # [batch_size]
                loss = per_sample_losses.mean() # scalar

                loss = nn.functional.mse_loss(noise_pred, noise)

                # 6. backprop
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=self.grad_clip)
                opt.step()

                # batch level metrics to log to wandb
                with torch.no_grad():
                    wandb_metrics = {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": opt.param_groups[0]['lr'],
                        "train/pred_noise_mean": noise_pred.mean().item(),
                        "train/pred_noise_std": noise_pred.std().item()
                    }

                    # timestep bucketing
                    wandb_metrics.update(compute_bucketed_losses(losses=per_sample_losses, time_steps=time_steps, total_steps=self.num_train_timesteps, num_buckets=5, wandb_title="train"))

                    # snr metrics
                    wandb_metrics.update(compute_snr_stats(time_steps=time_steps, alphas_cumprod=self.noise_scheduler.alphas_cumprod, wandb_title="train"))

            # Validation loop
            if val_loader is not None and (ep % val_frequency == 0):
                # Running the full sampling process for validation would slow down the training loop, so will implement validation the same way that we do training
                self._model.eval() # disable dropout, etc.
                val_losses = []
                with torch.no_grad(): # disable computation graph tracking (autograd)
                    for batch in val_loader:
                        c_feat_batch = batch['c_feat']
                        q_feat_batch = batch['q_feat']
                        x_0 = self.normalizer.normalize(q_feat_batch)
                        batch_size = x_0.shape[0]
                        # 1. sample noise
                        noise = torch.randn_like(x_0)
                        # 2. sample timesteps to train on for each data point in batch
                        time_steps = torch.randint(0, self.num_train_timesteps, (x_0.shape[0],)).long().to(self.device)
                        # 3. noise the samples (forward diffusion)
                        noisy_x = self.noise_scheduler.add_noise(x_0, noise, time_steps) # add_noise auto handles *noise^i given timestep i
                        # 4. get the model prediction for epsilon
                        noise_pred = self._model(noisy_x, time_steps, c_feat_batch)
                        # 5. calculate the loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        val_losses.append(loss.item())

                        # batch level metrics to log to wandb
                        # TODO^ > figure out how much i want to log here
                        # timestep bucketing
                        # wandb_metrics.update(compute_bucketed_losses(losses=loss, time_steps=time_steps, total_steps=self.num_diffusion_timesteps, num_buckets=5, wandb_title="val"))
                        # snr metrics
                        # wandb_metrics.update(compute_snr_stats(time_steps=time_steps, alphas_cumprod=self.noise_scheduler.alphas_cumprod, wandb_title="val"))
                wandb_metrics.update({
                    "val/loss": np.mean(val_losses)
                })
                self._model.train()

            # wandb tracking
            if self.wandb_run is not None:
                wandb.log(wandb_metrics, step=ep)
            
        self._model.eval()

    @torch.no_grad()
    def sample(self, h_world: np.ndarray, c_world: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float) -> np.ndarray: # probably something wrong with handling normalization here
        """
        h_world: [B, d_h] numpy
        returns: [B, n_samples, d_q] numpy Q=(x,y,theta)
        """
        if self._model is None:
            raise RuntimeError("Call fit() (or load()) before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # 1. move conditioning inputs into torch
        c_feat = c_world_to_feat(self.env, c_world, basexy_norm_type=self.basexy_norm_type)
        c_feat_batch = torch.from_numpy(c_feat.astype(np.float32)).to(self.device) # [B, d_c]
        B = c_feat_batch.shape[0] # batch size

        # 2. initialize from pure noise
        x = torch.randn((B * n_samples, self.d_q_feat), device=self.device) # x_T
        c_featb_repeated = c_feat_batch.repeat_interleave(n_samples, dim=0) # conditioning inputs

        # set up the scheduler for inference
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device) # creates inference-time timestep sequence

        # 3. denoising loop
        for t in self.noise_scheduler.timesteps:
            # model expects a batch of timesteps
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            # predict the noise residual
            residual = self._model(x, t_batch, c_featb_repeated) # predicted epsilon for x_t
            # remove the noise (x_t -> x_{t-1})
            x = self.noise_scheduler.step(residual, t, x).prev_sample # restored x_t-1

        # 4. unnormalize back to geometric feature space -- x is currently ~N(0,1)
        q_sample_feat = self.normalizer.unnormalize(x)

        print(f"q_sample_feat shape from model: {q_sample_feat.shape}")
        q_sample_feat_np = q_sample_feat.detach().cpu().numpy().astype(np.float32)
        q_sample_world = q_feat_to_world(self.env, q_sample_feat_np, h_world=h_world, basexy_norm_type=self.basexy_norm_type)
        print(f"q_sample_world shape: {q_sample_world.shape}")
        return q_sample_world.reshape(B, n_samples, self.d_q)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # class params
                "hidden_dim": int(self.hidden_dim),
                "d_q": int(self.d_q),
                "d_q_feat": self.d_q_feat,
                "d_c_feat": self.d_c_feat,
                "num_blocks": int(self.num_blocks),
                "basexy_norm_type": self.basexy_norm_type,
                "num_train_timesteps": self.num_train_timesteps,
                "num_inference_timesteps": self.num_inference_timesteps,
                # model param
                "state_dict": self._model.state_dict(),
                "normalizer_mean": self.normalizer.mean,
                "normalizer_std": self.normalizer.std
            },
            path
        )

    @classmethod
    def load(cls, env: Any, path: str | Path, device: str = "cpu") -> "DiffusionConditionalSampler":
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        
        sampler = cls(
            env=env,
            hidden_dim=int(ckpt["hidden_dim"]),
            d_q=int(ckpt["d_q"]),
            d_q_feat=int(ckpt["d_q_feat"]),
            d_c_feat=int(ckpt["d_c_feat"]),
            num_blocks=int(ckpt["num_blocks"]),
            basexy_norm_type=ckpt['basexy_norm_type'],
            num_train_timesteps=ckpt['num_train_timesteps'], # do not actually need to store this for sampling
            num_inference_timesteps=ckpt['num_inference_timesteps'],
            device=device
        )

        model = ResMLPDenoiser(
            d_q_feat=int(ckpt["d_q_feat"]),
            d_c_feat=int(ckpt["d_c_feat"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            num_res_blocks=int(ckpt["num_blocks"]),
        ).to(device)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sampler._model = model
        sampler.normalizer = Normalizer(int(ckpt['d_q_feat']), device=device)
        sampler.normalizer.mean = ckpt["normalizer_mean"].to(device)
        sampler.normalizer.std = ckpt["normalizer_std"].to(device)
        sampler.normalizer.fitted = True # TODO: not using this -- deprecate?

        return sampler
        
    def count_parameters(self):
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        