from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
from typing import Any

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reachability.data.loaders import DataLoader
from reachability.models.base import ConditionalGenerativeModel
from reachability.models.loss import fk_mse_from_qfeat_wrapper
from reachability.models.features import q_world_to_feat, c_world_to_feat, q_feat_to_world
from reachability.utils.utils import grad_global_norm
from reachability.models.submodels import MLP, ResidualMLP, gaussian_nll, kl_standard_normal, get_beta

class ConditionalVAE(nn.Module):
    """
    Encoder: (c_feat, q_feat) -> (mu_z, logvar_z) for Pr(z)
    Decoder: (c_feat, z) -> (mu_q, logvar_q) for Pr(q_feat)
    """
    def __init__(
            self,
            d_c_feat: int,
            d_q_feat: int,
            z_dim: int,
            # resnet setup
            hidden_dim: int = 512,
            num_enc_blocks: int = 3,
            num_dec_blocks: int = 3,
            # vanilla MLP setup
            # enc_hidden: Sequence[int],
            # dec_hidden: Sequence[int],
            logvar_clip: float = 6.0
    ):
        super().__init__()
        self.d_c_feat = d_c_feat
        self.d_q_feat = d_q_feat
        self.z_dim = z_dim
        self.logvar_clip = logvar_clip

        # self.encoder = MLP(dH + dQ_feat, enc_hidden, 2 * z_dim) # [mu, logvar]
        self.encoder = ResidualMLP(
            in_dim=d_c_feat + d_q_feat,
            hidden_dim=hidden_dim,
            out_dim=2 * z_dim,
            num_blocks=num_enc_blocks
        )
        print("num params in encoder = ", sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        # self.decoder = MLP(dH + z_dim, dec_hidden, 2 * dQ_feat) # [mu, logvar]
        self.decoder = ResidualMLP(
            in_dim=d_c_feat + z_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * d_q_feat,
            num_blocks=num_dec_blocks
        )
        print("num params in decoder = ", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))

    def encode(self, c_feat: torch.Tensor, q_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | c_feat, q_feat)"""
        x = torch.cat([c_feat, q_feat], dim=-1)
        out = self.encoder(x)
        mu, logvar = out[:, :self.z_dim], out[:, self.z_dim:]
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps - reparameterization trick!"""
        std = torch.exp(0.5 * logvar) # recover std dev from logvar
        eps = torch.randn_like(std) # eps ~ N(0, I) (same shape as std)
        return mu + std * eps

    def decode(self, c_feat: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(q_feat | c_feat, z)"""
        x = torch.cat([c_feat, z], dim=-1)
        out = self.decoder(x)
        mu, logvar = out[:, :self.d_q_feat], out[:, self.d_q_feat:]
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)
        return mu, logvar
    
    def forward(self, c_feat: torch.Tensor, q_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(c_feat, q_feat)
        z = self.reparameterize(mu_z, logvar_z)
        mu_q, logvar_q = self.decode(c_feat, z)
        return {
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "mu_q": mu_q,
            "logvar_q": logvar_q
        }

@dataclass
class CVAEConditionalSampler(ConditionalGenerativeModel):
    """Wraps a torch ConditionalVAE into fit/sample API."""
    env: Any
    d_q: int
    d_q_feat: int
    d_c_feat: int
    z_dim: int # test more vals -- theoretically need at least DoF of environment to have a chance at doing well
    # enc_hidden: tuple[int, ...] = (128, 128)
    # dec_hidden: tuple[int, ...] = (128, 128)
    hidden_dim: int = 512
    num_enc_blocks: int = 3,
    num_dec_blocks: int = 3,
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    beta: float = 1.0
    device: str = "cpu"
    seed: int = 0
    lambda_fk: float = 0.0 # FK penalty
    fk_ori_weight: float = 1.0 # FK orientation MSE weight
    wandb_run: object | None = None
    basexy_norm_type: str = "relative"
    add_fourier_feat: bool = False
    fourier_B: Any = None

    _model: ConditionalVAE | None = None

    def fit(self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        val_frequency: int = 10,
        save_best_val_ckpt: bool = True,
        save_path: str = ""
    ) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"d_c_feat = {self.d_c_feat}, d_q_feat = {self.d_q_feat}")

        self._model = ConditionalVAE(
            d_c_feat=self.d_c_feat,
            d_q_feat=self.d_q_feat,
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            num_enc_blocks=self.num_enc_blocks,
            num_dec_blocks=self.num_dec_blocks
            # enc_hidden=self.enc_hidden,
            # dec_hidden=self.dec_hidden
        ).to(self.device)

        print("num trainable params in model: ", self.count_parameters())

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        lowest_loss_so_far = float("inf")
        for ep in range(self.epochs):
            total_loss, total_rec, total_kl, total_fk = 0.0, 0.0, 0.0, 0.0
            n_seen = 0

            # Train loop
            for batch in train_loader:
                c_feat_batch = batch['c_feat']
                q_feat_batch = batch['q_feat']
                h_world_batch = torch.from_numpy(batch["h_world"]).to(self.device)

                out = self._model(c_feat_batch, q_feat_batch)
                rec = gaussian_nll(q_feat_batch, out["mu_q"], out["logvar_q"]) # [B]
                kl = kl_standard_normal(out["mu_z"], out["logvar_z"]) # [B]
                fk_err2 = fk_mse_from_qfeat_wrapper(self.env, out["mu_q"], h_world_batch, basexy_norm_type=self.basexy_norm_type, ori_weight=self.fk_ori_weight)
                current_beta = get_beta(ep, self.epochs, self.beta) # dynamic beta
                loss = torch.mean(rec + current_beta * kl + self.lambda_fk * fk_err2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                val_batch_size = c_feat_batch.shape[0] # last batch size might not equal self.batch_size
                total_loss += float(loss.item()) * val_batch_size
                total_rec += float(torch.mean(rec).item()) * val_batch_size
                total_kl += float(torch.mean(kl).item()) * val_batch_size
                total_fk += float(torch.mean(fk_err2).item()) * val_batch_size
                n_seen += val_batch_size

            # Training loss averages
            avg_loss = total_loss/n_seen
            avg_rec = total_rec/n_seen
            avg_kl = total_kl/n_seen
            avg_fk = total_fk/n_seen

            # More training metrics! (using metrics from last batch of this epoch)
            with torch.no_grad():
                # KL per dimension to show if one latent dim is "dead" (near-zero KL contribution)
                latent_mu, latent_logvar = out["mu_z"], out["logvar_z"] # [Batchsize, dQfeat]
                kl_per_dim = 0.5 * (latent_mu**2 + torch.exp(latent_logvar) - 1.0 - latent_logvar).mean(dim=0) # [dQfeat] sum (mean) across batches -> preserves dimension
                # Latent health metrics:
                latent_mu_abs_mean = latent_mu.abs().mean().item() # if near 0 always, encoder might be collapsing
                latent_logvar_mean = latent_logvar.mean().item() # very negative -> tiny std -> near-deterministic encoder
                # Gradient norm: helps detect instability or dead training
                gnorm = grad_global_norm(self._model.parameters())
            
            wandb_metrics = {
                "train/loss": avg_loss,
                "train/recon": avg_rec,
                "train/kl": avg_kl,
                "train/fk": avg_fk,
                "train/beta": current_beta,
                "train/latent_mu_mean": latent_mu_abs_mean,
                "train/latent_logvar_mean": latent_logvar_mean,
                "train/gradient_norm": gnorm,
                "epoch": ep,
            }
            z_dims_to_wandbtrack = min(self.z_dim, 10) # if gets above 10, only send first 10 to wandb
            for i in range(z_dims_to_wandbtrack):
                wandb_metrics[f"train/kl_zdim{i}"] = kl_per_dim[i].item()

            # Validation loop
            if val_loader is not None and (ep % val_frequency == 0):
                total_val_loss, total_val_rec, total_val_kl, total_val_fk = 0.0, 0.0, 0.0, 0.0
                val_n_seen = 0
                self._model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        h_world_batch = torch.from_numpy(batch["h_world"]).to(self.device)
                        out = self._model(batch['c_feat'], batch['q_feat'])
                        rec = gaussian_nll(batch['q_feat'], out["mu_q"], out["logvar_q"])
                        kl = kl_standard_normal(out["mu_z"], out["logvar_z"]) # [B]
                        fk_err2 = fk_mse_from_qfeat_wrapper(self.env, out["mu_q"], h_world_batch, basexy_norm_type=self.basexy_norm_type, ori_weight=self.fk_ori_weight) # fk computation based on predicted mu
                        current_beta = get_beta(ep, self.epochs, self.beta) # dynamic beta
                        loss = torch.mean(rec + current_beta * kl + self.lambda_fk * fk_err2)
                        
                        val_batch_size = h_world_batch.shape[0] # last batch size might not equal self.batch_size
                        total_val_loss += float(loss.item()) * val_batch_size
                        total_val_rec += float(torch.mean(rec).item()) * val_batch_size
                        total_val_kl += float(torch.mean(kl).item()) * val_batch_size
                        total_val_fk += float(torch.mean(fk_err2).item()) * val_batch_size
                        val_n_seen += val_batch_size
                avg_val_loss = total_val_loss/val_n_seen
                wandb_metrics.update({
                    "val/loss": avg_val_loss,
                    "val/rec": total_val_rec/val_n_seen,
                    "val/kl": total_val_kl/val_n_seen,
                    "val/fk_mu": total_val_fk/val_n_seen
                })
                # option to save model checkpoint with best val loss -- only kick in when half of the epochs have passed
                if save_best_val_ckpt and save_path and ep > self.epochs / 2:
                    if avg_val_loss < lowest_loss_so_far:
                        print(f"Epoch {ep}: Val loss improved from {lowest_loss_so_far:.4f} to {avg_val_loss:.4f}. Saving to {save_path}")
                        lowest_loss_so_far = avg_val_loss
                        self.save(save_path)
                self._model.train()
                    
            # wandb tracking
            if self.wandb_run is not None:
                wandb.log(wandb_metrics, step=ep)
            
        self._model.eval()

    def sample(self, h_world: np.ndarray, c_world: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float = 1.0) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # move conditioning inputs into torch
        c_feat = c_world_to_feat(self.env, c_world, basexy_norm_type=self.basexy_norm_type, add_fourier_feat=self.add_fourier_feat, fourier_B=self.fourier_B)
        c_feat_batch = torch.from_numpy(c_feat.astype(np.float32)).to(self.device) # [B, d_c_feat]
        B = c_feat_batch.shape[0] # batch size

        # sample z ~ N(0, I) for each (b, s) (batch element * sample index)
        z = torch.randn((B * n_samples, self.z_dim), device=self.device) # [B*s, z_dim]
        # repeat each conditioning vector to match the z latent samples
        c_featb_repeated = c_feat_batch.repeat_interleave(n_samples, dim=0) #[B*S, d_c_feat]

        with torch.no_grad():
            mu_q, logvar_q = self._model.decode(c_featb_repeated, z) #[B*S, d_q_feat]
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            q_sample_feat = mu_q + std_q * eps * sampling_temperature # sampling from N(mu_q, diag(var_q))

        q_sample_feat_np = q_sample_feat.detach().cpu().numpy().astype(np.float32) # [B*S, d_q_feat]
        q_sample_world = q_feat_to_world(self.env, q_sample_feat_np, h_world=h_world, basexy_norm_type=self.basexy_norm_type) # [B*S, d_q]
        q_sample_world = q_sample_world.reshape(B, n_samples, self.d_q)
        return q_sample_world
    
    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_q": self.d_q,
                "d_q_feat": self._model.d_q_feat,
                "d_c_feat": self._model.d_c_feat,
                "z_dim": self.z_dim,
                "hidden_dim": self.hidden_dim,
                "num_enc_blocks": self.num_enc_blocks,
                "num_dec_blocks": self.num_dec_blocks,
                # "enc_hidden": list(self.enc_hidden),
                # "dec_hidden": list(self.dec_hidden),
                "beta": self.beta,
                "basexy_norm_type": self.basexy_norm_type,
                "state_dict": self._model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, env: Any, path: str | Path, device: str = "cpu") -> "CVAEConditionalSampler":
        path = Path(path)
        ckpt = torch.load(path, map_location=device)

        sampler = cls(
            env=env,
            d_q=int(ckpt["d_q"]),
            d_q_feat=int(ckpt["d_q_feat"]),
            d_c_feat=int(ckpt["d_c_feat"]),
            z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_enc_blocks=ckpt["num_enc_blocks"],
            num_dec_blocks=ckpt["num_dec_blocks"],
            # enc_hidden=tuple(ckpt["enc_hidden"]),
            # dec_hidden=tuple(ckpt["dec_hidden"]),
            beta=ckpt["beta"],
            basexy_norm_type=ckpt['basexy_norm_type'],
            device=device,
        )

        model = ConditionalVAE(
            d_c_feat=ckpt["d_c_feat"],
            d_q_feat=ckpt["d_q_feat"],
            z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_enc_blocks=ckpt["num_enc_blocks"],
            num_dec_blocks=ckpt["num_dec_blocks"],
            # enc_hidden=ckpt["enc_hidden"],
            # dec_hidden=ckpt["dec_hidden"],
        ).to(device)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sampler._model = model
        return sampler

    def count_parameters(self):
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        