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
from reachability.utils.utils import q_to_qfeat, qfeat_to_q, grad_global_norm, h_to_hnorm
from reachability.models.cvae_helpers import MLP, ResidualMLP, gaussian_nll, kl_standard_normal, get_beta

class ConditionalVAE(nn.Module):
    """
    Encoder: (H, Q_feat) -> (mu_z, logvar_z) for Pr(z)
    Decoder: (H, z) -> (mu_q, logvar_q) for Pr(Q_feat)
    """
    def __init__(
            self,
            dH: int,
            dQ_feat: int,
            z_dim: int,
            # resnet setup
            hidden_dim: int = 512,
            num_blocks: int = 3,
            # vanilla MLP setup
            # enc_hidden: Sequence[int],
            # dec_hidden: Sequence[int],
            logvar_clip: float = 6.0
    ):
        super().__init__()
        self.dH = dH
        self.dQ_feat = dQ_feat
        self.z_dim = z_dim
        self.logvar_clip = logvar_clip

        # self.encoder = MLP(dH + dQ_feat, enc_hidden, 2 * z_dim) # [mu, logvar]
        self.encoder = ResidualMLP(
            in_dim=dH + dQ_feat,
            hidden_dim=hidden_dim,
            out_dim=2 * z_dim,
            num_blocks=num_blocks
        )
        # self.decoder = MLP(dH + z_dim, dec_hidden, 2 * dQ_feat) # [mu, logvar]
        self.decoder = ResidualMLP(
            in_dim=dH + z_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * dQ_feat,
            num_blocks=num_blocks
        )

    def encode(self, H: torch.Tensor, Q_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | H, Q_feat)"""
        x = torch.cat([H, Q_feat], dim=-1)
        out = self.encoder(x)
        mu, logvar = out[:, :self.z_dim], out[:, self.z_dim:]
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps - reparameterization trick!"""
        std = torch.exp(0.5 * logvar) # recover std dev from logvar
        eps = torch.randn_like(std) # eps ~ N(0, I) (same shape as std)
        return mu + std * eps

    def decode(self, H: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(Q | H, z)"""
        x = torch.cat([H, z], dim=-1)
        out = self.decoder(x)
        mu, logvar = out[:, :self.dQ_feat], out[:, self.dQ_feat:]
        logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)
        return mu, logvar
    
    def forward(self, H: torch.Tensor, Q_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(H, Q_feat)
        z = self.reparameterize(mu_z, logvar_z)
        mu_q, logvar_q = self.decode(H, z)
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
    dQ: int
    z_dim: int # test more vals -- theoretically need at least DoF of environment to have a chance at doing well
    # enc_hidden: tuple[int, ...] = (128, 128)
    # dec_hidden: tuple[int, ...] = (128, 128)
    hidden_dim: int = 512
    num_blocks: int = 3
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    beta: float = 1.0
    device: str = "cpu"
    seed: int = 0
    lambda_fk: float = 0.0 # FK penalty
    wandb_run: object | None = None
    basexy_norm_type: str = "bound"
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

        self.dH_feat = train_loader.dataset.dH_feat #TODO: Change self.dH to self.dH_feat?
        self.dQ_feat = train_loader.dataset.dQ_feat
        print(f"dH = {self.dH_feat}, dQ_feat = {self.dQ_feat}")

        self._model = ConditionalVAE(
            dH=self.dH_feat,
            dQ_feat=self.dQ_feat,
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            # enc_hidden=self.enc_hidden,
            # dec_hidden=self.dec_hidden
        ).to(self.device)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        lowest_loss_so_far = float("inf")
        for ep in range(self.epochs):
            total_loss, total_rec, total_kl, total_fk = 0.0, 0.0, 0.0, 0.0
            n_seen = 0

            # Train loop
            for batch in train_loader:
                Hbatch = batch['H_feat']
                Qbatch = batch['Q_feat']
                Hraw = torch.from_numpy(batch["H_raw"]).to(self.device)

                out = self._model(Hbatch, Qbatch)
                rec = gaussian_nll(Qbatch, out["mu_q"], out["logvar_q"]) # [B]
                kl = kl_standard_normal(out["mu_z"], out["logvar_z"]) # [B]
                fk_err2 = fk_mse_from_qfeat_wrapper(self.env, out["mu_q"], Hraw, basexy_norm_type=self.basexy_norm_type)
                current_beta = self.beta #get_beta(ep, self.epochs, self.beta) # dynamic beta
                loss = torch.mean(rec + current_beta * kl + self.lambda_fk * fk_err2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                batchsize = Hbatch.shape[0] # last batch size might not equal self.batch_size
                total_loss += float(loss.item()) * batchsize
                total_rec += float(torch.mean(rec).item()) * batchsize
                total_kl += float(torch.mean(kl).item()) * batchsize
                total_fk += float(torch.mean(fk_err2).item()) * batchsize
                n_seen += batchsize

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
                        Hraw = torch.from_numpy(batch["H_raw"]).to(self.device)
                        out = self._model(batch['H_feat'], batch['Q_feat'])
                        rec = gaussian_nll(batch['Q_feat'], out["mu_q"], out["logvar_q"])
                        kl = kl_standard_normal(out["mu_z"], out["logvar_z"]) # [B]
                        fk_err2 = fk_mse_from_qfeat_wrapper(self.env, out["mu_q"], Hraw, basexy_norm_type=self.basexy_norm_type)
                        current_beta = get_beta(ep, self.epochs, self.beta) # dynamic beta
                        loss = torch.mean(rec + current_beta * kl + self.lambda_fk * fk_err2)
                        
                        batchsize = Hraw.shape[0] # last batch size might not equal self.batch_size
                        total_val_loss += float(loss.item()) * batchsize
                        total_val_rec += float(torch.mean(rec).item()) * batchsize
                        total_val_kl += float(torch.mean(kl).item()) * batchsize
                        total_val_fk += float(torch.mean(fk_err2).item()) * batchsize
                        val_n_seen += batchsize
                avg_val_loss = total_val_loss/val_n_seen
                wandb_metrics.update({
                    "val/loss": avg_val_loss,
                    "val/rec": total_val_rec/val_n_seen,
                    "val/kl": total_val_kl/val_n_seen,
                    "val/fk": total_val_fk/val_n_seen
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

    def sample(
        self,
        H: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        sampling_temperature: float = 1.0
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # move conditioning inputs into torch
        H_feat = h_to_hnorm(self.env, H, basexy_norm_type=self.basexy_norm_type, add_fourier_feat=self.add_fourier_feat, fourier_B=self.fourier_B)
        Hbatch = torch.from_numpy(H_feat.astype(np.float32)).to(self.device) # [B, dH]
        B = Hbatch.shape[0] # batch size

        # sample z ~ N(0, I) for each (b, s) (batch element * sample index)
        z = torch.randn((B * n_samples, self.z_dim), device=self.device) # [B*s, z_dim]
        # repeat each conditioning vector to match the z latent samples
        Hrep = Hbatch.repeat_interleave(n_samples, dim=0) #[B*S, dH]

        with torch.no_grad():
            mu_q, logvar_q = self._model.decode(Hrep, z) #[B*S, dQ_feat]
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            Q_feat = mu_q + std_q * eps * sampling_temperature # sampling from N(mu_q, diag(var_q))

        Q_feat_np = Q_feat.detach().cpu().numpy().astype(np.float32) # [B*S, dQfeat]
        Q_np = qfeat_to_q(self.env, Q_feat_np, basexy_norm_type=self.basexy_norm_type) # [B*S, dQ]
        Q_np = Q_np.reshape(B, n_samples, self.dQ)
        return Q_np
    
    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dQ": self.dQ,
                "dH": self._model.dH,
                "dQ_feat": self._model.dQ_feat,
                "z_dim": self.z_dim,
                "hidden_dim": self.hidden_dim,
                "num_blocks": self.num_blocks,
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
            dQ=int(ckpt["dQ"]),
            z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_blocks=ckpt["num_blocks"],
            # enc_hidden=tuple(ckpt["enc_hidden"]),
            # dec_hidden=tuple(ckpt["dec_hidden"]),
            beta=ckpt["beta"],
            basexy_norm_type=ckpt['basexy_norm_type'],
            device=device,
        )

        model = ConditionalVAE(
            dH=ckpt["dH"],
            dQ_feat=ckpt["dQ_feat"],
            z_dim=ckpt["z_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_blocks=ckpt["num_blocks"],
            # enc_hidden=ckpt["enc_hidden"],
            # dec_hidden=ckpt["dec_hidden"],
        ).to(device)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sampler._model = model
        return sampler
