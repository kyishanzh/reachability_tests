from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

import yaml
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reachability.models.base import ConditionalGenerativeModel
from reachability.models.loss import fk_mse_from_qfeat
from reachability.utils.utils import q_to_qfeat_np as q_to_qfeat, qfeat_to_q_np as qfeat_to_q

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
            enc_hidden: Sequence[int],
            dec_hidden: Sequence[int],
            logvar_clip: float = 6.0
    ):
        super().__init__()
        self.dH = dH
        self.dQ_feat = dQ_feat
        self.z_dim = z_dim
        self.logvar_clip = logvar_clip

        self.encoder = MLP(dH + dQ_feat, enc_hidden, 2 * z_dim) # [mu, logvar]
        self.decoder = MLP(dH + z_dim, dec_hidden, 2 * dQ_feat) # [mu, logvar]

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

def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """x, mu, logvar: shape [B, D]
    returns: [B] NLL per example (sum over D)
    ^ math: return -log(p(x | mu, var)) -> minimize neg log-likelihood"""
    # 0.5 * [(x - mu)^2 / var + logvar + log 2pi]
    return 0.5 * torch.sum(
        ((x - mu) ** 2) * torch.exp(-logvar) + logvar + np.log(2.0 * np.pi),
        dim=-1
    )

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """D_KL[N(mu, diag(var)) || N(0, I)] per example: shape [B]"""
    return 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - 1.0 - logvar, dim=-1)

@dataclass
class CVAEConditionalSampler(ConditionalGenerativeModel):
    """Wraps a torch ConditionalVAE into fit/sample API."""
    z_dim: int = 4 # test more vals
    enc_hidden: tuple[int, ...] = (128, 128)
    dec_hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    beta: float = 1.0
    device: str = "cpu"
    seed: int = 0
    L: float = 1.0 # stick length from env config
    lambda_fk: float = 0.0 # FK penalty
    wandb_run: object | None = None

    _model: ConditionalVAE | None = None

    def fit(self, H_train: np.ndarray, Q_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        H = H_train.astype(np.float32)
        Q_feat = q_to_qfeat(Q_train.astype(np.float32))

        dH = H.shape[1]
        dQ_feat = Q_feat.shape[1]
        print(f"dH = {dH}, dQ_feat = {dQ_feat}")

        model = ConditionalVAE(
            dH = dH,
            dQ_feat=dQ_feat,
            z_dim=self.z_dim,
            enc_hidden=self.enc_hidden,
            dec_hidden=self.dec_hidden
        ).to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        n = H.shape[0]
        print(f"num samples = {n}")
        indices = np.arange(n)

        model.train()
        for ep in range(self.epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            total_rec = 0.0
            total_kl = 0.0
            total_fk = 0.0
            n_seen = 0
            
            for start in range(0, n, self.batch_size):
                idx = indices[start:start+self.batch_size]
                Hbatch = torch.from_numpy(H[idx]).to(self.device)
                Qbatch = torch.from_numpy(Q_feat[idx]).to(self.device)

                out = model(Hbatch, Qbatch)
                rec = gaussian_nll(Qbatch, out["mu_q"], out["logvar_q"]) # [B]
                kl = kl_standard_normal(out["mu_z"], out["logvar_z"]) # [B]
                fk_err2 = fk_mse_from_qfeat(out["mu_q"], Hbatch, L=self.L)
                loss = torch.mean(rec + self.beta * kl + self.lambda_fk * fk_err2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                batchsize = Hbatch.shape[0] # last batch size might not equal self.batch_size
                total_loss += float(loss.item()) * batchsize
                total_rec += float(torch.mean(rec).item()) * batchsize
                total_kl += float(torch.mean(kl).item()) * batchsize
                total_fk += float(torch.mean(fk_err2).item()) * batchsize
                n_seen += batchsize

            avg_loss = total_loss/n_seen
            avg_rec = total_rec/n_seen
            avg_kl = total_kl/n_seen
            avg_fk = total_fk/n_seen

            # wandb tracking
            if self.wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/recon": avg_rec,
                        "train/kl": avg_kl,
                        "train/fk": avg_fk,
                        "train/beta": self.beta,
                        "epoch": ep,
                    },
                    step=ep,
                )

        self._model = model.eval()

    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # move conditioning inputs into torch
        Hbatch = torch.from_numpy(H.astype(np.float32)).to(self.device) # [B, dH]
        B = Hbatch.shape[0] # batch size

        # sample z ~ N(0, I) for each (b, s) (batch element * sample index)
        z = torch.randn((B * n_samples, self.z_dim), device=self.device) # [B*s, z_dim]
        # repeat each conditioning vector to match the z latent samples
        Hrep = Hbatch.repeat_interleave(n_samples, dim=0) #[B*S, dH]

        with torch.no_grad():
            mu_q, logvar_q = self._model.decode(Hrep, z) #[B*S, dQ_feat]
            # std_q = torch.exp(0.5 * logvar_q)
            # eps = torch.randn_like(std_q)
            Q_feat = mu_q #+ std_q * eps # sampling from N(mu_q, diag(var_q)) -- noise makes performance worse w/o increasing diversity

        Q_feat_np = Q_feat.detach().cpu().numpy().astype(np.float32) # [B*S, 4]
        Q_np = qfeat_to_q(Q_feat_np) # [B*S, 3]
        Q_np = Q_np.reshape(B, n_samples, 3)
        return Q_np
    
    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "dH": self._model.dH,
                "dQ_feat": self._model.dQ_feat,
                "z_dim": self.z_dim,
                "enc_hidden": list(self.enc_hidden),
                "dec_hidden": list(self.dec_hidden),
                "beta": self.beta,
                "L": self.L,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "CVAEConditionalSampler":
        path = Path(path)
        ckpt = torch.load(path, map_location=device)

        sampler = cls(
            z_dim=ckpt["z_dim"],
            enc_hidden=tuple(ckpt["enc_hidden"]),
            dec_hidden=tuple(ckpt["dec_hidden"]),
            beta=ckpt["beta"],
            L=ckpt["L"],
            device=device,
        )

        model = ConditionalVAE(
            dH=ckpt["dH"],
            dQ_feat=ckpt["dQ_feat"],
            z_dim=ckpt["z_dim"],
            enc_hidden=ckpt["enc_hidden"],
            dec_hidden=ckpt["dec_hidden"],
        ).to(device)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sampler._model = model
        return sampler
    