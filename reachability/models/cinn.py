from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any

import wandb
import numpy as np
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from reachability.models.base import ConditionalGenerativeModel
from reachability.models.loss import fk_mse_from_qfeat_wrapper
from reachability.utils.utils import q_to_qfeat, qfeat_to_q, h_to_hnorm

class SimpleCINN(nn.Module):
    """
    Conditional INN (normalizing flow) for q_feat in R^4 conditioned on H in R^2. Uses FrEIA SequenceINN + AllInOneBlock.

    Forward: (q_feat, H) -> (z, log_det_J)
    Reverse: (z, H) -> (q_feat, log_det_J)
    """
    def __init__(
            self,
            q_dim: int = 4,
            h_dim: int = 2,
            n_blocks: int = 8,
            hidden: int = 128,
            clamp: float = 2.0
    ):
        super().__init__()
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.hidden = hidden

        self.inn = Ff.SequenceINN(q_dim)

        def subnet_fc(dims_in: int, dims_out: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(dims_in, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, dims_out)
            )
        
        # attach condition to every block via cond_shape
        # SequenceINN will pass c=[cond] into invertible modules
        for _ in range(n_blocks):
            self.inn.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_fc,
                cond=0,
                cond_shape=(h_dim,),
                affine_clamping=clamp,
                permute_soft=True # very slow when working w >512 dims according to FrEIA documentation
            ) # https://vislearn.github.io/FrEIA/_build/html/FrEIA.modules.html

    def forward(self, q_feat: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # SequenceINN expects conditions as a list when cond indices are used
        z, log_jacobian = self.inn(q_feat, c=[H])
        return z, log_jacobian
    
    @torch.no_grad()
    def reverse(self, z: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_feat, log_det = self.inn(z, c=[H], rev=True)
        return q_feat, log_det

@dataclass
class CINNConditionalSampler(ConditionalGenerativeModel):
    """Wraps a SimpleCINN into fit/sample API."""
    env: Any
    device: str = "cpu"
    n_blocks: int = 8
    hidden: int = 128
    clamp: float = 2.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    epochs: int = 50
    batch_size: int = 256
    dQ: int = 4
    dH: int = 2
    dQ_feat: int = 4 # MAKE THIS CLEANER SOMEHOW
    basexy_norm_type: str = "bound"

    # optional constraint shaping
    lambda_fk: float = 0.0

    _model: SimpleCINN | None = None
    wandb_run: object | None = None

    def fit(self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        val_frequency: int = 10
    ) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.dH = train_loader.dataset.dH
        self.dQ_feat = train_loader.dataset.dQ_feat
        print(f"dH = {self.dH}, dQ_feat = {self.dQ_feat}")

        self._model = SimpleCINN(
            q_dim=self.dQ_feat,
            h_dim=self.dH,
            n_blocks=int(self.n_blocks),
            hidden=self.hidden,
            clamp=float(self.clamp)
        ).to(self.device)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        for ep in range(self.epochs):
            total_loss = 0.0
            total_nll = 0.0
            total_fk = 0.0
            n_seen = 0
            
            # Train loop
            for batch in train_loader:
                Hbatch = batch['H_feat']
                Qbatch = batch['Q_feat']
                Hraw = torch.from_numpy(batch["H_raw"]).to(self.device)

                # forward: z = f(q; H), log_det = log|det df/dx|
                z, log_det = self._model(Qbatch, Hbatch)

                # NLL (dropping consts) for standard Normal base density:
                # -log p(q | H) = 0.5||z||^2 - log_det + const
                nll = 0.5 * torch.sum(z * z, dim=1) - log_det

                # extra regularizer - FK consistency constraint
                fk = torch.zeros_like(nll) # default FK loss is 0
                if self.lambda_fk > 0.0:
                    # sample z~N, invert to q_hat, penalize fk(q_hat)
                    z_samp = torch.randn_like(z)
                    q_hat, _ = self._model.reverse(z_samp, Hbatch)
                    fk = fk_mse_from_qfeat_wrapper(self.env, q_hat, Hraw, basexy_norm_type=self.basexy_norm_type)

                loss = torch.mean(nll + self.lambda_fk * fk)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                batchsize = Hbatch.shape[0] # last batch size might not equal self.batch_size
                total_loss += float(loss.item()) * batchsize
                total_nll += float(torch.mean(nll).item()) * batchsize
                total_fk += float(torch.mean(fk).item()) * batchsize
                n_seen += batchsize

            avg_loss = total_loss/n_seen
            avg_nll = total_nll/n_seen
            avg_fk = total_fk/n_seen

            wandb_metrics = {
                "train/loss": avg_loss,
                "train/nll": avg_nll,
                "train/fk": avg_fk,
                "epoch": ep,
            }

            # Validation loop
            if val_loader is not None and (ep % val_frequency == 0):
                total_val_loss, total_val_nll, total_val_fk = 0.0, 0.0, 0.0
                val_n_seen = 0
                self._model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        Hraw = torch.from_numpy(batch["H_raw"]).to(self.device)
                        z, log_det = self._model(batch['Q_feat'], batch['H_feat'])
                        nll = 0.5 * torch.sum(z * z, dim=1) - log_det
                        fk = torch.zeros_like(nll) # default FK loss is 0
                        if self.lambda_fk > 0.0:
                            # sample z~N, invert to q_hat, penalize fk(q_hat)
                            z_samp = torch.randn_like(z)
                            q_hat, _ = self._model.reverse(z_samp, batch['H_feat'])
                            fk = fk_mse_from_qfeat_wrapper(self.env, q_hat, Hraw, basexy_norm_type=self.basexy_norm_type)
                        loss = torch.mean(nll + self.lambda_fk * fk)
                        
                        batchsize = Hraw.shape[0] # last batch size might not equal self.batch_size
                        total_val_loss += float(loss.item()) * batchsize
                        total_val_nll += float(torch.mean(nll).item()) * batchsize
                        total_val_fk += float(torch.mean(fk).item()) * batchsize
                        val_n_seen += batchsize
                wandb_metrics.update({
                    "val/loss": total_val_loss/val_n_seen,
                    "val/nll": total_val_nll/val_n_seen,
                    "val/fk": total_val_fk/val_n_seen
                })
                self._model.train()

            # wandb tracking
            if self.wandb_run is not None:
                wandb.log(wandb_metrics, step=ep)

        self._model.eval()

    def sample(self, H: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float) -> np.ndarray: # probably something wrong with handling normalization here
        """
        H: [B,2] numpy
        returns: [B, n_samples, 3] numpy Q=(x,y,theta)
        """
        if self._model is None:
            raise RuntimeError("Call fit() (or load()) before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # move conditioning inputs into torch
        H_feat = h_to_hnorm(self.env, H, basexy_norm_type=self.basexy_norm_type, rng=rng)
        Hbatch = torch.from_numpy(H_feat.astype(np.float32)).to(self.device) # [B, dH]
        B = Hbatch.shape[0] # batch size

        # z ~ N(0, I)
        z = torch.randn((B * n_samples, self.dQ_feat), device=self.device)
        Hrep = Hbatch.repeat_interleave(n_samples, dim=0)

        with torch.no_grad():
            q_feat, _ = self._model.reverse(z, Hrep)
        
        q_feat_np = q_feat.detach().cpu().numpy().astype(np.float32)
        Q_np = qfeat_to_q(self.env, q_feat_np, basexy_norm_type=self.basexy_norm_type)
        return Q_np.reshape(B, n_samples, self.dQ)
    
    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_blocks": int(self.n_blocks),
                "hidden": int(self.hidden),
                "clamp": float(self.clamp),
                "dQ": self.dQ,
                "dH": self.dH,
                "dQ_feat": self.dQ_feat,
                "basexy_norm_type": self.basexy_norm_type,
                "state_dict": self._model.state_dict(),
                
            },
            path
        )

    @classmethod
    def load(cls, env: Any, path: str | Path, device: str = "cpu") -> "CINNConditionalSampler":
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        
        sampler = cls(
            env=env,
            n_blocks=int(ckpt["n_blocks"]),
            hidden=int(ckpt["hidden"]),
            clamp=float(ckpt["clamp"]),
            dQ=int(ckpt["dQ"]),
            dH=int(ckpt["dH"]),
            dQ_feat=int(ckpt["dQ_feat"]),
            basexy_norm_type=ckpt['basexy_norm_type'],
            device=device
        )

        model = SimpleCINN(
            q_dim=sampler.dQ_feat,
            h_dim=sampler.dH,
            n_blocks=int(sampler.n_blocks),
            hidden=sampler.hidden,
            clamp=float(sampler.clamp),
        ).to(device)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        sampler._model = model
        return sampler
        