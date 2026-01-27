from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Any, List

import wandb
import numpy as np
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from reachability.data.datasets import Normalizer
from reachability.models.submodels import ResidualMLP
from reachability.models.base import ConditionalGenerativeModel
from reachability.models.loss import fk_mse_from_qfeat_wrapper
from reachability.models.features import c_world_to_feat, q_feat_to_world, q_world_to_feat
from reachability.utils.utils import grad_global_norm

class CINNv2(nn.Module): # cINN architecture v2, inspired by https://github.com/vislearn/conditional_INNs/tree/master/mnist_minimal_example
    def __init__(self, hidden_dim, d_c_feat, d_q_feat, num_blocks, num_subnet_blocks=1, lr=5e-4, clamp=1.0, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_c_feat = d_c_feat # dimension of conditioning node
        self.d_q_feat = d_q_feat
        self.num_blocks = num_blocks
        self.num_subnet_blocks = num_subnet_blocks
        self.clamp = clamp
        self.dropout = dropout

        # build the model
        self.inn = self.build_inn()

        # initialize parameters with small random values
        self.trainable_parameters = [p for p in self.inn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p) # make everything smaller
        # TODO: potentially remove this manual loop and add to subnet to only zero the final layer of the subnet

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        # subnet
        def subnet(ch_in, ch_out):
            # net = nn.Sequential(nn.Linear(ch_in, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, ch_out))
            net = ResidualMLP(
                in_dim=ch_in, hidden_dim=self.hidden_dim, out_dim=ch_out, num_blocks=self.num_subnet_blocks, dropout=self.dropout, zero_init_last=True
            )# zero init the final layer to start model as the identity transform: x * exp(0) + 0 = x
            return net
        # ConditionNode: the graph takes an extra input c in R^10 (the one-hot label) -> this conditioning is fed into each coupling block so the transform becomes class-specific
        cond = Ff.ConditionNode(self.d_c_feat)

        # entry to model: InputNode
        nodes = [Ff.InputNode(1, self.d_q_feat)] # flatten to data x shape [1, dQfeat]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {})) # flatten x -> R^{dQfeat}
        # notation notes: Ff.Node(prev, module, ...) = take the output of prev, apply this invertible module, and produce a new node

        # repeated invertible blocks - {num_blocks} layers (stacking many of these gives an expressive bijection)
        for k in range(self.num_blocks): # TODO: figure out if performance is better with below manual blocks or using Fm.AllInOneBlock
            # 1. permuting step to mix dimensions
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            # 2. add ActNorm to stabilize the scale before the coupling transform
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))
            # 3. GLOWCoupling block (type of affine coupling block) [the actual transformation]
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {'subnet_constructor': subnet, 'clamp': self.clamp}, conditions=cond)) # coupling blocks read conditioning information through cond node

        nodes.append(Ff.OutputNode(nodes[-1])) # mark last node of graph as the graph output
        return Ff.GraphINN(nodes + [cond], verbose=False) # nodes + [cond] = single flat list containing all nodes on the data path, the condition node, and the output node

    def forward(self, q_feat: torch.Tensor, c_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # SequenceINN expects conditions as a list when cond indices are used
        z, log_jacobian = self.inn(q_feat, c=[c_feat], jac=True) # store jacobian to compute likelihood for loss (during training)
        return z, log_jacobian
    
    @torch.no_grad()
    def reverse(self, z: torch.Tensor, c_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_feat, log_det = self.inn(z, c=[c_feat], rev=True)
        return q_feat, log_det

class SimpleCINN(nn.Module):
    """
    Conditional INN (normalizing flow) for q_feat in R^4 conditioned on H in R^2. Uses FrEIA SequenceINN + AllInOneBlock.

    Forward: (q_feat, H) -> (z, log_det_J)
    Reverse: (z, H) -> (q_feat, log_det_J)
    """
    def __init__(self, d_q_feat: int, d_c_feat: int, num_blocks: int, hidden_dim: int, clamp: float):
        super().__init__()
        self.d_q_feat = d_q_feat
        self.d_c_feat = d_c_feat
        self.hidden_dim = hidden_dim

        self.inn = Ff.SequenceINN(d_q_feat)

        def subnet_fc(dims_in: int, dims_out: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(dims_in, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, dims_out)
            )
        
        # attach condition to every block via cond_shape
        # SequenceINN will pass c=[cond] into invertible modules
        for _ in range(num_blocks):
            self.inn.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_fc,
                cond=0,
                cond_shape=(d_c_feat,),
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

    # model
    hidden_dim: int
    d_q: int
    d_q_feat: int
    d_c_feat: int
    num_blocks: int
    num_subnet_blocks: int
    clamp: float = 1.0
    basexy_norm_type: str = "relative"

    # training
    device: str = "cpu"
    epochs: int = 60
    seed: int = 0
    lr: float = 5e-4
    lr_milestones: List[int] = field(default_factory=lambda: [20, 40])
    lr_gamma: float = 0.1
    batch_size: int = 256 # mnist cINN example also uses this batch size
    grad_clip: float = 10.
    dropout: float = 0.0

    # optional constraint shaping
    lambda_fk: float = 0.0

    _model: SimpleCINN | CINNv2 | None = None
    wandb_run: object | None = None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, val_frequency: int = 10) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"d_c_feat = {self.d_c_feat}, d_q_feat = {self.d_q_feat}")

        # self._model = SimpleCINN(
        #     dQ_feat=self.dQ_feat,
        #     dCond=self.dH,
        #     num_blocks=int(self.num_blocks),
        #     hidden_dim=self.hidden_dim,
        #     clamp=float(self.clamp)
        # ).to(self.device)
        self._model = CINNv2(
            hidden_dim=self.hidden_dim,
            d_c_feat=self.d_c_feat,
            d_q_feat=self.d_q_feat,
            num_blocks=self.num_blocks,
            num_subnet_blocks=self.num_subnet_blocks,
            lr=self.lr,
            clamp=self.clamp,
            dropout=self.dropout
        ).to(self.device)

        print("num trainable params in model: ", self.count_parameters())

        # opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        opt = self._model.optimizer
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, self.lr_milestones, gamma=self.lr_gamma)

        self.normalizer = Normalizer(self.d_q_feat, device=self.device)
        all_q_feats = []
        for batch in train_loader:
            all_q_feats.append(batch['q_feat'])
        full_data = torch.cat(all_q_feats).to(self.device)
        self.normalizer.fit(full_data)
        del full_data, all_q_feats
        print(f"normalizer mean: {self.normalizer.mean.cpu().numpy()} | normalizer std: {self.normalizer.std.cpu().numpy()}")
        
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
                h_world_batch = torch.from_numpy(batch["h_world"]).to(self.device)

                q_feat_batch = self.normalizer.normalize(q_feat_batch)
                # c_feat_batch = self.normalizer.normalize(c_feat_batch) # TODO: when conditioning becomes meaningful, decide if we want to make a self.cond_normalizer for conditioning data
                
                # forward: z = f(q; c), log_det = log|det df/dx|
                z, log_detj = self._model(q_feat_batch, c_feat_batch) # log_j = log absolute determinant of the Jacobian accumulated across the network -> shape [B]
                assert z.shape[0] == len(q_feat_batch)
                assert z.shape[1] == self.d_q_feat

                # NLL (dropping consts) for standard Normal base density:
                # -log p(q | c) = 0.5||z||^2 - log_det + const
                z_term = torch.mean(z ** 2) / 2 # z = [B, D] shape                
                log_detj_term = -torch.mean(log_detj) / self.d_q_feat
                nll = z_term + log_detj_term

                # extra regularizer - FK consistency constraint
                fk = torch.zeros_like(nll) # default FK loss is 0
                if self.lambda_fk > 0.0:
                    # sample z~N, invert to q_hat, penalize fk(q_hat)
                    z_samp = torch.randn_like(z)
                    q_hat, _ = self._model.reverse(z_samp, c_feat_batch)
                    fk = fk_mse_from_qfeat_wrapper(self.env, q_hat, h_world_batch, basexy_norm_type=self.basexy_norm_type)

                # backprop
                loss = torch.mean(nll + self.lambda_fk * fk)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self._model.trainable_parameters, self.grad_clip)
                grad_norm_post_clip = grad_global_norm(self._model.trainable_parameters)

                opt.step()
                opt.zero_grad(set_to_none=True)

                # batch level metrics to log to wandb
                with torch.no_grad():
                    wandb_metrics = {
                        "train/loss": nll.item(),
                        "train/z_term": z_term.item(),
                        "train/logdet_term": log_detj_term.item(),

                        "train/log_detj_mean": log_detj.mean().item(),
                        "train/log_detj_std": log_detj.std(correction=0).item(), # no Bessel's correction: https://docs.pytorch.org/docs/stable/generated/torch.std.html
                        "train/log_detj_max": log_detj.max().item(),
                        "train/log_detj_min": log_detj.min().item(),

                        "train/z_sq_mean": (z ** 2).mean().item(),
                        "train/z_abs_max": z.abs().max().item(),
                        "train/z_norm_mean": z.view(z.size(0), -1).norm(dim=1).mean().item(),
                        "train/z_norm_max": z.view(z.size(0), -1).norm(dim=1).max().item(),

                        "train/isfinite_nll": float(torch.isfinite(nll)),
                        "train/isfinite_z_term": float(torch.isfinite(z_term)),
                        "train/isfinite_logdet_term": float(torch.isfinite(log_detj_term)),
                        
                        "train/grad_norm_pre_clip": grad_norm,
                        "train/grad_norm_post_clip": grad_norm_post_clip
                    }

            # Validation loop
            if val_loader is not None and (ep % val_frequency == 0):
                total_val_loss, total_val_nll, total_val_fk = 0.0, 0.0, 0.0
                val_n_seen = 0
                self._model.eval() # switch model into eval mode - disable dropout, does not update batchnorm, etc. - but does not stop gradient tracking!
                with torch.no_grad(): # disable autograd within the entire block
                    for batch in val_loader:
                        q_feat_batch = self.normalizer.normalize(batch['q_feat'])
                        c_feat_batch = batch['c_feat']
                        # c_feat_batch = self.normalizer.normalize(batch['c_feat']) # TODO: make sure this matches whatever we do in the training loop
                        h_world_batch = torch.from_numpy(batch["h_world"]).to(self.device)
                        z, log_detj = self._model(q_feat_batch, c_feat_batch)
                        nll = torch.mean(z ** 2) / 2 - torch.mean(log_detj) / self.d_q_feat
                        fk = torch.zeros_like(nll) # default FK loss is 0
                        if self.lambda_fk > 0.0:
                            # sample z~N, invert to q_hat, penalize fk(q_hat)
                            z_samp = torch.randn_like(z)
                            q_hat, _ = self._model.reverse(z_samp, c_feat_batch)
                            fk = fk_mse_from_qfeat_wrapper(self.env, q_hat, h_world_batch, basexy_norm_type=self.basexy_norm_type)
                        loss = torch.mean(nll + self.lambda_fk * fk)
                        
                        batchsize = h_world_batch.shape[0] # last batch size might not equal self.batch_size
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
            
            # lr scheduler update
            scheduler.step()

        self._model.eval()

    def sample(self, h_world: np.ndarray, c_world: np.ndarray, n_samples: int, rng: np.random.Generator, sampling_temperature: float) -> np.ndarray: # probably something wrong with handling normalization here
        """
        h_world: [B,2] numpy
        returns: [B, n_samples, 3] numpy Q=(x,y,theta)
        """
        if self._model is None:
            raise RuntimeError("Call fit() (or load()) before sample().")
        
        # enforce numpy rng as sole source of randomness for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        # move conditioning inputs into torch
        c_feat = c_world_to_feat(self.env, c_world, basexy_norm_type=self.basexy_norm_type)
        c_feat_batch = torch.from_numpy(c_feat.astype(np.float32)).to(self.device) # [B, d_c]
        B = c_feat_batch.shape[0] # batch size

        # z ~ N(0, I)
        z = torch.randn((B * n_samples, self.d_q_feat), device=self.device)
        c_featb_repeated = c_feat_batch.repeat_interleave(n_samples, dim=0)

        with torch.no_grad():
            q_sample_feat, _ = self._model.reverse(z, c_featb_repeated)
        
        q_sample_feat = q_sample_feat.squeeze(1)
        q_sample_feat = self.normalizer.unnormalize(q_sample_feat)
        print(f"DEBUG: q_sample_feat.squeeze(1) shape from model: {q_sample_feat.shape}")
        q_sample_feat_np = q_sample_feat.detach().cpu().numpy().astype(np.float32)
        q_sample_world = q_feat_to_world(self.env, q_sample_feat_np, h_world=h_world, basexy_norm_type=self.basexy_norm_type)
        print(f"DEBUG: q_sample_world shape after conversion: {q_sample_world.shape}")
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
                "num_subnet_blocks": int(self.num_subnet_blocks),
                "clamp": float(self.clamp),
                "basexy_norm_type": self.basexy_norm_type,
                # model param
                "state_dict": self._model.state_dict(),
                "normalizer_mean": self.normalizer.mean,
                "normalizer_std": self.normalizer.std
            },
            path
        )

    @classmethod
    def load(cls, env: Any, path: str | Path, device: str = "cpu") -> "CINNConditionalSampler":
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        
        sampler = cls(
            env=env,
            hidden_dim=int(ckpt["hidden_dim"]),
            d_q=int(ckpt["d_q"]),
            d_q_feat=int(ckpt["d_q_feat"]),
            d_c_feat=int(ckpt["d_c_feat"]),
            num_blocks=int(ckpt["num_blocks"]),
            num_subnet_blocks=int(ckpt["num_subnet_blocks"]),
            clamp=float(ckpt["clamp"]),
            basexy_norm_type=ckpt['basexy_norm_type'],
            device=device
        )

        # model = SimpleCINN(
        #     dQ_feat=sampler.dQ_feat,
        #     dCond=sampler.dH,
        #     n_blocks=int(sampler.n_blocks),
        #     hidden_dim=sampler.hidden,
        #     clamp=float(sampler.clamp),
        # ).to(device)
        model = CINNv2(
            hidden_dim=int(ckpt["hidden_dim"]),
            d_c_feat=int(ckpt["d_c_feat"]),
            d_q_feat=int(ckpt["d_q_feat"]),
            num_blocks=int(ckpt["num_blocks"]),
            clamp=float(ckpt["clamp"])
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
        