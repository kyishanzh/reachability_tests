from typing import Sequence
import torch
import torch.nn as nn

# Architectural blocks

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

class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(), # swish activation
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_blocks: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        # 1. Project input to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # 2. Stack Residual Blocks
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        # 3. Final LayerNorm + output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)

# Training helpers

def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """x, mu, logvar: shape [B, D]
    returns: [B] NLL per example (sum over D)
    ^ math: return -log(p(x | mu, var)) -> minimize neg log-likelihood"""
    # 0.5 * [(x - mu)^2 / var + logvar + log 2pi]
    return 0.5 * torch.sum(
        ((x - mu) ** 2) * torch.exp(-logvar) + logvar + torch.log(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype)), # breaks when use torch.Tensor ?!
        dim=-1
    )

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """D_KL[N(mu, diag(var)) || N(0, I)] per example: shape [B]"""
    return 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - 1.0 - logvar, dim=-1) # dim=-1 -> sum across dimensions (to get KL per sample)

def get_beta(epoch: int, total_epochs: int, target_beta: float, cycles: int = 1) -> float:
    """Cyclic annealing schedule."""
    # simple linear warmup
    warmup_epochs = total_epochs // 2
    if epoch >= warmup_epochs:
        return target_beta
    return target_beta * (epoch / warmup_epochs)