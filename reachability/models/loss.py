import torch

def fk_mse_from_qfeat(mu_q: torch.Tensor, H: torch.Tensor, L: float, eps: float=1e-8) -> torch.Tensor:
    """
    mu_q: [B, 4] = (x, y, cos, sin)  (decoder mean)
    H: [B, 2] = (hx, hy)
    returns: [B] per-example squared FK error ||hand - H||^2
    """
    x = mu_q[:, 0]
    y = mu_q[:, 1]
    c = mu_q[:, 2]
    s = mu_q[:, 3]

    # normalize c, s onto the unit circle
    r = torch.sqrt(c * c + s * s + eps)
    c = c/r
    s = s/r

    hx_pred = x + L * c
    hy_pred = y + L * s

    return (hx_pred - H[:, 0]) ** 2 + (hy_pred - H[:, 1]) ** 2