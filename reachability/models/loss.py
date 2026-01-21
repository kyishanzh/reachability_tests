import torch
from reachability.utils.utils import revert_relative_to_worldscale

def normalize_to_unit_circle(cos, sin, eps):
    r = torch.sqrt(cos * cos + sin * sin + eps)
    return cos/r, sin/r

def fk_mse_from_qfeat_wrapper(env, mu_q: torch.Tensor, h_world: torch.Tensor, eps: float=1e-6, basexy_norm_type: str = "relative") -> torch.Tensor:
    if env.name == 'Simple':
        return fk_mse_from_qfeat_simple(env, mu_q, h_world, eps, basexy_norm_type=basexy_norm_type)
    elif env.name == 'RotaryLink':
        return fk_mse_from_qfeat_rotary(env, mu_q, h_world, eps, basexy_norm_type=basexy_norm_type)
    else:
        raise ValueError(f"Unknown environment type: {env.name}")

def fk_mse_from_qfeat_simple(env, mu_q: torch.Tensor, h_world: torch.Tensor, eps: float=1e-8, basexy_norm_type: str = "relative") -> torch.Tensor:
    """
    mu_q: [B, 4] = (normalized x in [-1, 1], normalized y in [-1, 1], cos(theta), sin(theta))  (decoder mean)
    H: [B, 2] = (hx, hy) in world coordinates
    returns: [B] per-example squared FK error ||hand - H||^2
    """
    L = env.L
    x = mu_q[:, 0]
    y = mu_q[:, 1]
    if basexy_norm_type == "relative":
        x_reverted = revert_relative_to_worldscale(env, x, h_world[:, 0])
        y_reverted = revert_relative_to_worldscale(env, y, h_world[:, 1])
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")
    cos_theta = mu_q[:, 2]
    sin_theta = mu_q[:, 3]

    # normalize c, s onto the unit circle - force the model to only encode orientation with c, s features
    cos_theta, sin_theta = normalize_to_unit_circle(cos_theta, sin_theta, eps)

    hx_pred = x_reverted + L * cos_theta
    hy_pred = y_reverted + L * sin_theta

    return (hx_pred - h_world[:, 0]) ** 2 + (hy_pred - h_world[:, 1]) ** 2

def fk_mse_from_qfeat_rotary(env, mu_q: torch.Tensor, h_world: torch.Tensor, eps: float=1e-8, basexy_norm_type: str = "relative") -> torch.Tensor:
    """
    mu_q: [B, 8] = (normalized x in [-1, 1], normalized y in [-1, 1], cos(psi), sin(psi), cos(theta1), sin(theta1), cos(theta2), sin(theta2))
    H: [B, 2] = (hx, hy) in world coordinates!
    """
    x = mu_q[:, 0]
    y = mu_q[:, 1]
    if basexy_norm_type == "relative":
        x_reverted = revert_relative_to_worldscale(env, x, h_world[:, 0])
        y_reverted = revert_relative_to_worldscale(env, y, h_world[:, 1])
    else: # add additional transformations in the future here
        raise RuntimeError(f"Only basexy_norm_type = relative supported right now, unknown mode {basexy_norm_type} was provided.")

    # normalize c, s onto the unit circle - force the model to only encode orientation with c, s features -> do for all angles
    c_psi, s_psi = normalize_to_unit_circle(mu_q[:, 2], mu_q[:, 3], eps)
    c_theta1, s_theta1 = normalize_to_unit_circle(mu_q[:, 4], mu_q[:, 5], eps)
    c_theta2, s_theta2 = normalize_to_unit_circle(mu_q[:, 6], mu_q[:, 7], eps)

    # cos(t1 + t2) = cos(t1)*cos(t2) - sin(t1)*sin(t2)
    # sin(t1 + t2) = sin(t1)*cos(t2) + cos(t1)*sin(t2)
    c_t1plust2 = c_theta1 * c_theta2 - s_theta1 * s_theta2
    s_t1plust2 = s_theta1 * c_theta2 + c_theta1 * s_theta2

    # local frame FK (relative to base):
    L1, L2 = env.link_lengths[0], env.link_lengths[1]
    local_x = L1 * c_theta1 + L2 * c_t1plust2
    local_y = L1 * s_theta1 + L2 * s_t1plust2

    # local frame -> world frame transformation (rotation matrix using psi)
    hx_pred = x_reverted + c_psi * local_x - s_psi * local_y
    hy_pred = y_reverted + s_psi * local_x + c_psi * local_y

    return (hx_pred - h_world[:, 0]) ** 2 + (hy_pred - h_world[:, 1]) ** 2 # MSE
