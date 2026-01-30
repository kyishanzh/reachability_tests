import torch
from reachability.utils.utils import revert_relative_to_worldscale

def normalize_to_unit_circle(tensor: torch.Tensor, eps):
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    return tensor / (norm + eps)

def rotate_vector_torch(vectors: torch.Tensor, angles_or_vecs: torch.Tensor) -> torch.Tensor:
    """
    Rotates 2D vectors. Supports two modes for the rotation specifier.
    vectors: [B, 2] (x, y)
    angles_or_vecs: (1) angles (radians, tensor [B] or [B, 1]) or (2) rotation vector ((cos, sin), tensor [B,2])
    """
    x, y = vectors[:, 0], vectors[:, 1]
    if angles_or_vecs.shape[-1] == 1 or angles_or_vecs.ndim == 1:
        # Mode 1: Angles
        theta = angles_or_vecs.view(-1)
        c, s = torch.cos(theta), torch.sin(theta)
    else:
        # Mode 2: Vectors (cos, sin)
        c, s = angles_or_vecs[:, 0], angles_or_vecs[:, 1]
    
    x_rot = x * c - y * s
    y_rot = x * s + y * c
    # print("x_rot shape = ", x_rot.shape)
    return torch.stack([x_rot, y_rot], dim=1)

def fk_mse_from_qfeat_wrapper(env, mu_q: torch.Tensor, h_world: torch.Tensor, eps: float=1e-6, basexy_norm_type: str = "relative", ori_weight: float = 1.0) -> torch.Tensor:
    if env.name == 'Simple':
        return fk_mse_from_qfeat_simple(env, mu_q, h_world, eps, basexy_norm_type=basexy_norm_type)
    elif env.name == 'RotaryLink':
        return fk_mse_from_qfeat_rotary(env, mu_q, h_world, eps, basexy_norm_type=basexy_norm_type)
    elif env.name == 'RotaryNLink':
        return fk_mse_from_qfeat_rotaryN(env, mu_q, h_world, eps, ori_weight=ori_weight) # default to norm_type "canonical"
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
    theta_cs = normalize_to_unit_circle(torch.stack([cos_theta, sin_theta], dim=-1), eps)
    cos_theta, sin_theta = theta_cs[:, 0], theta_cs[:, 1]

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
    psi_cs = normalize_to_unit_circle(torch.stack([mu_q[:, 2], mu_q[:, 3]], dim=-1), eps)  # (B,2)
    theta1_cs = normalize_to_unit_circle(torch.stack([mu_q[:, 4], mu_q[:, 5]], dim=-1), eps)  # (B,2)
    theta2_cs = normalize_to_unit_circle(torch.stack([mu_q[:, 6], mu_q[:, 7]], dim=-1), eps)  # (B,2)

    c_psi, s_psi = psi_cs[:, 0], psi_cs[:, 1]
    c_theta1, s_theta1 = theta1_cs[:, 0], theta1_cs[:, 1]
    c_theta2, s_theta2 = theta2_cs[:, 0], theta2_cs[:, 1]

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

def fk_mse_from_qfeat_rotaryN(env, mu_q: torch.Tensor, h_world: torch.Tensor, eps: float=1e-8, ori_weight: float=1.0) -> torch.Tensor:
    """
    Computes FK MSE for a N-link robot using canonical featurization.
    Args:
        env: RotaryNLinkEnv
        mu_q: [B, 4 + 2*N] -> (x_tilde, y_tilde, cos_psi, sin_psi, cos_th1, sin_th1, ...)
        h_world: [B, 3] -> (hx, hy, phi)
    """ # TODO check the math in this function + math in normalize to unit circle function (torch dimensions across this file) in much more detail
    # Extract features
    pos_feat = mu_q[:, 0:2] # [B, 2]
    psi_feat = mu_q[:, 2:4] # [B, 2]
    joint_feats = mu_q[:, 4:] # [B, 2*N]

    h_pos = h_world[:, 0:2] # [B, 2]
    h_phi = h_world[:, 2:3] # [B, 1]

    # 1. Revert base to world frame
    delta_scaled = pos_feat * env.get_robot_scale()
    base_pos_world = rotate_vector_torch(delta_scaled, h_phi) + h_pos
    phi_vec = torch.cat([torch.cos(h_phi), torch.sin(h_phi)], dim=1) # [B, 2]
    psi_world_vec = rotate_vector_torch(psi_feat, phi_vec)

    # Normalize base heading for stability
    psi_world_vec = normalize_to_unit_circle(psi_world_vec, eps)

    # 2. Arm FK
    num_links = env.n_links
    joints = joint_feats.reshape(mu_q.shape[0], num_links, 2) # reshape joints to [B, N, 2]
    joints = normalize_to_unit_circle(joints, eps)

    # Initialize cumulative heading with base heading
    curr_heading_vec = psi_world_vec
    arm_accum_x, arm_accum_y = 0.0, 0.0

    for i in range(num_links):
        # 1. Update heading - rotate current heading by next joint angle theta_i
        curr_heading_vec = rotate_vector_torch(curr_heading_vec, joints[:, i, :])

        # 2. Add link vector: L_i * heading
        L_i = env.link_lengths[i]
        arm_accum_x = arm_accum_x + L_i * curr_heading_vec[:, 0]
        arm_accum_y = arm_accum_y + L_i * curr_heading_vec[:, 1]

    # 3. Final transform: end effector = base position + arm accumulation
    pred_hx = base_pos_world[:, 0] + arm_accum_x
    pred_hy = base_pos_world[:, 1] + arm_accum_y

    # Position MSE loss
    pos_mse = (pred_hx - h_world[:, 0]) ** 2 + (pred_hy - h_world[:, 1]) ** 2

    # Orientation MSE loss: Loss = ||v_pred - v_targ||^2 = 2 - 2*cos(error) = monotonic with angle error
    ori_mse = (curr_heading_vec - phi_vec).pow(2).sum(dim=1)

    # MSE loss
    return pos_mse + ori_mse * ori_weight
