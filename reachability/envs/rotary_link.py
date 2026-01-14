from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from reachability.envs.simple import Workspace2D
from reachability.utils.utils import wrap_to_2pi

@dataclass(frozen=True)
class RotaryLinkEnv:
    """
    Simple robot setup:
        Q = (x, y, psi, theta1, theta2) = (base pos, link orientation)
        H = (hx, hy)
        FK: hand(Q) = [x, y] + R(psi)*[L1*cos(theta1) + L2*cos(theta1 + theta2), L1*sin(theta1) + L2*sin(theta1 + theta2)]
    """
    workspace: Workspace2D
    link_lengths: np.ndarray  # e.g. np.array([L1, L2]) for 2-link robot
    joint_limits: np.ndarray | None  # e.g. np.array([[th1_min, th1_max], [th2_min, th2_max]])
    n_links: int = 2 # only working out math for this case for now, extend later potentially
    base_pos_eps: float = 0.2
    base_heading_stddev: float = np.pi / 12
    name: str = "RotaryLink"

    @property
    def dH(self) -> int:
        return 2

    @property
    def dQ(self) -> int:
        return 5

    def sample_H(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample target H within defined workspace bounds"""
        hx = rng.uniform(self.workspace.hx_min, self.workspace.hx_max, size=(n,1))
        hy = rng.uniform(self.workspace.hy_min, self.workspace.hy_max, size=(n,1))
        return np.concatenate([hx, hy], axis=1).astype(np.float32)  #[n, 2]

    def sample_Q_given_H_uniform(self, H: np.ndarray, rng: np.random.Generator):
        """Sample Q from the ground-truth conditional given H (shape = n samples x 2), assuming H has a uniform prior across the workspace."""
        n = H.shape[0]
        L1, L2 = self.link_lengths[0], self.link_lengths[1]            
        hx = H[:, 0:1] # col 0
        hy = H[:, 1:2] # col 1

        # sample base position (x,y) via the "donut" of feasible spots
        r_min = np.fabs(L1 - L2)
        r_max = L1 + L2
        r_min_eps = r_min + self.base_pos_eps
        r_max_eps = r_max - self.base_pos_eps
        if r_min_eps < r_max_eps:
            feasible_base_radii = [r_min_eps, r_max_eps]
        else:
            feasible_base_radii = [r_min, r_max]
        r_sampled = rng.uniform(*feasible_base_radii, size=(n, 1)).astype(np.float32)
        # print("r_sampled[0] = ", r_sampled[0])
        # sample direction phi (where the base is around the donut)
        phi = rng.uniform(0, 2.0 * np.pi, size=(n, 1)).astype(np.float32)
        sinphi, cosphi = np.sin(phi), np.cos(phi)
        # compute base coordinates:
        # print("hx shape: ", hx.shape, " | r shape: ", r_sampled.shape, " | phi shape: ", phi.shape)
        x = hx - r_sampled * sinphi # elementwise multiplication
        y = hy - r_sampled * cosphi
        # print("x shape: ", x.shape, " | y shape: ", y.shape)
        
        # base orientation psi - general constraint = robot faces target
        dx = hx - x
        dy = hy - y
        psi_ideal = np.arctan2(dy, dx)
        # add noise to prevent the model from collapsing to a single deterministic mapping:
        psi = psi_ideal + rng.normal(0, self.base_heading_stddev, size=(n, 1)).astype(np.float32)
        sinpsi, cospsi = np.sin(psi), np.cos(psi)

        # analytical arm IK (solve for theta1, theta2)
        u = dx * cospsi + dy * sinpsi
        v = -dx * sinpsi + dy * cospsi
        h_local = np.hstack([u, v])
        # solve for theta2 (elbow angle)
        cos_th2 = (u**2 + v**2 - L1**2 - L2**2) / (2 * L1 * L2)
        # clip to avoid NaNs from floating point precision issues
        cos_th2 = np.clip(cos_th2, -1.0, 1.0)
        th2 = np.arccos(cos_th2)
        th2 *= rng.choice([-1, 1], size=(n, 1)) # flip elbow up vs. down randomly for half the samples
        # solve theta 1 (shoulder)
        th1 = np.arctan2(v, u) - np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
        
        # if joint limits != None:
        if self.joint_limits:
            th1_limits, th2_limits = self.joint_limits[0], self.joint_limits[1]

            mask_th1 = (th1 >= th1_limits[0]) & (th1 <= th1_limits[1])
            mask_th2 = (th2 >= th2_limits[0]) & (th2 <= th2_limits[1])

            valid_mask = (mask_th1 & mask_th2).flatten()
            # print("valid mask shape = ", valid_mask.shape)

            def filter_and_keep_dims(arr, mask):
                return arr[mask].reshape(-1, 1)

            x_filtered = filter_and_keep_dims(x, valid_mask)
            y_filtered = filter_and_keep_dims(y, valid_mask)
            psi_filtered = filter_and_keep_dims(psi, valid_mask)
            th1_filtered = filter_and_keep_dims(th1, valid_mask)
            th2_filtered = filter_and_keep_dims(th2, valid_mask)

            Q = np.concatenate([
                x_filtered, y_filtered, psi_filtered, th1_filtered, th2_filtered
            ], axis=1).astype(np.float32)
        else:
            Q = np.concatenate([x, y, psi, th1, th2], axis=1).astype(np.float32)
        return Q

    def fk_hand(self, Q: np.ndarray) -> np.ndarray:
        """Returns hand position f(Q) given Q (Q shape = n samples x 5)"""
        # print("Q = ", Q)
        single = (Q.ndim == 1)
        if single:
            Q = Q.reshape(1, -1)
            # print("reshaped Q = ", Q)
        
        # 1. extract states
        b_x = Q[:, 0:1]
        b_y = Q[:, 1:2]
        psi = Q[:, 2:3]
        th1 = Q[:, 3:4]
        th2 = Q[:, 4:5]

        L1, L2 = self.link_lengths[0], self.link_lengths[1]

        # 2. compute arm position in the robot's local frame:
        local_x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
        local_y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)

        # 3. rotate by base heading psi + add base position b
        hand_x = b_x + np.cos(psi) * local_x - np.sin(psi) * local_y
        hand_y = b_y + np.sin(psi) * local_x + np.cos(psi) * local_y

        hand = np.concatenate([hand_x, hand_y], axis=1).astype(np.float32)
        # print("hand = ", hand)
        return hand

    def plot(self, one_Q: np.ndarray, one_H: np.ndarray, save=False, save_path=""):
        """Visualize the mobile base + 2-link arm for a single configuration Q and target H"""
        # 1. extract Q components: [x, y, psi, th1, th2]
        bx, by, psi = float(one_Q[0]), float(one_Q[1]), float(one_Q[2])
        th1, th2 = float(one_Q[3]), float(one_Q[4])
        
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # 2. compute key joints for plotting
        el_lx = L1 * np.cos(th1)
        el_ly = L1 * np.sin(th1)
        
        # global elbow position
        ex = bx + np.cos(psi) * el_lx - np.sin(psi) * el_ly
        ey = by + np.sin(psi) * el_lx + np.cos(psi) * el_ly
        
        # global hand position (using your FK function)
        hx_fk, hy_fk = self.fk_hand(one_Q).flatten()

        fig, ax = plt.subplots(figsize=(8, 8))

        # workspace
        w = self.workspace
        rect_x = [w.hx_min, w.hx_max, w.hx_max, w.hx_min, w.hx_min]
        rect_y = [w.hy_min, w.hy_min, w.hy_max, w.hy_max, w.hy_min]
        ax.plot(rect_x, rect_y, color='black', label="Workspace")

        # robot base + heading
        ax.scatter([bx], [by], color='blue', s=100, label="Base")
        # orientation vector (psi)
        vec_len = (L1 + L2) * 0.2  # Length proportional to arm size
        ax.quiver(bx, by, np.cos(psi), np.sin(psi), color='blue', 
                  scale=1.5, scale_units='xy', width=0.005, label="Heading (ψ)")

        # 2-link arm
        # link 1: base to elbow
        ax.plot([bx, ex], [by, ey], color='red', linewidth=3, label="Link 1")
        # link 2: elbow to hand
        ax.plot([ex, hx_fk], [ey, hy_fk], color='orange', linewidth=3, label="Link 2")
        # elbow joint
        ax.scatter([ex], [ey], color='darkred', s=50, zorder=3)
        # hand (end effectors)
        ax.scatter([hx_fk], [hy_fk], marker="x", color='green', s=100, label="Hand (FK)")

        # target
        one_H = one_H.flatten()
        hx_target, hy_target = float(one_H[0]), float(one_H[1])
        ax.scatter([hx_target], [hy_target], marker="*", color='gold', s=200, label="Target", alpha=0.8)
        
        # error line (visualization)
        ax.plot([hx_fk, hx_target], [hy_fk, hy_target], linestyle=":", color='gray')

        # reachability donut around base
        t = np.linspace(0.0, 2.0 * np.pi, 200)
        r_outer = L1 + L2
        r_inner = np.abs(L1 - L2)
        ax.plot(bx + r_outer * np.cos(t), by + r_outer * np.sin(t), "r--", alpha=0.3)
        ax.plot(bx + r_inner * np.cos(t), by + r_inner * np.sin(t), "r--", alpha=0.3)

        # formatting
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (Global)")
        ax.set_ylabel("Y (Global)")
        ax.set_title(f"RotaryLinkEnv: ψ={np.degrees(psi):.1f}°, θ1={np.degrees(th1):.1f}°, θ2={np.degrees(th2):.1f}°")
        ax.legend(loc='upper right', fontsize='small')

        if save:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        
        return ax

    @staticmethod
    def target_bearing_world(Q: np.ndarray, H: np.ndarray) -> np.ndarray:
        """World-frame bearing angle from base position (x, y) to target H.
        -> Returns the 'implied' base heading psi that points directly 
        from the base (x, y) to the target (hx, hy).
        """
        hx, hy = H[:, 0], H[:, 1]
        bx, by = Q[:, 0], Q[:, 1]
        psi_implied = np.arctan2(hy - by, hx - bx)
        # print(psi_implied)
        psi_implied = wrap_to_2pi(psi_implied)
        return psi_implied.astype(np.float32)
