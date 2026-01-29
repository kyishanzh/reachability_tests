from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from reachability.envs.workspace import Workspace2D
from reachability.utils.utils import wrap_to_2pi, sample_from_union

@dataclass(frozen=True)
class RotaryNLinkEnv:
    """
    N-link planar robot mobile manipulator.
    State space Q: (x, y, psi, theta_1, ..., theta_n)
        - x, y : Base position
        - psi: Base heading
        - theta_i: Joint angles
    Target H: (hx, hy, phi)
        - hx, hy: Target position
        - phi: Target heading
    """
    workspace: Workspace2D
    link_lengths: np.ndarray  # e.g. np.array([L1, L2]) for 2-link robot
    joint_limits: list | None  # e.g. np.array([[th1_min, th1_max], [th2_min, th2_max]])
    n_links: int = 2 # only working out math for this case for now, extend later potentially
    name: str = "RotaryNLink"

    @property
    def d_h(self) -> int:
        return 3 # (hx, hy, phi)

    @property
    def d_q(self) -> int:
        return 3 + self.n_links # (x, y, psi) + (theta_1...theta_n)

    def get_robot_scale(self):
        return np.sum(self.link_lengths)

    def sample_h(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample target H = (hx, hy, phi) uniformly workspace bounds"""
        hx = rng.uniform(self.workspace.hx_min, self.workspace.hx_max, size=(n,1))
        hy = rng.uniform(self.workspace.hy_min, self.workspace.hy_max, size=(n,1))
        phi = rng.uniform(0, 2 * np.pi, size=(n, 1))
        return np.concatenate([hx, hy, phi], axis=1).astype(np.float32)  #[n, 3]

    def sample_q(self, h_world: np.ndarray, rng: np.random.Generator, max_retries=1000):
        """Samples Q given H, ensuring:
        1. Kinematic validity (via FK approach)
        2. No self-collisions (non-adjacent links)
        3. TODO: No environment collisions (e.g. obstacle avoidance)
        """
        n_total = h_world.shape[0]
        q_final = np.zeros((n_total, self.d_q), dtype=np.float32)
        solved_mask = np.zeros(n_total, dtype=bool) # Track which indices are done (found solution Q that reaches H)

        iteration = 0
        while not np.all(solved_mask):
            iteration += 1
            if iteration > max_retries:
                print(f"[WARNING] Could not find valid config for {np.sum(~solved_mask)} samples after {max_retries} retries.")
                break

            # 1. Identify which target H are unsolved (yet to find valid Q that reaches H)
            unsolved_mask = ~solved_mask # (n_total,)
            n_unsolved = np.sum(unsolved_mask)
            h_subset = h_world[unsolved_mask] # (n_unsolved,)

            # 2. Raw sample (FK-based sampler)
            q_candidate = self._sample_q_raw(h_subset, rng) # (n_unsolved, d_q)

            # 3. Validation: check for self-collision & env-collision
            no_self_collision = ~self.check_self_collision(q_candidate) # (n_unsolved,)
            # TODO: Check environment collision - no_env_collision = ~self.check_env_collision(q_candidate), combined_valid = no_self_coll & no_env_collision
            combined_valid = no_self_collision # (n_unsolved,)

            # 4. Fill valid samples into the final array
            current_indices = np.where(unsolved_mask)[0] # (n_unsolved,) containing global indices of unsolved rows in the original 0...N-1
            successful_indices = current_indices[combined_valid]

            q_final[successful_indices] = q_candidate[combined_valid] # store successful results from this round
            solved_mask[successful_indices] = True
    
        return q_final
        
    def check_self_collision(self, q_world: np.ndarray) -> np.ndarray:
        """
        Check for self-collision (intersection of non-adjacent links). Returns boolean mask (N,) where True = collision.
        """
        n_samples = q_world.shape[0]
        bx, by, psi = q_world[:, 0:1], q_world[:, 1:2], q_world[:, 2:3]
        thetas = q_world[:, 3:]

        # FK for all joints
        joints = np.zeros((n_samples, self.n_links + 1, 2), dtype=np.float32) # (N, n_links + 1, 2)
        joints[:, 0, 0] = bx[:, 0]
        joints[:, 0, 1] = by[:, 0]
        curr_angle = psi.copy()
        for i in range(self.n_links):
            curr_angle += thetas[:, i:i+1]
            L = self.link_lengths[i]
            # print("joints shape = ", joints.shape, " | L = ", L, " | curr_angle shape = ", curr_angle.shape, " | curr_angle = ", curr_angle)
            joints[:, i+1, 0] = joints[:, i, 0] + L * np.cos(curr_angle[:, 0])
            joints[:, i+1, 1] = joints[:, i, 1] + L * np.cos(curr_angle[:, 0])

        # Check every link (i, j) where j > i + 1 for collisions
        collided = np.zeros(n_samples, dtype=bool) # record collisions
        for i in range(self.n_links - 2): # no need to check last 2 links against forward
            # Link A: joints[:, i] to joints[:, i + 1]
            p1 = joints[:, i]
            p2 = joints[:, i + 1]
            for j in range(i + 2, self.n_links):
                # Link B: joints[:, j] to joints[:, j + 1]
                p3 = joints[:, j]
                p4 = joints[:, j + 1]

                # Cross product check orientation of A -> B -> C
                def ccw(A, B, C):
                    return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
                
                # Segments intersect if endpoints of one are on opposite sides of the other
                intersect = (ccw(p1, p3, p4) != ccw(p2, p3, p4)) & (ccw(p1, p2, p3) != ccw(p1, p2, p4))
                collided |= intersect # bitwise OR update intersections -> collisions

        return collided

    def _sample_q_raw(self, h_world: np.ndarray, rng: np.random.Generator): # sample_q given q uniform
        """Generates Q = (x, y, psi, thetas) pairs that satisfy the kinematic constraint for H.
        Pipeline:
        1. Sample random joint angles theta ~ Uniform(limits)
        2. Compute arm FK (p_BE, phi_BE)
        3. Solve for base (x, y, psi) s.t. T_WB * T_BE = T_WE = T_WH (end effector pose = H)
        """ # TODO walk through math in this function a lot more carefully
        n = h_world.shape[0]
        
        # 1. Sample arm angles
        thetas_list = []
        if self.joint_limits is not None:
            for i in range(self.n_links):
                th = sample_from_union(self.joint_limits[i], rng, size=(n, 1))
                thetas_list.append(th)
        else:
            # Default to -pi to pi if no limits
            for i in range(self.n_links):
                th = rng.uniform(-np.pi, np.pi, size=(n, 1))
                thetas_list.append(th)
        thetas = np.concatenate(thetas_list, axis=1) # (n, n_links)

        # 2. Compute FK in base frame (p_BE, phi_BE)
        alphas = np.cumsum(thetas, axis=1) # (n, n_links) -- cumulative angles, alpha_i = sum(theta_1...theta_i)

        # p_BE_x = sum(L_i * cos(alpha_i)), p_BE_y = sum(L_i * sin(alpha_i))
        # link lengths broadcast: (1, n_links) * (n, n_links)
        L = self.link_lengths.reshape(1, -1) # (n_links,) -> (1, n_links) [reshape(1, -1) = 1 row and reshape rest automatically]
        p_be_x = np.sum(L * np.cos(alphas), axis=1, keepdims=True)
        p_be_y = np.sum(L * np.sin(alphas), axis=1, keepdims=True)

        # phi_BE is the cumulative angle of the last link
        phi_be = alphas[:, -1:] # (n, 1)

        # 3. Solve for base pose    
        hx = h_world[:, 0:1] # col 0
        hy = h_world[:, 1:2] # col 1
        h_phi = h_world[:, 2:3]

        # base heading: psi = phi - phi_BE
        psi = wrap_to_2pi(h_phi - phi_be)

        # base position: p_B = p_H - R(psi) * p_BE
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        rot_p_be_x = p_be_x * cos_psi - p_be_y * sin_psi
        rot_p_be_y = p_be_x * sin_psi + p_be_y * cos_psi
        base_x = hx - rot_p_be_x
        base_y = hy - rot_p_be_y

        # Assemble Q: (x, y, psi, thetas)
        q_world = np.concatenate([base_x, base_y, psi, thetas], axis=1).astype(np.float32)
        return q_world

    def fk_hand(self, q_world: np.ndarray) -> np.ndarray:
        """Returns end-effector pose (hand_x, hand_y, hand_phi) given Q."""
        # print("Q = ", Q)
        single = (q_world.ndim == 1)
        if single:
            q_world = q_world.reshape(1, -1)
            # print("reshaped Q = ", Q)
        
        # 1. extract states
        b_x = q_world[:, 0:1]
        b_y = q_world[:, 1:2]
        psi = q_world[:, 2:3]
        thetas = q_world[:, 3:]

        # 2. Cumulative angles in local frame
        alphas = np.cumsum(thetas, axis=1)

        # 3. Local EE position
        L = self.link_lengths.reshape(1, -1) # (1, n_links)
        local_x = np.sum(L * np.cos(alphas), axis=1, keepdims=True)
        local_y = np.sum(L * np.sin(alphas), axis=1, keepdims=True)
        local_phi = alphas[:, -1:]

        # 4. Transform to world: Rotate local position by base heading psi + add base position b
        # 3. rotate by base heading psi + add base position b
        hand_x = b_x + np.cos(psi) * local_x - np.sin(psi) * local_y
        hand_y = b_y + np.sin(psi) * local_x + np.cos(psi) * local_y
        hand_phi = wrap_to_2pi(psi + local_phi)

        hand = np.concatenate([hand_x, hand_y, hand_phi], axis=1).astype(np.float32)
        # print("hand = ", hand)
        return hand

    def plot(self, one_q_world: np.ndarray, one_h_world: np.ndarray, save=False, save_path=""):
        """Visualize N-link robot and target H = (x, y, phi)"""
        # Unpack Q
        bx, by, psi = one_q_world[0], one_q_world[1], one_q_world[2]
        thetas = one_q_world[3:]

        # Unpack H
        hx, hy, h_phi = one_h_world[0], one_h_world[1], one_h_world[2]
        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Workspace
        w = self.workspace
        rect_x = [w.hx_min, w.hx_max, w.hx_max, w.hx_min, w.hx_min]
        rect_y = [w.hy_min, w.hy_min, w.hy_max, w.hy_max, w.hy_min]
        ax.plot(rect_x, rect_y, color='black', alpha=0.5, label="Workspace")

        # 2. Compute Joint Positions for Plotting
        # Base is first point
        joint_x = [bx]
        joint_y = [by]
        
        # Cumulative angle tracker
        current_angle = psi
        current_x = bx
        current_y = by
        
        for i, th in enumerate(thetas):
            current_angle += th
            L = self.link_lengths[i]
            
            next_x = current_x + L * np.cos(current_angle)
            next_y = current_y + L * np.sin(current_angle)
            
            joint_x.append(next_x)
            joint_y.append(next_y)
            
            current_x, current_y = next_x, next_y

        # 3. Draw Robot
        # Base
        ax.scatter([bx], [by], color='blue', s=100, label="Base", zorder=4)
        # Base heading arrow
        ax.quiver(bx, by, np.cos(psi), np.sin(psi), color='blue', 
                  scale=3, scale_units='xy', width=0.005, alpha=0.5)

        # Links
        ax.plot(joint_x, joint_y, color='red', linewidth=3, marker='o', 
                markerfacecolor='darkred', markersize=6, zorder=3, label="Arm")
        
        # End Effector (last joint)
        ee_x, ee_y = joint_x[-1], joint_y[-1]
        ax.scatter([ee_x], [ee_y], marker="x", color='green', s=120, linewidth=3, label="EE (FK)", zorder=5)
        # EE Orientation arrow
        ax.quiver(ee_x, ee_y, np.cos(current_angle), np.sin(current_angle), color='green',
                 scale=3, scale_units='xy', width=0.004, alpha=0.8)

        # 4. Draw Target H
        ax.scatter([hx], [hy], marker="*", color='gold', s=250, label="Target", edgecolors='k', zorder=2)
        # Target Heading arrow
        ax.quiver(hx, hy, np.cos(h_phi), np.sin(h_phi), color='orange', 
                  scale=3, scale_units='xy', width=0.008, label="Target Heading")

        # Formatting
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (World)")
        ax.set_ylabel("Y (World)")
        title_str = f"{self.name} (N={self.n_links})\nBase psi={np.degrees(psi):.1f}° | EE Delta={np.linalg.norm([ee_x-hx, ee_y-hy]):.4f}"
        ax.set_title(title_str)
        ax.legend(loc='upper right', fontsize='small')
        
        # Determine plot limits to ensure robot fits
        all_x = np.concatenate([joint_x, [hx]])
        all_y = np.concatenate([joint_y, [hy]])
        pad = 1.0
        ax.set_xlim(np.min(all_x) - pad, np.max(all_x) + pad)
        ax.set_ylim(np.min(all_y) - pad, np.max(all_y) + pad)

        if save:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        
        return ax

    @staticmethod # TODO figure out if this is still useful / how to adapt this metric to new envs
    def target_bearing_world(q_world: np.ndarray, h_world: np.ndarray) -> np.ndarray:
        """World-frame bearing angle from base position (x, y) to target H.
        -> Returns the 'implied' base heading psi that points directly 
        from the base (x, y) to the target (hx, hy).
        """
        hx, hy = h_world[:, 0], h_world[:, 1]
        bx, by = q_world[:, 0], q_world[:, 1]
        psi_implied = np.arctan2(hy - by, hx - bx)
        # print(psi_implied)
        psi_implied = wrap_to_2pi(psi_implied)
        return psi_implied.astype(np.float32) 
