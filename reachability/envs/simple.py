from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Workspace2D:
    hx_min: float
    hx_max: float
    hy_min: float
    hy_max: float

@dataclass(frozen=True)
class SimpleEnv:
    """
    Simple robot setup:
        Q = (x, y, theta) = (base pos, link orientation)
        H = (hx, hy)
        FK: hand(Q) = [x, y] + L*[cos theta, sin theta]
    """
    L: float
    workspace: Workspace2D

    @property
    def dH(self) -> int:
        return 2

    @property
    def dQ(self) -> int:
        return 3

    def sample_H(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample target H within defined workspace bounds"""
        hx = rng.uniform(self.workspace.hx_min, self.workspace.hx_max, size=(n,1))
        hy = rng.uniform(self.workspace.hy_min, self.workspace.hy_max, size=(n,1))
        return np.concatenate([hx, hy], axis=1).astype(np.float32)  #[n, 2]

    def sample_Q_given_H_uniform(self, H: np.ndarray, rng: np.random.Generator):
        """Sample Q from the ground-truth conditional given H (shape = n samples x 2):
        theta ~ Uniform[0, 2pi)
        x = hx - Lcos(theta)
        y = hy - Lsin(theta)"""
        n = H.shape[0]
        theta = rng.uniform(0, 2.0 * np.pi, size=(n, 1)).astype(np.float32)
        hx = H[:, 0:1] # col 0
        hy = H[:, 1:2] # col 1
        x = hx - self.L * np.cos(theta)
        y = hy - self.L * np.sin(theta)
        Q = np.concatenate([x, y, theta], axis=1).astype(np.float32)
        return Q

    def fk_hand(self, Q: np.ndarray) -> np.ndarray:
        """Returns hand position f(Q) given Q (Q shape = n samples x 3)"""
        print("Q = ", Q)
        single = (Q.ndim == 1)
        if single:
            Q = Q.reshape(1, 3)
            print("reshaped Q = ", Q)
        x = Q[:, 0:1]
        y = Q[:, 1:2]
        theta = Q[:, 2:3]
        hand = np.concatenate(
            [x + self.L * np.cos(theta), y + self.L * np.sin(theta)],
            axis=1
        ).astype(np.float32)
        print("hand = ", hand)
        return hand

    def plot(self, one_Q: np.ndarray, one_H: np.ndarray, save=False, save_path=""):
        """Visualize the toy robot for a single configuration Q and target H"""
        x, y, theta = float(one_Q[0]), float(one_Q[1]), float(one_Q[2])
        hx_fk, hy_fk = self.fk_hand(one_Q).flatten()

        fig, ax = plt.subplots()

        # workspace rectangle
        w = self.workspace
        rect_x = [w.hx_min, w.hx_max, w.hx_max, w.hx_min, w.hx_min]
        rect_y = [w.hy_min, w.hy_min, w.hy_max, w.hy_max, w.hy_min]
        ax.plot(rect_x, rect_y)

        # reach circle around base
        t = np.linspace(0.0, 2.0 * np.pi, 200)
        cx = x + self.L * np.cos(t)
        cy = y + self.L * np.sin(t)
        ax.plot(cx, cy, linestyle="--")

        # robot base, link, hand
        ax.scatter([x], [y], marker="o")  # base
        ax.plot([x, hx_fk], [y, hy_fk])   # link
        ax.scatter([hx_fk], [hy_fk], marker="x")  # hand (FK)

        # target H
        hx, hy = float(one_H[0]), float(one_H[1])
        ax.scatter([hx], [hy], marker="*")  # target
        ax.plot([hx_fk, hx], [hy_fk, hy], linestyle=":")  # line from hand to target (visual error)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        title = f"SimpleEnv: L={self.L:.3f}, theta={theta:.3f} rad"
        ax.set_title(title)
        if save:
            fig.savefig(f"{save_path}/simple_env_L{self.L:.3f}_theta{theta:.3f}.png", dpi=200, bbox_inches="tight")
        return ax

    @staticmethod
    def wrap_to_2pi(theta: np.ndarray) -> np.ndarray:
        """Wrap angles to [0, 2pi)"""
        return np.mod(theta, 2.0 * np.pi)

    @staticmethod
    def implied_theta_from_QH(Q: np.ndarray, H: np.ndarray) -> np.ndarray:
        """theta_implied = atan2(hy - y, hx - x)"""
        hx, hy, x, y = H[:, 0], H[:, 1], Q[:, 0], Q[:, 1]
        th = np.arctan2(hy - y, hx - x) # [-pi, pi]
        th = SimpleEnv.wrap_to_2pi(th)
        return th.astype(np.float32)
