from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from reachability.utils.utils import wrap_to_2pi

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
    name: str = "Simple"

    @property
    def d_h_world(self) -> int:
        return 2

    @property
    def d_q_world(self) -> int:
        return 3

    def sample_h(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample target H within defined workspace bounds"""
        hx = rng.uniform(self.workspace.hx_min, self.workspace.hx_max, size=(n,1))
        hy = rng.uniform(self.workspace.hy_min, self.workspace.hy_max, size=(n,1))
        return np.concatenate([hx, hy], axis=1).astype(np.float32)  #[n, 2]

    def sample_q_given_h_uniform(self, h_world: np.ndarray, rng: np.random.Generator):
        """Sample Q from the ground-truth conditional given H (shape = n samples x 2):
        theta ~ Uniform[0, 2pi)
        x = hx - Lcos(theta)
        y = hy - Lsin(theta)"""
        n = h_world.shape[0]
        theta = rng.uniform(0, 2.0 * np.pi, size=(n, 1)).astype(np.float32)
        hx = h_world[:, 0:1] # col 0
        hy = h_world[:, 1:2] # col 1
        x = hx - self.L * np.cos(theta)
        y = hy - self.L * np.sin(theta)
        q_world = np.concatenate([x, y, theta], axis=1).astype(np.float32)
        return q_world

    def fk_hand(self, q_world: np.ndarray) -> np.ndarray:
        """Returns hand position f(Q) given Q (Q shape = n samples x 3)"""
        # print("Q = ", Q)
        single = (q_world.ndim == 1)
        if single:
            q_world = q_world.reshape(1, 3)
            # print("reshaped Q = ", Q)
        x = q_world[:, 0:1]
        y = q_world[:, 1:2]
        theta = q_world[:, 2:3]
        hand = np.concatenate(
            [x + self.L * np.cos(theta), y + self.L * np.sin(theta)],
            axis=1
        ).astype(np.float32)
        # print("hand = ", hand)
        return hand

    def plot(self, one_q_world: np.ndarray, one_h_world: np.ndarray, save=False, save_path=""):
        """Visualize the toy robot for a single configuration Q and target H"""
        x, y, theta = float(one_q_world[0]), float(one_q_world[1]), float(one_q_world[2])
        hx_fk, hy_fk = self.fk_hand(one_q_world).flatten()

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
        print(one_h_world, "**")
        hx, hy = float(one_h_world[0]), float(one_h_world[1])
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
    def target_bearing_world(q_world: np.ndarray, h_world: np.ndarray) -> np.ndarray:
        """World-frame bearing angle from base position (x, y) to target H.
        -> theta_implied = atan2(hy - y, hx - x)"""
        hx, hy, x, y = h_world[:, 0], h_world[:, 1], q_world[:, 0], q_world[:, 1]
        th = np.arctan2(hy - y, hx - x) # [-pi, pi]
        th = wrap_to_2pi(th)
        return th.astype(np.float32)
