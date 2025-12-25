import numpy as np
from reachability.envs.simple import Workspace2D, SimpleEnv

def main():
    workspace = Workspace2D(-5, 5, -5, 5)
    env = SimpleEnv(1, workspace)
    env.plot(np.array([2, 3, 0.8 * np.pi]), np.array([4, 5]), save=True, save_path="outputs")

if __name__ == "__main__":
    # main()
    x = 50
    print(np.arange(x))
