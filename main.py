# DEPRECATED CODE, API has changed now
# import numpy as np
# from reachability.envs.simple import Workspace2D, SimpleEnv
# from reachability.models.cvae import CVAEConditionalSampler

# def main():
#     workspace = Workspace2D(-5, 5, -5, 5)
#     env = SimpleEnv(1, workspace)
    
#     model_path = "outputs/model_ckpts/cvae/cvae_132026.pt"
#     model = CVAEConditionalSampler.load(env, model_path, device="cpu")

#     H = np.array([[4.0, 5.0]], dtype=np.float32)

#     rng = np.random.default_rng(0)
#     Q_sampled = model.sample(H, n_samples=1, rng=rng).flatten()
#     print(Q_sampled)

#     env.plot(Q_sampled, H.flatten(), save=True, save_path="outputs/model_ckpts/cvae")

# if __name__ == "__main__":
#     main()
#     # x = 50
#     # print(np.arange(x))
