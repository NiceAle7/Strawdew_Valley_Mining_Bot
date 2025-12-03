# ppo_train_no_weeds.py
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import NO-WEEDS environment
from env.stardew_mine_env_weeds import StardewMineEnv_Weeds


# -------------------------------
#   ENV CREATION
# -------------------------------
def make_env(seed=None):
    return StardewMineEnv_Weeds(size=10, seed=seed)


# -------------------------------
#   MAIN TRAINING LOOP
# -------------------------------
def main():

    vec_env = DummyVecEnv([lambda: make_env(seed=0)])

    print("Training PPO in NO-WEEDS environment...")

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        seed=0
    )

    total_timesteps = 400_000
    model.learn(total_timesteps=total_timesteps)

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "ppo_mine_no_weeds.zip")

    model.save(save_path)
    print("✅ PPO model trained in NO-WEEDS environment")
    print(f"Saved at: {save_path}")


if __name__ == "__main__":
    main()
