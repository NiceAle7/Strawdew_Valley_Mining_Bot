# ppo_train_full.py
import sys
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    # Vectorized environment wrapper
    vec_env = DummyVecEnv([lambda: make_env(seed=0)])

    print("Training PPO...")

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        seed=0,
    )

    total_timesteps = 400_000
    model.learn(total_timesteps=total_timesteps)

    model.save("ppo_stardew_mine_full.zip")

    print("✅ Model trained!")
    print("Saved as ppo_stardew_mine_full.zip")


if __name__ == "__main__":
    main()
