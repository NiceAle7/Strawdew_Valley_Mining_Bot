# ppo_train_full.py
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.stardew_mine_env_weeds import StardewMineEnv
from env.reward_functions import compute_reward  # <-- fix here
from stable_baselines3.common.monitor import Monitor

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)



# -------------------------------
#   ENV CREATION
# -------------------------------
def make_env(seed=None):
    env = StardewMineEnv(size=10, seed=seed)
    env = Monitor(env, LOG_DIR)  
    return env


# -------------------------------
#   MAIN TRAINING LOOP
# -------------------------------
def main():
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
        seed=0
    )

    total_timesteps = 400_000
    model.learn(total_timesteps=total_timesteps)

    model.save("ppo_stardew_mine_full.zip")
    print("✅ Model trained!")
    print("Saved as ppo_stardew_mine_full.zip")

if __name__ == "__main__":
    main()
