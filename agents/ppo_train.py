# ppo_train.py
"""
Usage:
    python ppo_train.py                 # Train without weeds
    python ppo_train.py --weeds         # Train with weeds
"""

import sys
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.stardew_mine_env import StardewMineEnv


# -------------------------------
#   ENV CREATION
# -------------------------------
def make_env(seed=None, spawn_weed=False):
    """Create environment with optional weeds."""
    return StardewMineEnv(size=10, seed=seed, spawn_weed=spawn_weed)


# -------------------------------
#   MAIN TRAINING LOOP
# -------------------------------
def main(spawn_weed=False):
    """Train PPO model."""
    
    # Vectorized environment wrapper
    vec_env = DummyVecEnv([lambda: make_env(seed=0, spawn_weed=spawn_weed)])

    mode_str = "with weeds" if spawn_weed else "without weeds"
    print(f"Training PPO ({mode_str})...")

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

    # Determine save path and filename
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    if spawn_weed:
        filename = "ppo_mine_weeds.zip"
    else:
        filename = "ppo_mine_no_weeds.zip"

    save_path = os.path.join(models_dir, filename)
    model.save(save_path)

    print("✅ Model trained!")
    print(f"Saved as: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Stardew mining environment")
    parser.add_argument(
        "--weeds",
        action="store_true",
        help="Train with weeds enabled (default: no weeds)"
    )

    args = parser.parse_args()
    main(spawn_weed=args.weeds)
