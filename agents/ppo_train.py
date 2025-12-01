# ppo_train_full.py
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.stardew_mine_env import StardewMineEnv
from env.reward_functions import compute_reward  # <-- fix here



# -------------------------------
#   ENV CREATION
# -------------------------------
def make_env(seed=None):
    env = StardewMineEnv(size=10, seed=seed)
    return env


# -------------------------------
#   EVALUATION (RAW ENV)
# -------------------------------
def evaluate_model(model, episodes=10):
    metrics = {
        "total_ore": [],
        "energy_used": [],
        "exploration_rate": [],
        "max_floor": [],
    }

    for ep in range(episodes):
        env = make_env(seed=ep)
        obs, info = env.reset()

        done = False
        total_ore = 0
        starting_energy = float(obs["energy"][0])
        visited_tiles = set()
        max_floor_reached = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track ORE
            if info.get("last_mined_tile_type") == "ore":
                total_ore += 1

            # Track visited tiles
            x = info["agent_x"]
            y = info["agent_y"]
            visited_tiles.add((x, y))

            # Track floor progression
            max_floor_reached = max(max_floor_reached, info["floor"])

        metrics["total_ore"].append(total_ore)
        metrics["energy_used"].append(starting_energy - float(obs["energy"][0]))
        metrics["exploration_rate"].append(len(visited_tiles) / (env.SIZE ** 2))
        metrics["max_floor"].append(max_floor_reached)

    return metrics


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

    # -------------------
    # EVALUATE
    # -------------------
    print("\nRunning evaluation...")
    eval_metrics = evaluate_model(model, episodes=10)

    print("\n=== Evaluation Results ===")
    print(f"Average Ore Collected:      {np.mean(eval_metrics['total_ore']):.2f}")
    print(f"Average Energy Used:        {np.mean(eval_metrics['energy_used']):.2f}")
    print(f"Average Exploration Rate:   {np.mean(eval_metrics['exploration_rate']):.2f}")
    print(f"Average Max Floor Reached:  {np.mean(eval_metrics['max_floor']):.2f}")


if __name__ == "__main__":
    main()
