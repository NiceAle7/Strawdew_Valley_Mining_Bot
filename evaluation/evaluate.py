# evaluation/evaluate.py
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.stardew_mine_env import StardewMineEnv


# ------------------------------
# ENV FACTORY
# ------------------------------
def make_env(seed=0):
    return StardewMineEnv(size=10, seed=seed)


# ------------------------------
# EVALUATION LOGIC
# ------------------------------
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


# ------------------------------
# MAIN
# ------------------------------
def main():

    env = DummyVecEnv([lambda: make_env(0)])

    model_path = os.path.join(
        os.path.dirname(__file__), "../ppo_stardew_mine_full.zip"
    )
    model = PPO.load(model_path, env=env)

    metrics = evaluate_model(model, episodes=5)

    print("\n=== Evaluation Metrics ===")
    print(f"Avg Ore Collected:      {np.mean(metrics['total_ore']):.2f}")
    print(f"Avg Energy Used:        {np.mean(metrics['energy_used']):.2f}")
    print(f"Avg Exploration Rate:   {np.mean(metrics['exploration_rate']):.2f}")
    print(f"Avg Max Floor Reached:  {np.mean(metrics['max_floor']):.2f}")


if __name__ == "__main__":
    main()
