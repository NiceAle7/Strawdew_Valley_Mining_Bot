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
def evaluate_model(model, env, episodes=5):

    metrics = {
        "total_ore": [],
        "energy_used": [],
        "exploration_rate": [],
        "max_floor": [],
    }

    is_vec = hasattr(env, "envs")

    for ep in range(episodes):
        obs, info = env.reset()

        total_ore = 0
        visited = set()
        max_floor_reached = 0

        starting_energy = float(obs["energy"][0]) if is_vec else float(obs["energy"][0])

        done = False
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, infos = env.step(action)

            done = bool(terminated[0]) if is_vec else bool(terminated)

            info0 = infos[0] if is_vec else infos

            # -----------------------
            # Count ore mined
            # -----------------------
            if info0.get("last_mined_tile_type") == "ore":
                total_ore += 1

            # -----------------------
            # Agent location tracking
            # -----------------------
            agent_loc = obs["agent_location"][0] if is_vec else obs["agent_location"]
            ax, ay = int(agent_loc[0]), int(agent_loc[1])
            visited.add((ax, ay))

            # -----------------------
            # Floor tracking
            # -----------------------
            floor_val = int(obs["floor"][0]) if is_vec else int(obs["floor"])
            max_floor_reached = max(max_floor_reached, floor_val)

            print(f"EP {ep+1} STEP {step_count}: "
                  f"Loc=({ax},{ay}) Floor={floor_val}")

            step_count += 1

        # -----------------------
        # Final episode stats
        # -----------------------
        remaining_energy = float(obs["energy"][0])

        grid_size = env.envs[0].SIZE if is_vec else env.SIZE

        metrics["total_ore"].append(total_ore)
        metrics["energy_used"].append(starting_energy - remaining_energy)
        metrics["exploration_rate"].append(len(visited) / (grid_size ** 2))
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

    metrics = evaluate_model(model, env, episodes=5)

    print("\n=== Evaluation Metrics ===")
    print(f"Avg Ore Collected:      {np.mean(metrics['total_ore']):.2f}")
    print(f"Avg Energy Used:        {np.mean(metrics['energy_used']):.2f}")
    print(f"Avg Exploration Rate:   {np.mean(metrics['exploration_rate']):.2f}")
    print(f"Avg Max Floor Reached:  {np.mean(metrics['max_floor']):.2f}")


if __name__ == "__main__":
    main()
