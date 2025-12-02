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
def make_env(seed=0, use_weeds=True) -> StardewMineEnv:
    return StardewMineEnv(size=10, seed=seed, spawn_weed=use_weeds)


# ------------------------------
# EVALUATION LOGIC
# ------------------------------
def evaluate_model(model, episodes=10, use_weeds=True):

    # Ore types in the environment
    ore_types = {"copper", "iron", "gold", "magma", "mystic_stone"}
    
    metrics = {
        "total_ore": [],
        "energy_used": [],
        "exploration_rate": [],
        "max_floor": [],
    }

    for ep in range(episodes):

        env = make_env(seed=ep, use_weeds=use_weeds)
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

            # Track mined ore (count all ore types)
            if info.get("last_mined_tile_type") in ore_types:
                total_ore += 1

            # Track visited tiles
            visited_tiles.add((info["agent_x"], info["agent_y"]))

            # Track floor progress
            max_floor_reached = max(max_floor_reached, info["floor"])

        metrics["total_ore"].append(total_ore)

        metrics["energy_used"].append(
            starting_energy - float(obs["energy"][0])
        )

        metrics["exploration_rate"].append(
            len(visited_tiles) / (env.SIZE ** 2)
        )

        metrics["max_floor"].append(max_floor_reached)

    return metrics


# ------------------------------
# MAIN
# ------------------------------
def main():

    env = DummyVecEnv([lambda: make_env(0)])

    # Allow override via env var or CLI arg
    override = os.environ.get("MODEL_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)

    candidates = []
    if override:
        candidates.append(os.path.abspath(override))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(repo_root, "models")

    # Look for common model filenames in models/ and repo root
    names = ["ppo_mine_no_weeds", "ppo_mine_no_weeds.zip", "ppo_mine_weeds", "ppo_mine_weeds.zip", "ppo_stardew_mine_full", "ppo_stardew_mine_full.zip"]
    for name in names:
        candidates.append(os.path.join(models_dir, name))
        candidates.append(os.path.join(repo_root, name))

    candidates = [os.path.abspath(p) for p in candidates]

    model_path = None
    for p in candidates:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(f"No model file found. Checked: {candidates}")

    print(f"Using model file: {model_path}")
    
    # Infer use_weeds from model filename BEFORE loading
    model_basename = os.path.basename(model_path).lower()
    if "no_weeds" in model_basename:
        use_weeds = False
    elif "weeds" in model_basename:
        use_weeds = True
    else:
        # Default to True for unknown models
        use_weeds = True
    
    # Recreate env with correct weed setting before loading model
    env = DummyVecEnv([lambda: make_env(0, use_weeds=use_weeds)])
    model = PPO.load(model_path, env=env)

    metrics = evaluate_model(model, episodes=5, use_weeds=use_weeds)

    print("\n=== Evaluation Metrics ===")
    print(f"Avg Ore Collected:      {np.mean(metrics['total_ore']):.2f}")
    print(f"Avg Energy Used:        {np.mean(metrics['energy_used']):.2f}")
    print(f"Avg Exploration Rate:   {np.mean(metrics['exploration_rate']):.3f}")
    print(f"Avg Max Floor Reached:  {np.mean(metrics['max_floor']):.2f}")


if __name__ == "__main__":
    main()
