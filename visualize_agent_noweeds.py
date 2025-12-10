#!/usr/bin/env python3
"""
Visualize a trained agent's behavior (no weeds) and save as MP4.
"""

import argparse
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Install OpenCV: pip install opencv-python")

from stable_baselines3 import PPO
from env.stardew_mine_env import StardewMineEnv


# -------------------------------
# TILE COLORS
# -------------------------------
def tile_to_color(tile_type):
    colors = {
        "empty":        (200, 200, 200),
        "rock":         (100, 100, 100),
        "copper":       (240, 115, 40),
        "iron":         (142, 151, 151),
        "gold":         (255, 180, 15),
        "magma":        (235, 10, 10),
        "mystic_stone": (245, 0, 245),
        "weeds":        (100, 200, 100),
        "ladder":       (150, 75, 0),
        "agent":        (0, 100, 255),
        "unknown":      (50, 50, 50),
    }
    return colors.get(tile_type, colors["unknown"])


# -------------------------------
# FRAME RENDER
# -------------------------------
def render_frame(env, tile_size=40):
    grid = env.grid
    ax, ay = int(env.agent_location[0]), int(env.agent_location[1])

    h, w = env.SIZE, env.SIZE
    img = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            tile = int(grid[y, x])
            tile_type = env._map_tile_int_to_str(tile)
            r, g, b = tile_to_color(tile_type)
            color = (b, g, r)  # BGR for OpenCV

            x0, y0 = x * tile_size, y * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size

            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

    # draw agent
    ax0 = ax * tile_size + tile_size // 4
    ay0 = ay * tile_size + tile_size // 4
    ax1 = ax * tile_size + 3 * tile_size // 4
    ay1 = ay * tile_size + 3 * tile_size // 4
    cv2.rectangle(img, (ax0, ay0), (ax1, ay1), (255, 100, 0), -1)

    # overlay info
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Floor: {env.floor}", (10, 25), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Energy: {env.energy:.1f}", (10, 50), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Steps: {env.current_step}/{env.max_steps}", (10, 75), font, 0.6, (255, 255, 255), 1)

    return img


# -------------------------------
# VISUALIZE EPISODES
# -------------------------------
def visualize_episodes(model_path, num_episodes=3, output_path="agent_demo_no_weeds.mp4",
                       fps=10, tile_size=40, size=8, seed=42):

    if not HAS_CV2:
        return

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = PPO.load(model_path)

    frame_w = size * tile_size
    frame_h = size * tile_size
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    total_ores = 0

    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}", end=" ")

        # Create environment with no weeds
        env = StardewMineEnv(size=size, seed=seed + ep, spawn_weed=False)
        # Patch the floor generator to remove weeds
        env._generate_floor = lambda: _generate_floor_no_weeds(env)
        obs, info = env.reset()

        done = False
        episode_ores = 0
        steps = 0

        while not done:
            frame = render_frame(env, tile_size)
            writer.write(frame)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs, reward, done, info = step_res

            # Track all ore types
            ore_types = {"copper", "iron", "gold", "magma", "mystic_stone"}
            if info.get("last_mined_tile_type") in ore_types:
                episode_ores += 1
                total_ores += 1

            steps += 1

        for _ in range(5):
            writer.write(render_frame(env, tile_size))

        env.close()
        print(f"(ores={episode_ores}, steps={steps})")

    writer.release()
    print(f"\nVideo saved to {output_path}")
    print(f"Total ores mined: {total_ores}")


# -------------------------------
# FLOOR WITHOUT WEEDS
# -------------------------------
def _generate_floor_no_weeds(env):
    """Generate floor using multi-ore system without weeds."""
    env.grid = np.full((env.SIZE, env.SIZE), env.EMPTY, dtype=np.int32)
    probs = env._get_floor_spawn_probs(env.floor)
    p_rock   = probs["rock"]
    p_copper = probs["copper"]
    p_iron   = probs["iron"]
    p_gold   = probs["gold"]
    p_magma  = probs["magma"]
    p_mystic = probs["mystic"]
    
    possible = []
    ax, ay = int(env.agent_location[0]), int(env.agent_location[1])

    for y in range(env.SIZE):
        for x in range(env.SIZE):
            if (x, y) == (ax, ay):
                continue

            r = env.np_random.random()
            
            # Generate tiles using cumulative probabilities
            if r < p_rock:
                env.grid[y, x] = env.ROCK
                possible.append((x, y))
            elif r < p_rock + p_copper:
                env.grid[y, x] = env.COPPER
                possible.append((x, y))
            elif r < p_rock + p_copper + p_iron:
                env.grid[y, x] = env.IRON
                possible.append((x, y))
            elif r < p_rock + p_copper + p_iron + p_gold:
                env.grid[y, x] = env.GOLD
                possible.append((x, y))
            elif r < p_rock + p_copper + p_iron + p_gold + p_magma:
                env.grid[y, x] = env.MAGMA
                possible.append((x, y))
            elif r < p_rock + p_copper + p_iron + p_gold + p_magma + p_mystic:
                env.grid[y, x] = env.MYSTIC
                possible.append((x, y))

    if not possible:
        x = int(env.np_random.integers(0, env.SIZE))
        y = int(env.np_random.integers(0, env.SIZE))
        if (x, y) != (ax, ay):
            env.grid[y, x] = env.ROCK
            possible.append((x, y))

    if env.floor < env.MAX_FLOOR - 1 and possible:
        lx, ly = possible[int(env.np_random.integers(0, len(possible)))]
        env._ladder_location = (lx, ly)
        env.grid[ly, lx] = env.LADDER
    else:
        env._ladder_location = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", default="agent_demo_no_weeds.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tile_size", type=int, default=40)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    visualize_episodes(args.model, args.episodes, args.output, args.fps, args.tile_size, args.size, args.seed)
