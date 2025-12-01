#!/usr/bin/env python3
"""
Visualize a trained agent's behavior and save as MP4.

Usage:
python visualize_agent.py \
  --model ppo_stardew_mine_full.zip \
  --episodes 3 \
  --output agent_demo.mp4
"""

import argparse
import os
import sys
import numpy as np

# Add root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Install OpenCV: pip install opencv-python")

from stable_baselines3 import PPO
from env.stardew_mine_env_weeds import StardewMineEnv


# -------------------------------------------------------
# TILE COLORS
# -------------------------------------------------------
def tile_to_color(tile_type):
    colors = {
        "empty":  (200, 200, 200),
        "rock":   (100, 100, 100),
        "ore":    (255, 200, 0),
        "weeds":  (100, 200, 100),
        "ladder": (150, 75, 0),
        "agent":  (0, 100, 255),
        "unknown": (50, 50, 50),
    }
    return colors.get(tile_type, colors["unknown"])


# -------------------------------------------------------
# FRAME RENDER
# -------------------------------------------------------
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

    # draw agent square
    ax0 = ax * tile_size + tile_size // 4
    ay0 = ay * tile_size + tile_size // 4
    ax1 = ax * tile_size + 3 * tile_size // 4
    ay1 = ay * tile_size + 3 * tile_size // 4

    cv2.rectangle(img, (ax0, ay0), (ax1, ay1), (255, 100, 0), -1)

    # overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, f"Floor: {env.floor}", (10, 25), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Energy: {env.energy:.1f}", (10, 50), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Steps: {env.current_step}/{env.max_steps}",
                (10, 75), font, 0.6, (255, 255, 255), 1)

    return img


# -------------------------------------------------------
# MAIN VIDEO FUNCTION
# -------------------------------------------------------
def visualize_episodes(model_path, num_episodes=3, output_path="agent_demo.mp4",
                       fps=10, tile_size=40, size=8, seed=42):

    if not HAS_CV2:
        return

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = PPO.load(model_path)

    frame_w = size * tile_size
    frame_h = size * tile_size

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h)
    )

    total_ores = 0

    for ep in range(num_episodes):

        print(f"Episode {ep+1}/{num_episodes}", end=" ")

        env = StardewMineEnv(size=size, seed=seed + ep)
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

            if info.get("last_mined_tile_type") == "ore":
                episode_ores += 1
                total_ores += 1

            steps += 1

        # hold final frame briefly
        for _ in range(5):
            writer.write(render_frame(env, tile_size))

        env.close()
        print(f"(ores={episode_ores}, steps={steps})")

    writer.release()
    print(f"\nVideo saved to {output_path}")
    print(f"Total ores mined: {total_ores}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", default="agent_demo.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--tile_size", type=int, default=40)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    visualize_episodes(
        args.model,
        args.episodes,
        args.output,
        args.fps,
        args.tile_size,
        args.size,
        args.seed
    )
