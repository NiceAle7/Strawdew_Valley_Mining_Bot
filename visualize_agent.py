#!/usr/bin/env python3
"""
Visualize a trained agent's behavior and save as MP4.
Usage:
    python visualize_agent.py --model models/ppo_mine_validation_20k_20000_20251130_215404.zip --episodes 3 --output agent_demo.mp4
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not installed. Install with: pip install opencv-python")

from stable_baselines3 import PPO
try:
    from gymnasium.wrappers import FlattenObservation
except ImportError:
    from gym.wrappers import FlattenObservation

from env.stardew_mine_env import StardewMineEnv


def tile_to_color(tile_type):
    """Convert tile type to RGB color."""
    colors = {
        "empty": (200, 200, 200),      # light gray
        "rock": (100, 100, 100),       # dark gray
        "ore": (255, 200, 0),          # gold
        "weeds": (100, 200, 100),      # green
        "ladder": (150, 75, 0),        # brown
        "agent": (0, 100, 255),        # blue
        "unknown": (50, 50, 50),       # black
    }
    return colors.get(tile_type, colors["unknown"])


def render_frame(env, tile_size=40):
    """Render a frame of the environment as an image array (H, W, 3) in BGR format."""
    grid = env.grid.copy()
    agent_x, agent_y = int(env.agent_location[0]), int(env.agent_location[1])
    
    # Create image
    height, width = env.SIZE, env.SIZE
    img = np.zeros((height * tile_size, width * tile_size, 3), dtype=np.uint8)
    
    # Draw grid
    for y in range(height):
        for x in range(width):
            tile = int(grid[y, x])
            tile_type = env._map_tile_int_to_str(tile)
            color = tile_to_color(tile_type)
            # Convert RGB to BGR for OpenCV
            color = (color[2], color[1], color[0])
            x0, y0 = x * tile_size, y * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    
    # Draw agent
    agent_x0 = agent_x * tile_size + tile_size // 4
    agent_y0 = agent_y * tile_size + tile_size // 4
    agent_x1 = agent_x * tile_size + 3 * tile_size // 4
    agent_y1 = agent_y * tile_size + 3 * tile_size // 4
    cv2.rectangle(img, (agent_x0, agent_y0), (agent_x1, agent_y1), (255, 100, 0), -1)
    
    # Add text info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    text_y = 25
    cv2.putText(img, f"Floor: {env.floor}", (10, text_y), font, font_scale, (255, 255, 255), thickness)
    text_y += 25
    cv2.putText(img, f"Energy: {env.energy:.1f}", (10, text_y), font, font_scale, (255, 255, 255), thickness)
    text_y += 25
    cv2.putText(img, f"Steps: {env.current_step}/{env.max_steps}", (10, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return img


def visualize_episodes(model_path, num_episodes=3, output_path="agent_demo.mp4", fps=10, tile_size=40, size=8, seed=42):
    """Run agent episodes and save as MP4."""
    
    if not HAS_CV2:
        print("Error: opencv-python required for video output. Install with: pip install opencv-python")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    print(f"Recording {num_episodes} episodes to {output_path}...")
    
    # Setup video writer
    frame_width = size * tile_size
    frame_height = size * tile_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    total_ores = 0
    
    for ep in range(num_episodes):
        print(f"  Episode {ep + 1}/{num_episodes}...", end=" ", flush=True)
        
        # Create and reset environment (unwrap to access grid directly)
        env = FlattenObservation(StardewMineEnv(size=size, seed=seed + ep))
        obs, info = env.reset()
        
        done = False
        episode_ores = 0
        steps = 0
        
        while not done and steps < 1000:
            # Render frame
            frame = render_frame(env.unwrapped, tile_size=tile_size)
            out.write(frame)
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Step
            step_res = env.step(action)
            if isinstance(step_res, tuple) and len(step_res) == 5:
                obs, reward, terminated, truncated, step_info = step_res
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, step_info = step_res
            
            # Count ores
            if isinstance(step_info, dict) and step_info.get('last_mined_tile_type') == 'ore':
                episode_ores += 1
                total_ores += 1
            
            steps += 1
        
        # Render final frame a few times so it's visible
        for _ in range(5):
            frame = render_frame(env.unwrapped, tile_size=tile_size)
            out.write(frame)
        
        print(f"ores={episode_ores}, steps={steps}")
        env.close()
    
    out.release()
    print(f"\nVideo saved to {output_path}")
    print(f"Total ores mined: {total_ores}/{num_episodes} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize trained agent as MP4")
    parser.add_argument('--model', required=True, help="Path to trained model (zip file)")
    parser.add_argument('--episodes', type=int, default=3, help="Number of episodes to record")
    parser.add_argument('--output', default='agent_demo.mp4', help="Output MP4 filename")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second")
    parser.add_argument('--tile_size', type=int, default=40, help="Pixel size of each tile")
    parser.add_argument('--size', type=int, default=8, help="Grid size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for episodes")
    
    args = parser.parse_args()
    visualize_episodes(args.model, args.episodes, args.output, args.fps, args.tile_size, args.size, args.seed)
