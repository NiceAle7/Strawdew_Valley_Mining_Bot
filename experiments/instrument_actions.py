#!/usr/bin/env python3
import sys,os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

from stable_baselines3 import PPO
try:
    from gymnasium.wrappers import FlattenObservation
except Exception:
    from gym.wrappers import FlattenObservation
from env.stardew_mine_env import StardewMineEnv

MODEL = os.path.join(ROOT, 'models', 'ppo_moderate_ore_dense_100000_20251130_214525.zip')

model = PPO.load(MODEL)

env = FlattenObservation(StardewMineEnv(size=8, seed=42, move_cost=0.05))

action_counts = {i:0 for i in range(env.action_space.n)}

episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    steps = 0
    ores = 0
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        res = env.step(action)
        if isinstance(res, tuple) and len(res)==5:
            obs, reward, terminated, truncated, info = res
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = res
        if isinstance(info, dict):
            if info.get('last_mined_tile_type') == 'ore' or info.get('last_mined_tile') == env.unwrapped.ORE:
                ores += 1
        steps += 1
    print(f'Ep{ep}: ores={ores}, steps={steps}, energy={env.unwrapped.energy}')

print('Action counts:', action_counts)
