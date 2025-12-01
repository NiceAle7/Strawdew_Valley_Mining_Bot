#!/usr/bin/env python3
import argparse
import os
import time
import csv
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

REWARD_FILE = os.path.join(ROOT, 'reward_functions.py')

REWARD_TEMPLATE = '''"""Configurable reward function for the mining environment.

Top-level constants are defined so training experiments can quickly
adjust incentives by editing these values programmatically.
"""

# --- Tunable constants (edit between runs) -------------------------------
# Primary rewards
ORE_REWARD = {ore_reward}
ROCK_REWARD = {rock_reward}
WEED_PENALTY = {weed_penalty}

# Mining action bonuses/penalties
MINE_SUCCESS_BONUS = {mine_success}
MINE_FAILURE_PENALTY = {mine_failure}
MINE_ATTEMPT_BONUS = {mine_attempt}

# Exploration shaping
NEW_TILE_BONUS = {new_tile}
REVISIT_PENALTY = {revisit}

# Energy shaping (multiplier applied to energy delta)
ENERGY_MULTIPLIER = {energy_mult}

# Progress
DESCEND_BONUS = {descend}
DESCEND_ATTEMPT_BONUS = {descend_attempt}
DESCEND_FAILURE_PENALTY = {descend_failure}
LADDER_REACH_BONUS = {ladder_reach}
LADDER_VISIBLE_BONUS = {ladder_visible}
# -------------------------------------------------------------------------


def compute_reward(tile_type: str, action: str, has_visited_before: bool,
                   previous_energy: int, current_energy: int,
                   previous_floor: int, current_floor: int,
                   ladder_visible: bool = False,
                   include_weeds: bool = True) -> float:
    """Reward shaping tuned toward encouraging mining behavior.

    Important: object rewards (ORE/ROCK) are only applied when the agent
    actually takes the `mine` action. This prevents repeatedly collecting
    the ore reward by standing on top of an ore tile.
    """

    reward = 0.0

    # Mining action incentives and object rewards only when mining
    try:
        if action == "mine":
            # small encouragement to try mining (bounded)
            reward += MINE_ATTEMPT_BONUS

            if tile_type == "ore":
                reward += ORE_REWARD + MINE_SUCCESS_BONUS
            elif tile_type == "rock":
                reward += ROCK_REWARD + MINE_FAILURE_PENALTY
            elif tile_type == "weeds":
                reward -= WEED_PENALTY if include_weeds else 0.0
            else:
                reward += MINE_FAILURE_PENALTY
    except Exception:
        pass

    # Exploration / revisit shaping
    if has_visited_before:
        reward += REVISIT_PENALTY
    else:
        reward += NEW_TILE_BONUS

    # Energy penalty
    try:
        reward += (current_energy - previous_energy) * ENERGY_MULTIPLIER
    except Exception:
        pass

    # Dense shaping: small reward when ladder is visible in local view
    try:
        if ladder_visible:
            reward += LADDER_VISIBLE_BONUS
    except Exception:
        pass

    # Descend / ladder incentives
    # If the agent chose the `descend` action, give a small attempt bonus.
    # If the descend actually succeeded (floor increased) give a larger bonus.
    # If it failed (no ladder at agent location), give a small penalty to discourage blind spamming.
    try:
        if action == "descend":
            reward += DESCEND_ATTEMPT_BONUS
            if current_floor > previous_floor:
                reward += DESCEND_BONUS
            else:
                reward += DESCEND_FAILURE_PENALTY
    except Exception:
        pass
    # Reward for moving onto the ladder tile (one-off per first visit)
    try:
        if action == "move" and tile_type == "ladder" and not has_visited_before:
            reward += LADDER_REACH_BONUS
    except Exception:
        pass

    return float(reward)
'''


def write_reward_file(params):
    with open(REWARD_FILE, 'w') as f:
        f.write(REWARD_TEMPLATE.format(**params))


def run_training(config_name, params, timesteps, move_cost, size, seed, eval_episodes):
    write_reward_file(params)
    # run training
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    try:
        from gymnasium.wrappers import FlattenObservation
    except Exception:
        from gym.wrappers import FlattenObservation

    from env.stardew_mine_env import StardewMineEnv

    def make_env():
        return FlattenObservation(StardewMineEnv(size=size, seed=seed, move_cost=move_cost, ladder_near_prob=args.ladder_near_prob))

    vec_env = DummyVecEnv([make_env])

    model = PPO('MlpPolicy', env=vec_env, verbose=1, seed=seed)
    print(f"Training {config_name} for {timesteps} timesteps (seed={seed})...")
    model.learn(total_timesteps=timesteps)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    models_dir = os.path.join(ROOT, 'models')
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f'ppo_{config_name}_{timesteps}_{timestamp}.zip')
    model.save(model_path)
    print('Model saved to', model_path)

    # evaluation
    eval_env = FlattenObservation(StardewMineEnv(size=size, seed=seed+1, move_cost=move_cost))
    eval_results = []
    for ep in range(eval_episodes):
        reset = eval_env.reset()
        if isinstance(reset, tuple) and len(reset) == 2:
            obs, info = reset
        else:
            obs = reset
            info = {}
        done = False
        total_reward = 0.0
        total_ore = 0
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            step_res = eval_env.step(action)
            if isinstance(step_res, tuple) and len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, info = step_res
            total_reward += float(reward)
            if isinstance(info, dict):
                if info.get('last_mined_tile_type') == 'ore' or info.get('last_mined_tile') == eval_env.unwrapped.ORE:
                    total_ore += 1
            steps += 1
        eval_results.append({'episode': ep, 'reward': total_reward, 'ores': total_ore, 'energy': eval_env.unwrapped.energy, 'steps': steps})
        print(f"Ep {ep}: reward={total_reward:.2f}, ores={total_ore}, energy={eval_env.unwrapped.energy}, steps={steps}")

    # save eval csv
    csv_path = os.path.join(results_dir, f'eval_{config_name}_{timesteps}_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['episode', 'reward', 'ores', 'energy', 'steps'])
        writer.writeheader()
        for r in eval_results:
            writer.writerow(r)
    print('Eval results saved to', csv_path)

    return model_path, csv_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--ore', type=float, default=50.0)
    parser.add_argument('--mine_success', type=float, default=25.0)
    parser.add_argument('--mine_failure', type=float, default=-0.2)
    parser.add_argument('--mine_attempt', type=float, default=0.5)
    parser.add_argument('--energy_mult', type=float, default=0.3)
    parser.add_argument('--descend', type=float, default=5.0)
    parser.add_argument('--descend_attempt', type=float, default=1.0)
    parser.add_argument('--descend_failure', type=float, default=-1.0)
    parser.add_argument('--ladder_reach', type=float, default=5.0)
    parser.add_argument('--ladder_visible', type=float, default=1.0)
    parser.add_argument('--ladder_near_prob', type=float, default=0.0,
                        help='Probability to spawn ladder near agent during floor generation')
    parser.add_argument('--new_tile', type=float, default=0.5)
    parser.add_argument('--revisit', type=float, default=-0.5)
    parser.add_argument('--rock', type=float, default=1.0)
    parser.add_argument('--weed', type=float, default=0.5)
    parser.add_argument('--move_cost', type=float, default=0.1)
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=10)

    args = parser.parse_args()

    params = {
        'ore_reward': args.ore,
        'rock_reward': args.rock,
        'weed_penalty': args.weed,
        'mine_success': args.mine_success,
        'mine_failure': args.mine_failure,
        'mine_attempt': args.mine_attempt,
        'new_tile': args.new_tile,
        'revisit': args.revisit,
        'energy_mult': args.energy_mult,
        'descend': args.descend,
        'descend_attempt': args.descend_attempt,
        'descend_failure': args.descend_failure,
        'ladder_reach': args.ladder_reach,
        'ladder_visible': args.ladder_visible,
        'ladder_near_prob': args.ladder_near_prob,
    }

    run_training(args.name, params, args.timesteps, args.move_cost, args.size, args.seed, args.eval_episodes)
