"""Configurable reward function for the mining environment.

Top-level constants are defined so training experiments can quickly
adjust incentives by editing these values programmatically.
"""

# --- Tunable constants (edit between runs) -------------------------------
# Primary rewards
ORE_REWARD = 50.0
ROCK_REWARD = 1.0
WEED_PENALTY = 0.5

# Mining action bonuses/penalties
MINE_SUCCESS_BONUS = 30.0
MINE_FAILURE_PENALTY = -0.1
MINE_ATTEMPT_BONUS = 1.0

# Exploration shaping
NEW_TILE_BONUS = 0.2
REVISIT_PENALTY = -0.2

# Energy shaping (multiplier applied to energy delta)
ENERGY_MULTIPLIER = 0.05

# Progress
DESCEND_BONUS = 100.0
DESCEND_ATTEMPT_BONUS = 1.0
DESCEND_FAILURE_PENALTY = -1.0
LADDER_REACH_BONUS = 20.0
LADDER_VISIBLE_BONUS = 2.0
DESCEND_VISIBLE_ATTEMPT_BONUS = 10.0  # extra incentive to press descend when ladder is visible
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
            # if ladder is visible, give a stronger nudge to try descend
            if ladder_visible:
                reward += DESCEND_VISIBLE_ATTEMPT_BONUS
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
