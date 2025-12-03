# reward_functions.py

# Generic ore reward value
ORE_REWARD_VALUES = {
    "copper": 5.0,
    "iron": 8.0,
    "gold": 12.0,
    "magma": 15.0,
    "mystic_stone": 25.0,
}   # adjust during experiments

def ore_reward(tile_type, action):
    """
    Reward for mining actual ore.
    """
    if action == "mine":
        if tile_type in ORE_REWARD_VALUES:
            return ORE_REWARD_VALUES[tile_type]
    return 0.0


def exploration_reward(has_visited_before):
    """
    Encourage exploring new tiles.
    This should be small to avoid overpowering ore rewards.
    """
    return 0.15 if not has_visited_before else -0.01


def movement_reward(action, has_visited_before):
    """
    Small reward for meaningful movement.
    Keep this tiny because exploration_reward already handles novelty.
    """
    if action == "move":
        return 0.03 if not has_visited_before else 0.0
    return 0.0

<<<<<<< HEAD

# def energy_shaping(previous_energy, current_energy):
#     """
#     Penalize wasting energy.
#     """
#     if current_energy < previous_energy:
#         return -0.005
#     return 0.0


=======
>>>>>>> f6cef72 (Working project.)
def useless_action_penalty(action, tile_type):
    """
    Penalize mining nothing or junk.
    """
    if action == "mine":
        if tile_type == "empty":
            return -0.6
        if tile_type == "weeds":
            return -0.2
        if tile_type == "ladder":
            return -1.0  # never mine the ladder!
        if tile_type == "rock":
            # Very small penalty becuase mining rock is often needed to find ladder
            return -0.02
    return 0.0


def floor_progression(current_floor, previous_floor):
    """
    Reward going deeper.
    """
    if current_floor > previous_floor:
        base = 4.0
        depth_bonus = 0.5 * current_floor  # deeper floor -> slightly more
        return base + depth_bonus
    return 0.0


def compute_reward(
    tile_type,
    action,
    has_visited_before,
    previous_energy,
    current_energy,
    previous_floor,
    current_floor,
    include_weeds=False
):
    """Combine all reward components."""
    reward = 0.0

    reward += ore_reward(tile_type, action)

    reward += exploration_reward(has_visited_before)
    reward += movement_reward(action, has_visited_before)
<<<<<<< HEAD
    # reward += energy_shaping(previous_energy, current_energy)
=======
>>>>>>> f6cef72 (Working project.)
    reward += useless_action_penalty(action, tile_type)
    reward += floor_progression(current_floor, previous_floor)

    # reward for clearing weeds
    if include_weeds and action == "mine" and tile_type == "weeds":
        reward += 0.1

    return reward


def update_ores_collected(tile_type, action, ores_collected):
    """
    Track ore count for evaluation.
    """
    if action == "mine":
        if tile_type in ORE_REWARD_VALUES:
            if isinstance(ores_collected, dict):
                ores_collected[tile_type] = ores_collected.get(tile_type, 0) + 1
            else:
                ores_collected += 1
<<<<<<< HEAD
    return ores_collected
=======
    return ores_collected
>>>>>>> f6cef72 (Working project.)
