# reward_functions.py

# Generic ore reward value
ORE_REWARD_VALUE = 10.0   # adjust during experiments

def ore_reward(tile_type, action):
    """
    Reward for mining actual ore.
    Environment only has tile type 'ore', not named ores like 'copper'.
    """
    if action == "mine" and tile_type == "ore":
        return ORE_REWARD_VALUE
    return 0.0


def exploration_reward(has_visited_before):
    """
    Encourage exploring new tiles.
    This should be small to avoid overpowering ore rewards.
    """
    return 0.2 if not has_visited_before else -0.02


def movement_reward(action, has_visited_before):
    """
    Small reward for meaningful movement.
    Keep this tiny because exploration_reward already handles novelty.
    """
    if action == "move":
        return 0.05 if not has_visited_before else 0.0
    return 0.0


def energy_shaping(previous_energy, current_energy):
    """
    Penalize wasting energy.
    """
    if current_energy < previous_energy:
        return -0.01
    return 0.0


def useless_action_penalty(action, tile_type):
    """
    Penalize mining nothing or junk.
    """
    if action == "mine":
        if tile_type == "empty":
            return -0.5
        if tile_type == "weeds":
            return -0.2
        if tile_type == "ladder":
            return -1.0  # never mine the ladder!
    return 0.0


def floor_progression(current_floor, previous_floor):
    """
    Reward going deeper.
    """
    if current_floor > previous_floor:
        return 3.0   # flat reward per floor
    return 0.0


def compute_reward(
    tile_type,
    action,
    has_visited_before,
    previous_energy,
    current_energy,
    previous_floor,
    current_floor,
    include_weeds=True
):
    """Combine all reward components."""
    reward = 0.0

    reward += ore_reward(tile_type, action)
    reward += exploration_reward(has_visited_before)
    reward += movement_reward(action, has_visited_before)
    reward += energy_shaping(previous_energy, current_energy)
    reward += useless_action_penalty(action, tile_type)
    reward += floor_progression(current_floor, previous_floor)

    # Optional: reward for clearing weeds
    if include_weeds and action == "mine" and tile_type == "weeds":
        reward += 0.1

    return reward


def update_ores_collected(tile_type, action, ores_collected):
    """
    Track ore count for evaluation.
    """
    if action == "mine" and tile_type == "ore":
        ores_collected += 1
    return ores_collected
