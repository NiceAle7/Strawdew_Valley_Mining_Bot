from typing import Optional
import numpy as np
import gymnasium as gym

from reward_functions import compute_reward

__all__ = ["StardewMineEnv"]


class StardewMineEnv(gym.Env):
    """Grid-based mining environment with reward shaping."""

    def __init__(
        self,
        size: int = 10,
        max_floor: int = 10,
        max_energy: int = 500,
        local_view_size: int = 5,
        move_cost: float = 0.0,
        seed: Optional[int] = None,
        \
        spawn_weed: bool = False
    ):

        # Grid config
        self.SIZE = size
        self.MAX_FLOOR = max_floor
        self.MAX_ENERGY = max_energy
        self.LOCAL_VIEW_SIZE = local_view_size
        # configurable movement energy cost (default 0.0 for backward compatibility)
        self.MOVE_COST = float(move_cost)

        # State
        self.agent_location = np.array([0, 0], dtype=np.int32)
        self.grid = np.full((self.SIZE, self.SIZE), 0, dtype=np.int32)
        self.energy = self.MAX_ENERGY
        self.floor = 0
        self.visited = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self._ladder_location = None
        self.last_mined_tile = None

        # ⭐ FIXED: Maximum episode length
        self.max_steps = 500
        self.current_step = 0

        # Tile types
        self.EMPTY = 0
        self.LADDER = 1
        self.ROCK = 2
        self.COPPER = 3
        self.IRON = 4
        self.GOLD = 5
        self.MAGMA = 6
        self.MYSTIC = 7
        self.OUT_OF_BOUND = -1
        if spawn_weed:
            self.WEED = 8
            self.MAX_TILE_TYPE = 8
        else:
            self.MAX_TILE_TYPE = 7
        self.AGENT = 9

        # Actions
        self.action_space = gym.spaces.Discrete(17)

        # Movement
        self.ACTION_MOVE_RIGHT = 0
        self.ACTION_MOVE_UP_RIGHT = 1
        self.ACTION_MOVE_UP = 2
        self.ACTION_MOVE_UP_LEFT = 3
        self.ACTION_MOVE_LEFT = 4
        self.ACTION_MOVE_DOWN_LEFT = 5
        self.ACTION_MOVE_DOWN = 6
        self.ACTION_MOVE_DOWN_RIGHT = 7

        # Mining
        self.ACTION_MINE_RIGHT = 8
        self.ACTION_MINE_UP_RIGHT = 9
        self.ACTION_MINE_UP = 10
        self.ACTION_MINE_UP_LEFT = 11
        self.ACTION_MINE_LEFT = 12
        self.ACTION_MINE_DOWN_LEFT = 13
        self.ACTION_MINE_DOWN = 14
        self.ACTION_MINE_DOWN_RIGHT = 15

        # Descend
        self.ACTION_DESCEND = 16

        # Movement vectors
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([1, -1]),
            2: np.array([0, -1]),
            3: np.array([-1, -1]),
            4: np.array([-1, 0]),
            5: np.array([-1, 1]),
            6: np.array([0, 1]),
            7: np.array([1, 1]),
        }

        # Observation space (SB3 MultiInputPolicy compatible)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_location": gym.spaces.Box(
                    low=0.0, high=float(self.SIZE - 1), shape=(2,), dtype=np.float32
                ),
                "energy": gym.spaces.Box(
                    low=0.0, high=float(self.MAX_ENERGY), shape=(1,), dtype=np.float32
                ),
                "floor": gym.spaces.Box(
                    low=0.0, high=float(self.MAX_FLOOR - 1), shape=(1,), dtype=np.float32
                ),
                "local_view": gym.spaces.Box(
                    low=float(self.OUT_OF_BOUND),
                    high=float(self.MAX_TILE_TYPE),
                    shape=(self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE, 1),
                    dtype=np.float32,
                ),
            }
        )

        # Seed RNG if provided (gym's super().reset(seed=...) will also set rng)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    # ----------------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Let gymnasium handle seeding properly
        super().reset(seed=seed)

        # If gym didn't set RNG, ensure one exists
        if not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.current_step = 0  # ⭐ FIXED
        self.energy = self.MAX_ENERGY
        self.floor = 0

        self.agent_location = np.array(
            self.np_random.integers(0, self.SIZE, size=2),
            dtype=np.int32
        )

        self.visited = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.visited[self.agent_location[1], self.agent_location[0]] = True
        self.last_mined_tile = None

        self._generate_floor()

        return self._get_obs(), self._get_info()

    # ----------------------------------------------------------------------
    # STEP
    # ----------------------------------------------------------------------
    def step(self, action: int):

        # SB3 safety check
        if isinstance(action, np.ndarray):
            action = int(action.item())

        action = int(action)
        self.current_step += 1  # ⭐ FIXED

        prev_energy = self.energy
        prev_floor = self.floor
        self.last_mined_tile = None

        terminated = False
        truncated = False
        has_visited_before = False

        # ACTIONS -----------------------------------------
        if 0 <= action <= 7:  # movement
            dest = self._compute_destination_from_action(action)
            new_x = int(np.clip(dest[0], 0, self.SIZE - 1))
            new_y = int(np.clip(dest[1], 0, self.SIZE - 1))
            has_visited_before = self.visited[new_y, new_x]

            self._move_agent_to(new_x, new_y)
            self.visited[new_y, new_x] = True

            # apply movement energy cost if configured
            if self.MOVE_COST != 0.0:
                # allow fractional energy; clamp at zero
                self.energy -= self.MOVE_COST
                if self.energy < 0:
                    self.energy = 0

        elif 8 <= action <= 15:  # mining
            tx, ty = self._compute_target_from_mine_action(action)
            if 0 <= tx < self.SIZE and 0 <= ty < self.SIZE:
                has_visited_before = self.visited[ty, tx]
                self.visited[ty, tx] = True

            self._mine_tile(action)

        elif action == self.ACTION_DESCEND:  # descend
            if self._ladder_location is not None:
                lx, ly = self._ladder_location
                has_visited_before = self.visited[ly, lx]
            self._attempt_descend()

        # TERMINATION -------------------------------------
        if self.energy <= 0:
            terminated = True

        if self._is_grid_empty():
            terminated = True

        # truncated if too many steps
        if self.current_step >= self.max_steps:
            truncated = True

        # REWARD ------------------------------------------
        if 8 <= action <= 15:
            tile_type = self._map_tile_int_to_str(self.last_mined_tile)
        else:
            tile_type = self._get_tile_type()

        reward = compute_reward(
            tile_type=tile_type,
            action=self._action_name(action),
            has_visited_before=bool(has_visited_before),
            previous_energy=prev_energy,
            current_energy=self.energy,
            previous_floor=prev_floor,
            current_floor=self.floor,
            include_weeds=True,
        )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ----------------------------------------------------------------------
    # MOVEMENT / MINING HELPERS
    # ----------------------------------------------------------------------
    def _compute_destination_from_action(self, action_idx: int):
        direction = self._action_to_direction.get(action_idx, np.array([0, 0]))
        return self.agent_location + direction

    def _compute_target_from_mine_action(self, action: int):
        direction = self._action_to_direction.get(action - 8, np.array([0, 0]))
        return (
            int(self.agent_location[0] + direction[0]),
            int(self.agent_location[1] + direction[1]),
        )

    def _move_agent_to(self, x: int, y: int):
        self.agent_location = np.array([x, y], dtype=np.int32)

    def _mine_tile(self, action):
        self.energy -= 1
        tx, ty = self._compute_target_from_mine_action(action)

        if not (0 <= tx < self.SIZE and 0 <= ty < self.SIZE):
            self.last_mined_tile = None
            return

        tile = int(self.grid[ty, tx])
        self.last_mined_tile = tile

        if tile != self.LADDER:
            self.grid[ty, tx] = self.EMPTY

    # ----------------------------------------------------------------------
    # DESCEND FLOOR
    # ----------------------------------------------------------------------
    def _attempt_descend(self):
        if self.floor >= self.MAX_FLOOR - 1:
            return

        ax, ay = int(self.agent_location[0]), int(self.agent_location[1])
        if self._ladder_location != (ax, ay):
            return

        self.energy -= 1
        self.floor += 1
        self._generate_floor()

        self.visited = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.visited[self.agent_location[1], self.agent_location[0]] = True
        self.last_mined_tile = None

    # ----------------------------------------------------------------------
    # GRID GENERATION
    # ----------------------------------------------------------------------
    def _generate_floor(self):
        self.grid = np.full((self.SIZE, self.SIZE), self.EMPTY, dtype=np.int32)

        # Increase ore density for training experiments to make mining more
        # learnable. Lower rock probability slightly and reduce weeds.
        
        probs = self._get_floor_spawn_probs(self.floor)
        p_rock   = probs["rock"]
        p_copper = probs["copper"]
        p_iron   = probs["iron"]
        p_gold   = probs["gold"]
        p_magma  = probs["magma"]
        p_mystic = probs["mystic"]

        p_weed = 0.05 if hasattr(self, "WEED") else 0.0


        possible = []
        ax, ay = int(self.agent_location[0]), int(self.agent_location[1])

        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if (x, y) == (ax, ay):
                    continue

                r = self.np_random.random()

                # weeds first if enabled
                if hasattr(self, "WEED") and r < p_weed:
                    self.grid[y, x] = self.WEED

                # Ores & rocks (in cumulative order)
                elif r < p_weed + p_rock:
                    self.grid[y, x] = self.ROCK
                    possible.append((x, y))
                elif r < p_weed + p_rock + p_copper:
                    self.grid[y, x] = self.COPPER
                    possible.append((x, y))
                elif r < p_weed + p_rock + p_copper + p_iron:
                    self.grid[y, x] = self.IRON
                    possible.append((x, y))
                elif r < p_weed + p_rock + p_copper + p_iron + p_gold:
                    self.grid[y, x] = self.GOLD
                    possible.append((x, y))
                elif r < p_weed + p_rock + p_copper + p_iron + p_gold + p_magma:
                    self.grid[y, x] = self.MAGMA
                    possible.append((x, y))
                elif r < p_weed + p_rock + p_copper + p_iron + p_gold + p_magma + p_mystic:
                    self.grid[y, x] = self.MYSTIC_STONE
                    possible.append((x, y))
                # else EMPTY

        # Ensure at least 1 ore/rock tile for ladder placement
        if not possible:
            x = int(self.np_random.integers(0, self.SIZE))
            y = int(self.np_random.integers(0, self.SIZE))
            if (x, y) != (ax, ay):
                self.grid[y, x] = self.ROCK
                possible.append((x, y))

        # Place ladder
        if self.floor < self.MAX_FLOOR - 1 and possible:
            lx, ly = possible[int(self.np_random.integers(0, len(possible)))]
            self._ladder_location = (lx, ly)
            self.grid[ly, lx] = self.LADDER
        else:
            self._ladder_location = None

    
    def _get_floor_spawn_probs(self, floor: int) -> dict:
        if floor <= 1:
            return dict(rock=0.30, copper=0.08, iron=0.00, gold=0.00, magma=0.00, mystic=0.00)
        elif floor <= 3:
            return dict(rock=0.25, copper=0.12, iron=0.00, gold=0.00, magma=0.00, mystic=0.00)
        elif floor <= 5:
            return dict(rock=0.20, copper=0.10, iron=0.10, gold=0.00, magma=0.00, mystic=0.00)
        elif floor <= 7:
            return dict(rock=0.15, copper=0.08, iron=0.12, gold=0.05, magma=0.02, mystic=0.00)
        else:  # floors 8-9
            return dict(rock=0.10, copper=0.06, iron=0.12, gold=0.08, magma=0.04, mystic=0.02)


    # ----------------------------------------------------------------------
    # MAPPING + CHECKS
    # ----------------------------------------------------------------------
    def _is_grid_empty(self):
        mineables = (
            (self.grid == self.ROCK)
            | (self.grid == self.COPPER)
            | (self.grid == self.IRON)
            | (self.grid == self.GOLD)
            | (self.grid == self.MAGMA_GEODE)
            | (self.grid == self.MYSTIC_STONE)
        )
        if hasattr(self, "WEED"):
            mineables |= (self.grid == self.WEED)
        return not np.any(mineables)

    def _get_tile_type(self):
        """Return the string type of the tile under the agent."""
        x, y = int(self.agent_location[0]), int(self.agent_location[1])
        if 0 <= x < self.SIZE and 0 <= y < self.SIZE:
            tile = int(self.grid[y, x])
            return self._map_tile_int_to_str(tile)
        return "unknown"

    def _map_tile_int_to_str(self, tile_int):
        mapping = {
            self.EMPTY: "empty",
            self.ROCK: "rock",
            self.COPPER: "copper",
            self.IRON: "iron",
            self.GOLD: "gold",
            self.MAGMA: "magma",
            self.MYSTIC_STONE: "mystic_stone",
            self.LADDER: "ladder",
        }
        if hasattr(self, "WEED"):
            mapping[self.WEED] = "weeds"
        return mapping.get(tile_int, "unknown")

    def _action_name(self, action):
        if action == self.ACTION_DESCEND:
            return "descend"
        if 8 <= action <= 15:
            return "mine"
        return "move"

    # ----------------------------------------------------------------------
    # OBS / INFO / RENDER
    # ----------------------------------------------------------------------
    def _get_obs(self):
        half = self.LOCAL_VIEW_SIZE // 2
        ax, ay = int(self.agent_location[0]), int(self.agent_location[1])

        x0, y0 = ax - half, ay - half
        x1, y1 = ax + half + 1, ay + half + 1

        local_view = np.full(
            (self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE),
            self.OUT_OF_BOUND,
            dtype=np.float32,
        )

        gx0 = max(0, x0)
        gy0 = max(0, y0)
        gx1 = min(self.SIZE, x1)
        gy1 = min(self.SIZE, y1)

        patch = self.grid[gy0:gy1, gx0:gx1].astype(np.float32)
        px0, py0 = gx0 - x0, gy0 - y0

        local_view[py0 : py0 + patch.shape[0], px0 : px0 + patch.shape[1]] = patch
        local_view = np.expand_dims(local_view, axis=-1)

        return {
            "agent_location": self.agent_location.astype(np.float32),
            "energy": np.array([float(self.energy)], dtype=np.float32),
            "floor": np.array([float(self.floor)], dtype=np.float32),
            "local_view": local_view,
        }

    def _get_info(self):
        # Return both integer tile and string tile type for convenience
        last_tile_type = None
        if self.last_mined_tile is not None:
            last_tile_type = self._map_tile_int_to_str(self.last_mined_tile)

        return {
            "ladder_location": self._ladder_location,
            "last_mined_tile": (
                None if self.last_mined_tile is None else int(self.last_mined_tile)
            ),
            "last_mined_tile_type": last_tile_type,
            "agent_x": int(self.agent_location[0]),
            "agent_y": int(self.agent_location[1]),
            "energy": float(self.energy),
            "floor": int(self.floor),
        }

    def render(self, render_mode: str = "human"):
        grid_copy = self.grid.copy()
        x, y = int(self.agent_location[0]), int(self.agent_location[1])
        grid_copy[y, x] = self.AGENT

        print("Floor:", self.floor, "Energy:", self.energy)
        print(grid_copy)
