from typing import Optional
import numpy as np
import gymnasium as gym

from reward_functions import compute_reward


class StardewMineEnv_Weeds(gym.Env):
    """Grid-based mining environment WITH weeds."""

    def __init__(
        self,
        size: int = 10,
        max_floor: int = 10,
        max_energy: int = 100,
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
        self.MOVE_COST = float(move_cost)

        # State
        self.agent_location = np.array([0, 0], dtype=np.int32)
        self.grid = np.full((self.SIZE, self.SIZE), 0, dtype=np.int32)
        self.energy = self.MAX_ENERGY
        self.floor = 0
        self.visited = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self._ladder_location = None
        self.last_mined_tile = None

        self.max_steps = 500
        self.current_step = 0

        # Tile types (with weeds)
        self.EMPTY = 0
        self.LADDER = 1
        self.ROCK = 2
        self.ORE = 3
        self.OUT_OF_BOUND = -1
        # Optional weeds: when spawn_weed is True, enable WEED tile type
        if spawn_weed:
            self.WEED = 4
            self.MAX_TILE_TYPE = 4
        else:
            self.MAX_TILE_TYPE = 3
        self.AGENT = 9

        self.action_space = gym.spaces.Discrete(17)

        # Direction vectors
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

        # Movement and mining constants
        self.ACTION_MOVE_RIGHT = 0
        self.ACTION_MOVE_UP_RIGHT = 1
        self.ACTION_MOVE_UP = 2
        self.ACTION_MOVE_UP_LEFT = 3
        self.ACTION_MOVE_LEFT = 4
        self.ACTION_MOVE_DOWN_LEFT = 5
        self.ACTION_MOVE_DOWN = 6
        self.ACTION_MOVE_DOWN_RIGHT = 7

        self.ACTION_MINE_RIGHT = 8
        self.ACTION_MINE_UP_RIGHT = 9
        self.ACTION_MINE_UP = 10
        self.ACTION_MINE_UP_LEFT = 11
        self.ACTION_MINE_LEFT = 12
        self.ACTION_MINE_DOWN_LEFT = 13
        self.ACTION_MINE_DOWN = 14
        self.ACTION_MINE_DOWN_RIGHT = 15

        self.ACTION_DESCEND = 16

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                "agent_location": gym.spaces.Box(0, self.SIZE - 1, (2,), np.float32),
                "energy": gym.spaces.Box(0, self.MAX_ENERGY, (1,), np.float32),
                "floor": gym.spaces.Box(0, self.MAX_FLOOR - 1, (1,), np.float32),
                "local_view": gym.spaces.Box(
                    self.OUT_OF_BOUND,
                    self.MAX_TILE_TYPE,
                    (self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE, 1),
                    np.float32,
                ),
            }
        )

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    # ---------------- RESET / STEP / HELPERS ------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()
        self.current_step = 0
        self.energy = self.MAX_ENERGY
        self.floor = 0
        self.agent_location = self.np_random.integers(0, self.SIZE, size=2).astype(int)
        self.visited = np.zeros((self.SIZE, self.SIZE), bool)
        self.visited[self.agent_location[1], self.agent_location[0]] = True
        self.last_mined_tile = None
        self._generate_floor()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        action = int(action)

        self.current_step += 1
        prev_energy = self.energy
        prev_floor = self.floor
        self.last_mined_tile = None
        has_visited_before = False
        terminated = False
        truncated = False

        if 0 <= action <= 7:  # movement
            dest = self.agent_location + self._action_to_direction[action]
            new_x = int(np.clip(dest[0], 0, self.SIZE - 1))
            new_y = int(np.clip(dest[1], 0, self.SIZE - 1))
            has_visited_before = self.visited[new_y, new_x]
            self.visited[new_y, new_x] = True
            self.agent_location = np.array([new_x, new_y])
            if self.MOVE_COST != 0:
                self.energy = max(0, self.energy - self.MOVE_COST)

        elif 8 <= action <= 15:  # mining
            tx, ty = self._compute_target_from_mine_action(action)
            if 0 <= tx < self.SIZE and 0 <= ty < self.SIZE:
                has_visited_before = self.visited[ty, tx]
                self.visited[ty, tx] = True
            self._mine_tile(action)

        elif action == self.ACTION_DESCEND:
            if self._ladder_location:
                lx, ly = self._ladder_location
                has_visited_before = self.visited[ly, lx]
            self._attempt_descend()

        if self.energy <= 0 or self._is_grid_empty():
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        tile_type = (
            self._map_tile_int_to_str(self.last_mined_tile)
            if 8 <= action <= 15
            else self._get_tile_type()
        )

        reward = compute_reward(
            tile_type,
            self._action_name(action),
            bool(has_visited_before),
            prev_energy,
            self.energy,
            prev_floor,
            self.floor,
            include_weeds=True,
        )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ---------------- Internal helpers -----------------------
    def _compute_target_from_mine_action(self, action):
        direction = self._action_to_direction[action - 8]
        return (
            int(self.agent_location[0] + direction[0]),
            int(self.agent_location[1] + direction[1]),
        )

    def _mine_tile(self, action):
        self.energy -= 1
        tx, ty = self._compute_target_from_mine_action(action)
        if 0 <= tx < self.SIZE and 0 <= ty < self.SIZE:
            tile = int(self.grid[ty, tx])
            self.last_mined_tile = tile
            if tile != self.LADDER:
                self.grid[ty, tx] = self.EMPTY

    def _attempt_descend(self):
        if self.floor >= self.MAX_FLOOR - 1:
            return
        ax, ay = self.agent_location
        if self._ladder_location != (int(ax), int(ay)):
            return
        self.energy -= 1
        self.floor += 1
        self._generate_floor()
        self.visited = np.zeros((self.SIZE, self.SIZE), bool)
        self.visited[self.agent_location[1], self.agent_location[0]] = True
        self.last_mined_tile = None

    def _generate_floor(self):
        self.grid = np.full((self.SIZE, self.SIZE), self.EMPTY)
        possible = []
        prob_rock = 0.25
        prob_ore = 0.10
        prob_weed = 0.20
        ax, ay = self.agent_location
        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if (x, y) == (ax, ay):
                    continue
                r = self.np_random.random()
                if r < prob_rock:
                    self.grid[y, x] = self.ROCK
                    possible.append((x, y))
                elif r < prob_rock + prob_ore:
                    self.grid[y, x] = self.ORE
                    possible.append((x, y))
                elif r < prob_rock + prob_ore + prob_weed:
                    self.grid[y, x] = self.WEED
                    possible.append((x, y))

        if not possible:
            x = int(self.np_random.integers(0, self.SIZE))
            y = int(self.np_random.integers(0, self.SIZE))
            self.grid[y, x] = self.ROCK
            possible.append((x, y))

        if self.floor < self.MAX_FLOOR - 1:
            lx, ly = possible[int(self.np_random.integers(0, len(possible)))]
            self._ladder_location = (lx, ly)
            self.grid[ly, lx] = self.LADDER
        else:
            self._ladder_location = None

    def _is_grid_empty(self):
        return not np.any(
            (self.grid == self.ROCK) | (self.grid == self.ORE) | (self.grid == self.WEED)
        )

    def _map_tile_int_to_str(self, v):
        mapping = {
            self.EMPTY: "empty",
            self.ROCK: "rock",
            self.ORE: "ore",
            self.LADDER: "ladder",
            self.WEED: "weed",
        }
        return mapping.get(v, "unknown")

    def _get_tile_type(self):
        x, y = self.agent_location
        return self._map_tile_int_to_str(int(self.grid[y, x]))

    def _action_name(self, action):
        if action == self.ACTION_DESCEND:
            return "descend"
        if 8 <= action <= 15:
            return "mine"
        return "move"

    def _get_obs(self):
        half = self.LOCAL_VIEW_SIZE // 2
        ax, ay = self.agent_location
        x0, y0 = ax - half, ay - half
        x1, y1 = ax + half + 1, ay + half + 1

        local_view = np.full(
            (self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE),
            self.OUT_OF_BOUND,
            float,
        )

        gx0, gy0 = max(0, x0), max(0, y0)
        gx1, gy1 = min(self.SIZE, x1), min(self.SIZE, y1)
        patch = self.grid[gy0:gy1, gx0:gx1].astype(float)
        px0, py0 = gx0 - x0, gy0 - y0
        local_view[py0:py0+patch.shape[0], px0:px0+patch.shape[1]] = patch
        local_view = np.expand_dims(local_view, axis=-1)

        return {
            "agent_location": self.agent_location.astype(float),
            "energy": np.array([float(self.energy)], float),
            "floor": np.array([float(self.floor)], float),
            "local_view": local_view,
        }

    def _get_info(self):
        return {
            "ladder_location": self._ladder_location,
            "last_mined_tile": (
                None if self.last_mined_tile is None else int(self.last_mined_tile)
            ),
            "last_mined_tile_type": (
                None if self.last_mined_tile is None else
                self._map_tile_int_to_str(self.last_mined_tile)
            ),
            "agent_x": int(self.agent_location[0]),
            "agent_y": int(self.agent_location[1]),
            "energy": float(self.energy),
            "floor": int(self.floor),
        }

    def render(self, render_mode="human"):
        g = self.grid.copy()
        x, y = self.agent_location
        g[y, x] = self.AGENT
        print("Floor:", self.floor, "Energy:", self.energy)
        print(g)