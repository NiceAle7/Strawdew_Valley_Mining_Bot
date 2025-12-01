from typing import Optional
import numpy as np
import gymnasium as gym

__all__ = ["StardewMineEnv"]


class StardewMineEnv(gym.Env):
    """A small grid-based mining environment inspired by the notebook.

    Coordinates: agent_location is a numpy array [x, y] where x is column and y is row.
    """

    def __init__(self, size: int = 10, max_floor: int = 10, max_energy: int = 100, local_view_size: int = 5, seed: Optional[int] = None):
        # Grid size (size x size)
        self.SIZE = size
        self.MAX_FLOOR = max_floor
        self.MAX_ENERGY = max_energy
        self.LOCAL_VIEW_SIZE = local_view_size

        # State
        self.agent_location = np.array([0, 0], dtype=np.int32)
        self.grid = np.full((self.SIZE, self.SIZE), 0, dtype=np.int32)
        self.energy = self.MAX_ENERGY
        self.floor = 0
        self._ladder_location = None

        # Tile types
        self.EMPTY = 0
        self.LADDER = 1
        self.WEED = 2
        self.ROCK = 3
        self.ORE = 4
        self.OUT_OF_BOUND = -1
        self.MAX_TILE_TYPE = 4
        self.AGENT = 9

        # Actions
        self.action_space = gym.spaces.Discrete(17)
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

        # Movement mapping (indices 0..7)
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

        # Observation space
        self.observation_space = gym.spaces.Dict({
            'agent_location': gym.spaces.Box(low=0, high=self.SIZE - 1, shape=(2,), dtype=np.int32),
            'energy': gym.spaces.Box(low=0, high=self.MAX_ENERGY, shape=(), dtype=np.int32),
            'floor': gym.spaces.Box(low=0, high=self.MAX_FLOOR - 1, shape=(), dtype=np.int32),
            'local_view': gym.spaces.Box(low=self.OUT_OF_BOUND, high=self.MAX_TILE_TYPE, shape=(self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE), dtype=np.int32),
        })

        # Seed RNG (use gym's seeding API on reset)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Use gym's seeding to create self.np_random (numpy Generator)
        super().reset(seed=seed)
        # Place agent randomly inside the grid
        self.agent_location = np.array(self.np_random.integers(0, self.SIZE, size=2), dtype=np.int32)
        self.energy = self.MAX_ENERGY
        self.floor = 0
        self._generate_floor()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False

        # Descend
        if action == self.ACTION_DESCEND:
            if self.floor < self.MAX_FLOOR - 1 and self._ladder_location is not None:
                ax, ay = int(self.agent_location[0]), int(self.agent_location[1])
                lx, ly = self._ladder_location
                if abs(lx - ax) <= 1 and abs(ly - ay) <= 1:
                    self.floor += 1
                    self._generate_floor()
                    reward += 0.5

        # Mining actions
        elif 8 <= action <= 15:
            reward += self._mine_tile(action)

        # Movement actions
        else:
            if action not in range(0, 8):
                reward -= 0.1
            else:
                direction = self._action_to_direction[action]
                new_pos = self.agent_location + direction
                new_x, new_y = int(new_pos[0]), int(new_pos[1])
                new_x = max(0, min(self.SIZE - 1, new_x))
                new_y = max(0, min(self.SIZE - 1, new_y))
                if self.grid[new_y, new_x] == self.EMPTY:
                    self.agent_location = np.array([new_x, new_y], dtype=np.int32)
                    reward -= 0.05
                else:
                    reward -= 0.5

        # Terminal conditions
        if self._is_grid_empty():
            terminated = True
            reward += 5
        elif self.energy <= 0:
            terminated = True
            reward -= 10

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _generate_floor(self):
        self.grid = np.full((self.SIZE, self.SIZE), self.EMPTY, dtype=np.int32)
        prob_weed, prob_rock, prob_ore = 0.1, 0.3, 0.05
        possible = []
        for y in range(self.SIZE):
            for x in range(self.SIZE):
                if (x, y) == (int(self.agent_location[0]), int(self.agent_location[1])):
                    continue
                r = self.np_random.random()
                if r < prob_weed:
                    self.grid[y, x] = self.WEED
                elif r < prob_weed + prob_rock:
                    self.grid[y, x] = self.ROCK
                    possible.append((x, y))
                elif r < prob_weed + prob_rock + prob_ore:
                    self.grid[y, x] = self.ORE
                    possible.append((x, y))

        if not possible:
            while True:
                x = int(self.np_random.integers(0, self.SIZE))
                y = int(self.np_random.integers(0, self.SIZE))
                if (x, y) != (int(self.agent_location[0]), int(self.agent_location[1])) and self.grid[y, x] == self.EMPTY:
                    self.grid[y, x] = self.ROCK
                    possible.append((x, y))
                    break

        if self.floor < self.MAX_FLOOR - 1:
            idx = int(self.np_random.integers(0, len(possible)))
            self._ladder_location = possible[idx]
        else:
            self._ladder_location = None

    def _mine_tile(self, action: int):
        reward = 0.0
        self.energy -= 1
        dir_idx = action - 8
        direction = self._action_to_direction.get(dir_idx, np.array([0, 0]))
        tx = int(self.agent_location[0] + direction[0])
        ty = int(self.agent_location[1] + direction[1])
        if not (0 <= tx < self.SIZE and 0 <= ty < self.SIZE):
            return reward - 1
        tile = self.grid[ty, tx]
        if tile == self.ORE:
            reward += 1
        elif tile == self.EMPTY:
            reward -= 1
        else:
            reward -= 0.01
        self.grid[ty, tx] = self.EMPTY
        if self._ladder_location is not None and (tx, ty) == self._ladder_location:
            self.grid[ty, tx] = self.LADDER
        return reward

    def _is_grid_empty(self):
        return not np.any((self.grid == self.ROCK) | (self.grid == self.ORE))

    def _get_obs(self):
        half = self.LOCAL_VIEW_SIZE // 2
        ax = int(self.agent_location[0])
        ay = int(self.agent_location[1])
        x0 = ax - half
        y0 = ay - half
        x1 = ax + half + 1
        y1 = ay + half + 1
        local_view = np.full((self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE), self.OUT_OF_BOUND, dtype=np.int32)
        gx0 = max(0, x0)
        gy0 = max(0, y0)
        gx1 = min(self.SIZE, x1)
        gy1 = min(self.SIZE, y1)
        patch = self.grid[gy0:gy1, gx0:gx1]
        px0 = gx0 - x0
        py0 = gy0 - y0
        local_view[py0:py0 + patch.shape[0], px0:px0 + patch.shape[1]] = patch
        obs = {
            'agent_location': self.agent_location.copy(),
            'energy': np.int32(self.energy),
            'floor': np.int32(self.floor),
            'local_view': local_view,
        }
        return obs

    def _get_info(self):
        return {}

    def render(self, render_mode: str = 'human'):
        grid_copy = self.grid.copy()
        ax = int(self.agent_location[0])
        ay = int(self.agent_location[1])
        if 0 <= ay < self.SIZE and 0 <= ax < self.SIZE:
            grid_copy[ay, ax] = self.AGENT
        print('Floor:', self.floor, 'Energy:', self.energy)
        print(grid_copy)


if __name__ == '__main__':
    # Quick smoke test
    env = StardewMineEnv(size=8, seed=0)
    obs, info = env.reset()
    env.render()
    print('Local view:\n', obs['local_view'])
