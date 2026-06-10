import numpy as np
import gymnasium as gym
import minigrid
from minigrid.core.constants import DIR_TO_VEC
import torch
import pickle
import random
from pathlib import Path
from PIL import Image
import os

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Re-use the same ReplayBuffer class exactly
from data_collector_v2 import ReplayBuffer


class FourRoomsCollector:
    """
    Collects transitions in MiniGrid-FourRooms-v0.

    Key difference from Empty collector:
        FourRooms has internal walls — can't visit every (x,y).
        We detect walkable cells by checking the grid object type.
        Wall cells are skipped automatically.
    """
    def __init__(self, env_id="MiniGrid-FourRooms-v0", img_size=64):
        self.env_id   = env_id
        self.img_size = img_size
        self.env      = gym.make(env_id, render_mode="rgb_array")

        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height

        # Discover walkable cells from the grid
        self.walkable = self._find_walkable_cells()
        self.n_dirs   = 4

        total_states = len(self.walkable) * self.n_dirs
        print(f"Environment:   {env_id}")
        print(f"Grid size:     {self.width} x {self.height}")
        print(f"Walkable cells:{len(self.walkable)}")
        print(f"Total states:  {total_states}")
        print(f"Transitions per reset: {total_states * 3}")

    def _find_walkable_cells(self):
        """
        Find all non-wall cells by inspecting the grid.
        Returns list of (x, y) tuples.
        """
        walkable = []
        grid = self.env.unwrapped.grid
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                cell = grid.get(x, y)
                # None = empty floor, Goal = goal cell — both walkable
                if cell is None or cell.type == 'goal':
                    walkable.append((x, y))
        return walkable

    def preprocess(self, frame):
        img = Image.fromarray(frame).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        return np.array(img, dtype=np.uint8).transpose(2, 0, 1)

    def set_agent_state(self, x, y, direction):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def collect_from_state(self, x, y, direction, buffer):
        """Collect all 3 actions from a single (x, y, dir) state."""
        self.set_agent_state(x, y, direction)
        frame_t = self.env.render()
        obs_t   = self.preprocess(frame_t)
        pos_t   = np.array([x, y], dtype=np.float32)

        next_dir_left    = (direction - 1) % 4
        next_dir_right   = (direction + 1) % 4
        next_dir_forward = direction

        # Action 0: Left
        self.set_agent_state(x, y, direction)
        self.env.step(0)
        obs_t1_left = self.preprocess(self.env.render())
        # Get actual next position (forward into wall stays put)
        next_pos_left = np.array(self.env.unwrapped.agent_pos, dtype=np.float32)
        buffer.add(obs_t, 0, obs_t1_left, pos_t, direction, next_dir_left)

        # Action 1: Right
        self.set_agent_state(x, y, direction)
        self.env.step(1)
        obs_t1_right = self.preprocess(self.env.render())
        buffer.add(obs_t, 1, obs_t1_right, pos_t, direction, next_dir_right)

        # Action 2: Forward
        self.set_agent_state(x, y, direction)
        self.env.step(2)
        obs_t1_fwd = self.preprocess(self.env.render())
        next_pos_fwd = np.array(self.env.unwrapped.agent_pos, dtype=np.float32)
        buffer.add(obs_t, 2, obs_t1_fwd, pos_t, direction, next_dir_forward)

    def collect(self, buffer, n_resets=20, log_every=5):
        """
        Systematic collection across all walkable states.
        n_resets: FourRooms randomizes wall openings each reset,
                  so we rediscover walkable cells each time.
        """
        print(f"\nCollecting across {n_resets} resets...")

        for reset_idx in range(n_resets):
            self.env.reset()

            # Rediscover walkable cells — doorway positions change each reset
            self.walkable = self._find_walkable_cells()

            for (x, y) in self.walkable:
                for direction in range(self.n_dirs):
                    self.collect_from_state(x, y, direction, buffer)

            if (reset_idx + 1) % log_every == 0:
                print(f"  Reset [{reset_idx+1:>3}/{n_resets}]  "
                      f"buffer_size={len(buffer):>7}")

        print(f"\nCollection complete. Total: {len(buffer)} transitions")
        buffer.shuffle()
        return buffer


if __name__ == "__main__":
    # First, inspect the environment
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env.reset()
    print(f"Grid dimensions: {env.unwrapped.width} x {env.unwrapped.height}")
    env.close()

    buffer    = ReplayBuffer(capacity=70_000)  # larger — FourRooms is bigger
    collector = FourRoomsCollector(
        env_id   = "MiniGrid-FourRooms-v0",
        img_size = 64,
    )

    collector.collect(buffer, n_resets=20, log_every=5)
    buffer.save("data/replay_buffer_fourrooms.pkl")

    # Verify
    batch = buffer.sample(32)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    all_actions = buffer.actions[:buffer.size]
    print(f"\nAction distribution:")
    for a, name in enumerate(["Left", "Right", "Forward"]):
        count = (all_actions == a).sum()
        print(f"  {name}: {count} ({count/buffer.size*100:.1f}%)")