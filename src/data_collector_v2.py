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


class ReplayBuffer:
    """
    Identical to v1 — stores (obs_t, action, obs_t+1, position).
    Pre-allocated numpy arrays for speed.
    uint8 storage, float32 on sample.
    """
    def __init__(self, capacity=200_000, obs_shape=(3, 64, 64)):
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.ptr        = 0
        self.size       = 0

        self.obs        = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs   = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions    = np.zeros((capacity,),            dtype=np.int64)
        self.positions  = np.zeros((capacity, 2),          dtype=np.float32)
        self.directions = np.zeros((capacity,),            dtype=np.int64)  # NEW
        self.next_directions = np.zeros((capacity,), dtype=np.int64)

    def add(self, obs, action, next_obs, position, direction, next_direction):
        self.obs[self.ptr]             = obs
        self.next_obs[self.ptr]        = next_obs
        self.actions[self.ptr]         = action
        self.positions[self.ptr]       = position
        self.directions[self.ptr]      = direction
        self.next_directions[self.ptr] = next_direction
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def shuffle(self):
        indices = np.random.permutation(self.size)
        self.obs[:self.size]        = self.obs[indices]
        self.next_obs[:self.size]   = self.next_obs[indices]
        self.actions[:self.size]    = self.actions[indices]
        self.positions[:self.size]  = self.positions[indices]
        self.directions[:self.size] = self.directions[indices]  # ADD THIS
        self.next_directions[:self.size] = self.next_directions[indices]
        print("Buffer shuffled.")
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs        = torch.tensor(
                self.obs[idxs], dtype=torch.float32) / 255.0,
            next_obs   = torch.tensor(
                self.next_obs[idxs], dtype=torch.float32) / 255.0,
            actions    = torch.tensor(
                self.actions[idxs], dtype=torch.long),
            positions  = torch.tensor(
                self.positions[idxs], dtype=torch.float32),
            directions = torch.tensor(
                self.directions[idxs], dtype=torch.long),     # NEW
            next_directions = torch.tensor(
                self.next_directions[idxs], dtype=torch.long),
        )

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Buffer saved to {path} ({self.size} transitions)")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"Buffer loaded from {path} ({self.size} transitions)")

    def __len__(self):
        return self.size


class SystematicDataCollector:
    """
    Collects transitions by visiting every (x, y, direction) state
    explicitly and sampling all 3 actions from each state.

    For each state:
        1. Set agent position and direction directly
        2. Record (obs_t, left,    obs_t+1_left)    → undo
        3. Record (obs_t, right,   obs_t+1_right)   → undo
        4. Record (obs_t, forward, obs_t+1_forward)

    Repeat across N random environment resets to get visual diversity
    (goal square appears in different positions each reset).

    Final buffer is shuffled to break sequential ordering.
    """
    def __init__(self, env_id="MiniGrid-Empty-16x16-v0", img_size=64):
        self.env_id   = env_id
        self.img_size = img_size
        self.env      = gym.make(env_id, render_mode="rgb_array")

        # Get grid dimensions
        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height

        # Walkable cells: exclude boundary walls
        # In MiniGrid, boundary is 1 cell thick
        self.x_range = range(1, self.width  - 1)
        self.y_range = range(1, self.height - 1)
        self.n_dirs  = 4   # 0=right, 1=down, 2=left, 3=up

        total_states = (
            len(self.x_range) * len(self.y_range) * self.n_dirs
        )
        print(f"Environment: {env_id}")
        print(f"Grid size:   {self.width} x {self.height}")
        print(f"Total states:{total_states} "
              f"({len(self.x_range)}x{len(self.y_range)} cells "
              f"x {self.n_dirs} directions)")
        print(f"Transitions per reset: {total_states * 3}")

    def preprocess(self, frame):
        """Raw frame → (3, 64, 64) uint8."""
        img = Image.fromarray(frame).resize(
            (self.img_size, self.img_size),
            Image.BILINEAR
        )
        arr = np.array(img, dtype=np.uint8)
        return arr.transpose(2, 0, 1)

    def set_agent_state(self, x, y, direction):
        """
        Directly set agent position and direction.
        Much cleaner than navigation chains.
        No deepcopy needed.
        """
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def collect_from_state(self, x, y, direction, buffer):
        self.set_agent_state(x, y, direction)
        frame_t = self.env.render()
        obs_t   = self.preprocess(frame_t)
        pos_t   = np.array([x, y], dtype=np.float32)

        # Compute next directions mathematically
        # Left turn:    direction rotates counterclockwise → (direction - 1) % 4
        # Right turn:   direction rotates clockwise        → (direction + 1) % 4
        # Forward:      direction unchanged                → direction
        next_dir_left    = (direction - 1) % 4
        next_dir_right   = (direction + 1) % 4
        next_dir_forward = direction

        # Action 0: Left
        self.set_agent_state(x, y, direction)
        self.env.step(0)
        frame_t1 = self.env.render()
        obs_t1   = self.preprocess(frame_t1)
        buffer.add(obs_t, 0, obs_t1, pos_t, direction, next_dir_left)

        # Action 1: Right
        self.set_agent_state(x, y, direction)
        self.env.step(1)
        frame_t1 = self.env.render()
        obs_t1   = self.preprocess(frame_t1)
        buffer.add(obs_t, 1, obs_t1, pos_t, direction, next_dir_right)

        # Action 2: Forward
        self.set_agent_state(x, y, direction)
        self.env.step(2)
        frame_t1 = self.env.render()
        obs_t1   = self.preprocess(frame_t1)
        buffer.add(obs_t, 2, obs_t1, pos_t, direction, next_dir_forward)

    def collect(self, buffer, n_resets=20, log_every=5):
        """
        Full systematic collection.

        n_resets: number of environment resets
                  each reset randomizes goal position
                  giving visual diversity across same dynamics

        For each reset:
            traverse all (x, y, direction) states
            collect 3 transitions per state
        """
        transitions_per_reset = (
            len(self.x_range) *
            len(self.y_range) *
            self.n_dirs * 3
        )
        total_expected = transitions_per_reset * n_resets

        print(f"\nCollecting {total_expected} transitions "
              f"across {n_resets} resets...")
        print(f"({transitions_per_reset} transitions per reset)\n")

        for reset_idx in range(n_resets):

            # Reset environment → randomizes goal position
            self.env.reset()

            # Traverse every (x, y, direction) state
            for x in self.x_range:
                for y in self.y_range:
                    for direction in range(self.n_dirs):
                        self.collect_from_state(
                            x, y, direction, buffer
                        )

            if (reset_idx + 1) % log_every == 0:
                print(f"  Reset [{reset_idx+1:>3}/{n_resets}]  "
                      f"buffer_size={len(buffer):>7}")

        # Shuffle to break sequential ordering
        print(f"\nCollection complete. Total: {len(buffer)} transitions")
        buffer.shuffle()
        return buffer


if __name__ == "__main__":
    buffer    = ReplayBuffer(capacity=200_000)
    collector = SystematicDataCollector(
        env_id   = "MiniGrid-Empty-16x16-v0",
        img_size = 64
    )

    collector.collect(buffer, n_resets=20, log_every=5)
    buffer.save("data/replay_buffer_phase1.pkl")

    # Verify shapes and balance
    batch = buffer.sample(32)
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} | dtype: {v.dtype}")

    # Verify action balance
    all_actions = buffer.actions[:buffer.size]
    print(f"\nAction distribution:")
    for a, name in enumerate(["Left", "Right", "Forward"]):
        count = (all_actions == a).sum()
        pct   = count / buffer.size * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"\nPixel range: [{batch['obs'].min():.3f}, "
          f"{batch['obs'].max():.3f}]")