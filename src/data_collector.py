import numpy as np
import gymnasium as gym
import minigrid
from collections import deque
import random
import torch
from pathlib import Path
import pickle
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

class ReplayBuffer:
    """
    Circular buffer storing (obs_t, action, obs_t+1) transitions.
    Observations stored as uint8 (0-255) to save memory, 
    converted to float32 tensors on sampling.
    """
    def __init__(self, capacity=100_000, obs_shape=(3, 64, 64)):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.ptr = 0        # where to write next
        self.size = 0       # how many valid entries exist

        # Pre-allocate memory upfront — faster than appending
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros((capacity,), dtype=np.int64)

        # Ground truth positions for linear probe evaluation ONLY
        # The world model never sees these during training
        self.positions = np.zeros((capacity, 2), dtype=np.float32)

    def add(self, obs, action, next_obs, position):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.positions[self.ptr] = position
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs      = torch.tensor(self.obs[idxs],      dtype=torch.float32) / 255.0,
            next_obs = torch.tensor(self.next_obs[idxs], dtype=torch.float32) / 255.0,
            actions  = torch.tensor(self.actions[idxs],  dtype=torch.long),
            positions= torch.tensor(self.positions[idxs],dtype=torch.float32),
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


class DataCollector:
    """
    Rolls out a random agent in MiniGrid and fills the ReplayBuffer.
    Handles all preprocessing: RGB rendering, resizing, channel ordering.
    """
    def __init__(self, env_id="MiniGrid-Empty-16x16-v0", img_size=64):
        self.env_id   = env_id
        self.img_size = img_size
        # render_mode="rgb_array" gives us raw pixel frames
        self.env = gym.make(env_id, render_mode="rgb_array")

    def preprocess(self, frame):
        """
        Convert raw env render to model-ready format:
        (H, W, 3) uint8  →  (3, 64, 64) uint8
        Resize via simple numpy slicing (PIL-free for speed)
        """
        from PIL import Image
        img = Image.fromarray(frame).resize(
            (self.img_size, self.img_size), 
            Image.BILINEAR
        )
        arr = np.array(img, dtype=np.uint8)   # (64, 64, 3)
        return arr.transpose(2, 0, 1)          # (3, 64, 64)

    def get_agent_position(self):
        """Extract (x, y) ground truth — used ONLY for probe evaluation."""
        return np.array(self.env.unwrapped.agent_pos, dtype=np.float32)

    def collect(self, buffer, n_transitions=50_000, log_every=5000):
        """
        Fill buffer with random-agent transitions.
        Action space: 0=turn left, 1=turn right, 2=move forward
        We bias toward forward to encourage more exploration.
        """
        print(f"Collecting {n_transitions} transitions in {self.env_id}...")
        n_actions = self.env.action_space.n

        collected = 0
        episodes  = 0

        while collected < n_transitions:
            obs_raw, _ = self.env.reset()
            frame       = self.env.render()
            obs         = self.preprocess(frame)
            done        = False
            episode_len = 0

            while not done and collected < n_transitions:
                # Bias action distribution: more forward, less turn
                # This gives richer spatial coverage
                action = random.choices(
                    population=[0, 1, 2],
                    weights   =[0.25, 0.25, 0.50]
                )[0]

                position = self.get_agent_position()

                _, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_frame   = self.env.render()
                next_obs     = self.preprocess(next_frame)

                buffer.add(obs, action, next_obs, position)

                obs = next_obs
                collected  += 1
                episode_len += 1

                if collected % log_every == 0:
                    print(f"  [{collected:>6}/{n_transitions}] "
                          f"episodes={episodes+1}  "
                          f"buffer_size={len(buffer)}")

            episodes += 1

        print(f"Done. {collected} transitions across {episodes} episodes.")
        return buffer


if __name__ == "__main__":
    # Quick test — collect 1000 transitions and verify shapes
    buf = ReplayBuffer(capacity=100_000)
    collector = DataCollector(env_id="MiniGrid-Empty-16x16-v0", img_size=64)
    collector.collect(buf, n_transitions=1_000, log_every=200)

    batch = buf.sample(32)
    print("\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} | dtype: {v.dtype}")

    print(f"\nObs pixel range: [{batch['obs'].min():.3f}, {batch['obs'].max():.3f}]")
    print(f"Actions sample:  {batch['actions'][:10].tolist()}")
    print(f"Positions sample:\n{batch['positions'][:5]}")