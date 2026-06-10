import pickle
import numpy as np

with open("data/replay_buffer_phase1.pkl", "rb") as f:
    buf = pickle.load(f)

actions = buf['actions']
positions = buf['positions']
directions = buf['directions']

# Find where the real data ends
# Real positions should be in range [1,14], zeros are fake
real_mask = (positions[:, 0] > 0) | (positions[:, 1] > 0)
print(f"Entries with non-zero position: {real_mask.sum()}")
print(f"Buffer size field: {buf['size']}")

# Check action distribution on REAL entries only
real_actions = actions[real_mask]
print(f"\nReal data action distribution:")
print(f"  Left:    {(real_actions==0).sum()} ({(real_actions==0).mean()*100:.1f}%)")
print(f"  Right:   {(real_actions==1).sum()} ({(real_actions==1).mean()*100:.1f}%)")
print(f"  Forward: {(real_actions==2).sum()} ({(real_actions==2).mean()*100:.1f}%)")

# Check if obs frames are also zeros for fake entries
obs = buf['obs']
obs_nonzero = (obs.reshape(len(obs), -1).sum(axis=1) > 0)
print(f"\nEntries with non-zero obs: {obs_nonzero.sum()}")
print(f"Overlap (real pos AND real obs): {(real_mask & obs_nonzero).sum()}")