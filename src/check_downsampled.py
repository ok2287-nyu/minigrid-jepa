import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from data_collector_v2 import ReplayBuffer
import gymnasium as gym
import minigrid

buffer = ReplayBuffer(capacity=200_000)
buffer.load("data/replay_buffer_phase1.pkl")

# Also load environment to know agent positions
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
fig.suptitle("Zoomed in on agent triangle\nCan direction be distinguished?")

dir_names = ["East", "South", "West", "North"]

# Use positions stored in buffer to find the agent
for d in range(4):
    mask = buffer.directions[:buffer.size] == d
    idxs = np.where(mask)[0][:2]

    for row, idx in enumerate(idxs):
        ax  = axes[row, d]
        img = buffer.obs[idx].transpose(1, 2, 0)  # (64, 64, 3)

        # Get agent position from buffer
        pos = buffer.positions[idx]  # (x, y) in grid coords
        x, y = int(pos[0]), int(pos[1])

        # Convert grid position to pixel position in 64x64 image
        # Grid is 16x16, image is 64x64 → 4 pixels per cell
        px = int((x / 16) * 64)
        py = int((y / 16) * 64)

        # Crop 16x16 pixels around agent
        x1 = max(0, px - 8)
        x2 = min(64, px + 8)
        y1 = max(0, py - 8)
        y2 = min(64, py + 8)

        crop = img[y1:y2, x1:x2, :]

        ax.imshow(crop, interpolation='nearest')
        ax.set_title(f"{dir_names[d]}\npos=({x},{y})", fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig("notebooks/triangle_zoom.png", dpi=200)
plt.show()
env.close()