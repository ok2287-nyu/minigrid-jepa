import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from data_collector_v2 import ReplayBuffer

buffer = ReplayBuffer(capacity=200_000)
buffer.load("data/replay_buffer_phase1.pkl")

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle("Partial observations by direction\n(should look different for each direction)")

dir_names = ["East", "South", "West", "North"]

for d in range(4):
    # Find buffer entries with this direction
    mask = buffer.directions[:buffer.size] == d
    idxs = np.where(mask)[0][:2]  # get 2 examples

    for row, idx in enumerate(idxs):
        ax  = axes[row, d]
        img = buffer.obs[idx].transpose(1, 2, 0)  # (64, 64, 3)
        ax.imshow(img)
        ax.set_title(f"{dir_names[d]}\nidx={idx}")
        ax.axis("off")

plt.tight_layout()
plt.savefig("notebooks/partial_obs_check.png", dpi=100)
plt.show()
print("Saved to notebooks/partial_obs_check.png")