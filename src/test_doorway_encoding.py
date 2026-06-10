# src/test_doorway_encoding.py
import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import gymnasium as gym
import minigrid
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)
from encoder_v2 import Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder(256).to(device)
ckpt = torch.load("checkpoints/jepa_fourrooms_final.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
env.reset()

grid = env.unwrapped.grid

def encode(x, y, d=0):
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)

# Find doorway cells and regular floor cells
# Find the actual doorway cells — floor cells with walls on exactly 2 opposite sides
doorway_cells = []
floor_cells   = []
wall_adjacent = []

for x in range(1, 18):
    for y in range(1, 18):
        cell = grid.get(x, y)
        if cell is not None:
            continue  # skip walls

        neighbors = {
            'E': grid.get(x+1, y),
            'W': grid.get(x-1, y),
            'N': grid.get(x, y-1),
            'S': grid.get(x, y+1),
        }

        def is_wall(c):
            return c is not None and c.type == 'wall'

        n_walls = sum(is_wall(v) for v in neighbors.values())

        # Doorway = floor cell with walls on exactly 2 opposite sides
        if (is_wall(neighbors['E']) and is_wall(neighbors['W'])) or \
           (is_wall(neighbors['N']) and is_wall(neighbors['S'])):
            doorway_cells.append((x, y))
        elif n_walls >= 1:
            wall_adjacent.append((x, y))
        else:
            floor_cells.append((x, y))

print(f"Doorway cells:       {len(doorway_cells)}")
print(f"Wall-adjacent floor: {len(wall_adjacent)}")
print(f"Open floor cells:    {len(floor_cells)}")
# Encode samples from each category
print("\n=== Z distances between cell types ===")

# Encode a few of each
n = min(5, len(doorway_cells), len(floor_cells), len(wall_adjacent))

door_zs  = torch.cat([encode(*c) for c in doorway_cells[:n]])
floor_zs = torch.cat([encode(*c) for c in floor_cells[:n]])
wall_zs  = torch.cat([encode(*c) for c in wall_adjacent[:n]])

# Average pairwise distances between categories
def avg_dist(zs1, zs2):
    dists = []
    for z1 in zs1:
        for z2 in zs2:
            dists.append((z1 - z2).norm().item())
    return np.mean(dists)

print(f"door  vs floor: {avg_dist(door_zs, floor_zs):.4f}")
print(f"door  vs wall_adj: {avg_dist(door_zs, wall_zs):.4f}")
print(f"floor vs wall_adj: {avg_dist(floor_zs, wall_zs):.4f}")
print(f"floor vs floor: {avg_dist(floor_zs, floor_zs):.4f}  (baseline)")

print("\n=== Can encoder distinguish doorway from wall-adjacent? ===")
print("If door vs wall_adj >> floor vs floor → encoder sees doorways as special")
print("If door vs wall_adj ≈ floor vs floor  → encoder can't distinguish them")