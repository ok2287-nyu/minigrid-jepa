import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import gymnasium as gym
import minigrid
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)

from encoder_v2 import Encoder, Predictor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load FourRooms world model
ckpt = torch.load("checkpoints/jepa_fourrooms_final.pt", map_location=device, weights_only=False)
encoder = Encoder(256).to(device)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

import copy
predictor_net = Predictor(256, 3).to(device)
predictor_net.load_state_dict(ckpt["predictor"])
predictor_net.eval()

env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
env.reset()

def encode(x, y, d):
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)

def build_state(z, d):
    dir_oh = F.one_hot(torch.tensor([d], dtype=torch.long).to(device), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def predict_next(z, d, action):
    state = build_state(z, d)
    action_t = torch.tensor([action], dtype=torch.long).to(device)
    with torch.no_grad():
        return predictor_net(state, action_t)

# ── Test 1: Z consistency ─────────────────────────────────────────────
print("=" * 55)
print("TEST 1: Z consistency (same pos encoded 3 times)")
print("=" * 55)
for (x, y, d) in [(3, 3, 0), (9, 9, 1), (15, 15, 2)]:
    zs = torch.cat([encode(x, y, d) for _ in range(3)], dim=0)
    std = zs.std(dim=0).mean().item()
    dist = (zs[0] - zs[1]).norm().item()
    print(f"  ({x},{y}) dir={d}: std={std:.6f}  dist(z0,z1)={dist:.6f}")

# ── Test 2: Z distances between positions ─────────────────────────────
print("\n" + "=" * 55)
print("TEST 2: Z distances between different positions")
print("=" * 55)
pairs = [
    ((3, 3, 0), (4, 3, 0)),    # adjacent, same dir
    ((3, 3, 0), (3, 3, 1)),    # same pos, diff dir
    ((3, 3, 0), (15, 15, 0)),  # far apart
    ((9, 1, 0), (9, 9, 0)),    # cross room
]
for (p1, p2) in pairs:
    z1 = encode(*p1)
    z2 = encode(*p2)
    dist = (z1 - z2).norm().item()
    print(f"  {p1} vs {p2}: dist={dist:.4f}")

# ── Test 3: Wall detection (THE CRITICAL NEW TEST) ────────────────────
print("\n" + "=" * 55)
print("TEST 3: Wall detection")
print("Expected: forward into wall ≈ same z (agent stays put)")
print("Expected: forward into open >> different z (agent moves)")
print("=" * 55)

# Find wall cells by checking grid
grid = env.unwrapped.grid

def is_walkable(x, y):
    cell = grid.get(x, y)
    return cell is None or cell.type == 'goal'

wall_tests = []
open_tests  = []

for x in range(2, 17):
    for y in range(2, 17):
        if not is_walkable(x, y):
            continue
        for d in range(4):
            dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
            nx, ny = x+dx, y+dy
            if not is_walkable(nx, ny):
                wall_tests.append((x, y, d))
            elif is_walkable(nx, ny):
                open_tests.append((x, y, d))

wall_tests = wall_tests[:10]
open_tests  = open_tests[:10]

wall_dists = []
for (x, y, d) in wall_tests:
    z_curr = encode(x, y, d)
    z_pred = predict_next(z_curr, d, 2)  # forward
    dist = (z_pred[:, :256] - z_curr).norm().item()
    wall_dists.append(dist)

open_dists = []
for (x, y, d) in open_tests:
    z_curr = encode(x, y, d)
    z_pred = predict_next(z_curr, d, 2)  # forward
    dist = (z_pred[:, :256] - z_curr).norm().item()
    open_dists.append(dist)

print(f"  Forward into WALL: avg dist = {np.mean(wall_dists):.4f}  (want SMALL)")
print(f"  Forward into OPEN: avg dist = {np.mean(open_dists):.4f}  (want LARGE)")
ratio = np.mean(open_dists) / (np.mean(wall_dists) + 1e-8)
print(f"  Ratio open/wall = {ratio:.2f}x  (want >> 1.0)")

# ── Test 4: Direction transitions ─────────────────────────────────────
print("\n" + "=" * 55)
print("TEST 4: Direction transition accuracy")
print("=" * 55)
correct = 0
total   = 0
for (x, y) in [(3,3), (9,9), (15,15), (5,5), (12,3)]:
    if not is_walkable(x, y):
        continue
    for d in range(4):
        for action, expected_next_d in [(0, (d-1)%4), (1, (d+1)%4), (2, d)]:
            z = encode(x, y, d)
            pred_state = predict_next(z, d, action)
            pred_dir = pred_state[:, 256:].argmax(dim=-1).item()
            correct += (pred_dir == expected_next_d)
            total   += 1

print(f"  Direction accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print("\nDone.")