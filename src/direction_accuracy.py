import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
from PIL import Image
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder
from data_collector_v2 import ReplayBuffer

# Load encoder
ckpt    = torch.load(
    "checkpoints/jepa_5000.pt",
    map_location="cpu", weights_only=False
)
encoder = Encoder(256)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

# Load buffer
buffer = ReplayBuffer(capacity=200_000)
buffer.load("data/replay_buffer_phase1.pkl")

# Collect z vectors and directions from environment
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

z_list   = []
dir_list = []
pos_list = []

print("Collecting samples...")
for _ in range(2000):
    env.reset()
    # Random position and direction
    x   = np.random.randint(1, 15)
    y   = np.random.randint(1, 15)
    d   = np.random.randint(0, 4)
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d

    frame = env.render()
    img   = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32) / 255.0
    arr   = arr.transpose(2, 0, 1)
    t     = torch.tensor(arr).unsqueeze(0)

    with torch.no_grad():
        z = encoder(t)

    z_list.append(z)
    dir_list.append(d)
    pos_list.append([x, y])

env.close()

z_all   = torch.cat(z_list, dim=0)          # (2000, 256)
dir_all = torch.tensor(dir_list)            # (2000,)
pos_all = torch.tensor(pos_list, dtype=torch.float32)  # (2000, 2)

# ── Test 1: Can a linear layer predict direction? ──────────────────
print("\nTraining direction probe...")
dir_probe = nn.Linear(256, 4)
opt       = torch.optim.Adam(dir_probe.parameters(), lr=1e-3)

# Train/val split
N     = len(z_all)
split = int(0.8 * N)
z_tr, z_val     = z_all[:split],  z_all[split:]
d_tr, d_val     = dir_all[:split], dir_all[split:]

for epoch in range(200):
    logits = dir_probe(z_tr)
    loss   = F.cross_entropy(logits, d_tr)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    val_logits   = dir_probe(z_val)
    val_pred     = val_logits.argmax(dim=-1)
    dir_accuracy = (val_pred == d_val).float().mean().item()

dir_names = ["East", "South", "West", "North"]
print(f"Direction prediction accuracy: {dir_accuracy*100:.1f}%")
print(f"Random baseline: 25.0%")

# Per direction
for i, name in enumerate(dir_names):
    mask = d_val == i
    if mask.sum() > 0:
        acc = (val_pred[mask] == d_val[mask]).float().mean().item()
        print(f"  {name}: {acc*100:.1f}%")

# ── Test 2: Can a linear layer predict position? ───────────────────
print("\nTraining position probe...")
pos_probe = nn.Linear(256, 2)
opt2      = torch.optim.Adam(pos_probe.parameters(), lr=1e-3)
p_tr, p_val = pos_all[:split], pos_all[split:]
p_max       = pos_all.max()

for epoch in range(200):
    pred = pos_probe(z_tr)
    loss = F.mse_loss(pred, p_tr / p_max)
    opt2.zero_grad()
    loss.backward()
    opt2.step()

with torch.no_grad():
    pred_val = pos_probe(z_val) * p_max
    pos_mse  = F.mse_loss(pred_val, p_val).item()
    pos_rmse = pos_mse ** 0.5

print(f"Position prediction RMSE: {pos_rmse:.2f} cells")
print(f"(lower is better, random ≈ 6-7 cells)")