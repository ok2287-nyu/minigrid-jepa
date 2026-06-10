# src/test_encoder_variance.py
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
ckpt = torch.load("checkpoints/jepa_phase1_final.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
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

# Collect z vectors from varied positions
zs = []
positions = [(1,1,0),(1,1,1),(1,1,2),(1,1,3),
             (7,7,0),(7,7,1),(14,14,0),(14,14,2),
             (3,5,1),(10,2,3),(8,8,0),(4,12,2)]

for (x,y,d) in positions:
    z = encode(x, y, d)
    zs.append(z)

zs = torch.cat(zs, dim=0)  # (12, 256)

# Check variance
print(f"z mean:  {zs.mean().item():.4f}")
print(f"z std:   {zs.std().item():.4f}")
print(f"z min:   {zs.min().item():.4f}")
print(f"z max:   {zs.max().item():.4f}")

# Pairwise distances between different positions
print("\nPairwise L2 distances between different positions:")
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = (zs[i] - zs[j]).norm().item()
        print(f"  {positions[i]} vs {positions[j]}: {dist:.4f}")