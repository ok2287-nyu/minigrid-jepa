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

device = 'cuda'
encoder = Encoder(256).to(device)
ckpt = torch.load('checkpoints/jepa_fourrooms_final.pt', map_location=device, weights_only=False)
encoder.load_state_dict(ckpt['online_encoder'])
encoder.eval()

env = gym.make('MiniGrid-FourRooms-v0', render_mode='rgb_array')
env.reset()

def encode(x, y, d=0):
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)

# Sample positions from each room
# FourRooms center divider is at x=9, y=9
rooms = {
    0: [(3,3), (4,5), (6,2), (2,7)],      # top-left
    1: [(12,3), (14,5), (11,2), (13,7)],   # top-right
    2: [(3,12), (4,14), (6,11), (2,13)],   # bottom-left
    3: [(12,12), (14,14), (11,11), (13,13)] # bottom-right
}

# Encode all positions
room_zs = {}
for room_id, positions in rooms.items():
    zs = [encode(x, y) for x, y in positions]
    room_zs[room_id] = torch.cat(zs, dim=0)  # (4, 256)

print('=' * 55)
print('Average distances WITHIN same room vs BETWEEN rooms')
print('=' * 55)

# Within-room distances
within_dists = []
for room_id in range(4):
    zs = room_zs[room_id]
    dists = []
    for i in range(len(zs)):
        for j in range(i+1, len(zs)):
            dists.append((zs[i] - zs[j]).norm().item())
    avg = np.mean(dists)
    within_dists.append(avg)
    print(f'Room {room_id} within-room avg dist: {avg:.4f}')

print(f'\nOverall within-room avg: {np.mean(within_dists):.4f}')
print()

# Between-room distances
between_dists = []
for r1 in range(4):
    for r2 in range(r1+1, 4):
        dists = []
        for z1 in room_zs[r1]:
            for z2 in room_zs[r2]:
                dists.append((z1 - z2).norm().item())
        avg = np.mean(dists)
        between_dists.append(avg)
        print(f'Room {r1} vs Room {r2} avg dist: {avg:.4f}')

print(f'\nOverall between-room avg: {np.mean(between_dists):.4f}')
print()
print('=' * 55)
print(f'Ratio between/within: {np.mean(between_dists)/np.mean(within_dists):.2f}x')
print('Want >> 1.0 for rooms to be clustered in z space')
print('=' * 55)