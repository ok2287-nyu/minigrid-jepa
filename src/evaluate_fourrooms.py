import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid
from collections import deque
import sys
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)

from encoder_v2 import Encoder
from controller_bc import Controller

def bfs_path(start, goal, walkable_set):
    if start == goal: return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (x,y), path = queue.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if (nx,ny) in walkable_set and (nx,ny) not in visited:
                new_path = path + [(nx,ny)]
                if (nx,ny) == goal: return new_path
                visited.add((nx,ny))
                queue.append(((nx,ny), new_path))
    return None

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder(256).to(device)
ckpt = torch.load("checkpoints/jepa_fourrooms_final.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

controller = Controller(state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2).to(device)
ckpt2 = torch.load("checkpoints/controller_fourrooms_3000.pt", map_location=device, weights_only=False)
controller.load_state_dict(ckpt2["controller"])
controller.eval()

env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
env.reset()
max_x = env.unwrapped.width  - 2
max_y = env.unwrapped.height - 2

def find_walkable():
    grid = env.unwrapped.grid
    cells = []
    for x in range(1, env.unwrapped.width-1):
        for y in range(1, env.unwrapped.height-1):
            cell = grid.get(x,y)
            if cell is None or cell.type in ('goal','door'):
                cells.append((x,y))
    return cells

def encode(x, y, d):
    env.unwrapped.agent_pos = np.array([x,y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img = Image.fromarray(frame).resize((64,64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.tensor(arr.transpose(2,0,1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)

def build_state(z, d):
    dir_oh = F.one_hot(torch.tensor([d], dtype=torch.long).to(device), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def build_pos(x, y):
    return torch.tensor([[x/max_x, y/max_y]], dtype=torch.float32).to(device)

def manual_step(x, y, d, action, walkable_set):
    if action == 0: return x, y, (d-1)%4
    elif action == 1: return x, y, (d+1)%4
    else:
        dx,dy = [1,0,-1,0][d],[0,1,0,-1][d]
        nx,ny = x+dx,y+dy
        if (nx,ny) in walkable_set: return nx,ny,d
        return x,y,d

print("Evaluating FourRooms controller over 500 episodes...")
print("=" * 60)

num_tests = 500
success_count = 0
buckets = {
    "Short (1-5)":   {"success": 0, "total": 0},
    "Medium (6-12)": {"success": 0, "total": 0},
    "Long (13+)":    {"success": 0, "total": 0},
}

env.reset()
walkable     = find_walkable()
walkable_set = set(walkable)

for test_idx in range(num_tests):
    # Sample valid start/goal pair
    for _ in range(200):
        start = walkable[np.random.randint(len(walkable))]
        goal  = walkable[np.random.randint(len(walkable))]
        if start == goal: continue
        path = bfs_path(start, goal, walkable_set)
        if path and len(path) >= 2: break
    else:
        continue

    x_start, y_start = start
    x_goal,  y_goal  = goal
    d_start = np.random.randint(0, 4)
    d_goal  = np.random.randint(0, 4)
    bfs_dist = len(path) - 1

    if bfs_dist <= 5:    bucket = "Short (1-5)"
    elif bfs_dist <= 12: bucket = "Medium (6-12)"
    else:                bucket = "Long (13+)"
    buckets[bucket]["total"] += 1

    # Encode goal
    z_goal     = encode(x_goal, y_goal, d_goal)
    state_goal = build_state(z_goal, d_goal)
    pos_goal   = build_pos(x_goal, y_goal)

    x, y, d   = x_start, y_start, d_start
    max_steps = bfs_dist * 4 + 10
    success   = False

    for step in range(max_steps):
        z_curr     = encode(x, y, d)
        state_curr = build_state(z_curr, d)
        pos_curr   = build_pos(x, y)

        with torch.no_grad():
            action = controller(
                state_curr, state_goal, pos_curr, pos_goal
            ).argmax(dim=-1).item()

        x, y, d = manual_step(x, y, d, action, walkable_set)
        if (x,y) == (x_goal, y_goal):
            success = True
            break

    if success:
        success_count += 1
        buckets[bucket]["success"] += 1

    if (test_idx + 1) % 100 == 0:
        print(f"  Progress: {test_idx+1}/500  running_sr={success_count/(test_idx+1)*100:.1f}%")

overall_sr = success_count / num_tests * 100
print("\n" + "="*25 + " FINAL RESULTS " + "="*25)
print(f"Overall Success Rate: {overall_sr:.2f}% ({success_count}/{num_tests})")
print("-" * 65)
print("Performance by BFS distance:")
for name, data in buckets.items():
    if data["total"] > 0:
        pct = data["success"] / data["total"] * 100
        print(f"  {name:15s}: {pct:>5.1f}% ({data['success']}/{data['total']})")
print("=" * 65)