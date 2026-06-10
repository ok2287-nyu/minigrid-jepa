"""
Diagnose hierarchical system bottlenecks.

Test 1: High-level accuracy — does it predict correct subgoals?
Test 2: Low-level accuracy — given correct subgoals, does it reach them?
Test 3: Full system — both together
"""

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
from controller_highlevel import HighLevelController, bfs_path, find_doorways, extract_subgoals

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load everything
encoder = Encoder(256).to(device)
ckpt = torch.load("checkpoints/jepa_fourrooms_final.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

lowlevel = Controller(state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2).to(device)
ckpt2 = torch.load("checkpoints/controller_fourrooms_fourrooms_final.pt", map_location=device, weights_only=False)
lowlevel.load_state_dict(ckpt2["controller"])
lowlevel.eval()
for p in lowlevel.parameters():
    p.requires_grad = False

highlevel = HighLevelController(state_dim=260, hidden_dim=256, pos_dim=2).to(device)
ckpt3 = torch.load("checkpoints/highlevel_hierarchical_final.pt", map_location=device, weights_only=False)
highlevel.load_state_dict(ckpt3["highlevel"])
highlevel.eval()

env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
env.reset()
width  = env.unwrapped.width
height = env.unwrapped.height
max_x  = width  - 2
max_y  = height - 2

def find_walkable():
    grid = env.unwrapped.grid
    cells = []
    for x in range(1, width-1):
        for y in range(1, height-1):
            cell = grid.get(x, y)
            if cell is None or cell.type in ('goal', 'door'):
                cells.append((x, y))
    return cells

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

def build_pos(x, y):
    return torch.tensor([[x/max_x, y/max_y]], dtype=torch.float32).to(device)

def manual_step(x, y, d, action, walkable_set):
    if action == 0: return x, y, (d-1)%4
    elif action == 1: return x, y, (d+1)%4
    else:
        dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
        nx, ny = x+dx, y+dy
        if (nx,ny) in walkable_set: return nx,ny,d
        return x,y,d

N_TESTS = 100

# ═══════════════════════════════════════════════════════════
# TEST 1: High-level subgoal prediction accuracy
# Does the high-level predict the right doorway?
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: High-level subgoal prediction accuracy")
print("=" * 60)

correct_subgoal  = 0
within_2_cells   = 0
within_5_cells   = 0
total_hl         = 0
cross_room_total = 0
cross_room_correct = 0

env.reset()
walkable     = find_walkable()
walkable_set = set(walkable)

for _ in range(N_TESTS):
    env.reset()
    walkable     = find_walkable()
    walkable_set = set(walkable)
    doorways     = find_doorways(env.unwrapped.grid, width, height)
    doorway_set  = set(doorways)

    for _ in range(200):
        start = walkable[np.random.randint(len(walkable))]
        goal  = walkable[np.random.randint(len(walkable))]
        if start == goal: continue
        path = bfs_path(start, goal, walkable_set)
        if path and len(path)-1 >= 3: break
    else:
        continue

    x_start, y_start = start
    x_goal,  y_goal  = goal
    d_goal = np.random.randint(0, 4)

    subgoals = extract_subgoals(path, doorway_set, (x_goal, y_goal))
    first_subgoal = subgoals[0]   # what the high-level should predict first
    crosses_room  = first_subgoal in doorway_set

    # Encode
    z_curr   = encode(x_start, y_start, 0)
    sc       = build_state(z_curr, 0)
    pc       = build_pos(x_start, y_start)
    z_goal_v = encode(x_goal, y_goal, d_goal)
    sg       = build_state(z_goal_v, d_goal)
    pg       = build_pos(x_goal, y_goal)

    with torch.no_grad():
        pred = highlevel(sc, sg, pc, pg)

    pred_x = int(round(pred[0,0].item() * max_x))
    pred_y = int(round(pred[0,1].item() * max_y))
    pred_x = max(1, min(max_x, pred_x))
    pred_y = max(1, min(max_y, pred_y))

    true_x, true_y = first_subgoal
    dist = abs(pred_x - true_x) + abs(pred_y - true_y)

    total_hl += 1
    if dist == 0:      correct_subgoal += 1
    if dist <= 2:      within_2_cells  += 1
    if dist <= 5:      within_5_cells  += 1

    if crosses_room:
        cross_room_total += 1
        if dist <= 2: cross_room_correct += 1

print(f"Exact match:        {correct_subgoal}/{total_hl} = {correct_subgoal/total_hl*100:.1f}%")
print(f"Within 2 cells:     {within_2_cells}/{total_hl} = {within_2_cells/total_hl*100:.1f}%")
print(f"Within 5 cells:     {within_5_cells}/{total_hl} = {within_5_cells/total_hl*100:.1f}%")
print(f"Cross-room (w/in 2):{cross_room_correct}/{cross_room_total} = {cross_room_correct/max(cross_room_total,1)*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# TEST 2: Low-level success rate given CORRECT subgoals
# If we give the low-level the exact BFS subgoal, how often does it succeed?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Low-level with correct BFS subgoals")
print("=" * 60)

ll_success    = 0
ll_total      = 0
by_dist       = {"1-3": [0,0], "4-6": [0,0], "7-10": [0,0]}

env.reset()
walkable     = find_walkable()
walkable_set = set(walkable)

for _ in range(N_TESTS):
    env.reset()
    walkable     = find_walkable()
    walkable_set = set(walkable)

    for _ in range(200):
        start = walkable[np.random.randint(len(walkable))]
        goal  = walkable[np.random.randint(len(walkable))]
        if start == goal: continue
        path = bfs_path(start, goal, walkable_set)
        if path and 1 <= len(path)-1 <= 10: break
    else:
        continue

    x_start, y_start = start
    x_goal,  y_goal  = goal
    d_start = np.random.randint(0, 4)
    d_goal  = np.random.randint(0, 4)
    bfs_dist = len(path) - 1

    if bfs_dist <= 3:   bucket = "1-3"
    elif bfs_dist <= 6: bucket = "4-6"
    else:               bucket = "7-10"

    # Encode goal
    z_goal_v   = encode(x_goal, y_goal, d_goal)
    state_goal = build_state(z_goal_v, d_goal)
    pos_goal   = build_pos(x_goal, y_goal)

    x, y, d   = x_start, y_start, d_start
    max_steps = bfs_dist * 4 + 10
    success   = False

    for step in range(max_steps):
        z_curr     = encode(x, y, d)
        state_curr = build_state(z_curr, d)
        pos_curr   = build_pos(x, y)

        with torch.no_grad():
            action = lowlevel(
                state_curr, state_goal, pos_curr, pos_goal
            ).argmax(dim=-1).item()

        x, y, d = manual_step(x, y, d, action, walkable_set)
        if (x,y) == (x_goal, y_goal):
            success = True
            break

    ll_total += 1
    by_dist[bucket][1] += 1
    if success:
        ll_success += 1
        by_dist[bucket][0] += 1

print(f"Overall low-level:  {ll_success}/{ll_total} = {ll_success/ll_total*100:.1f}%")
for bucket, (s, t) in by_dist.items():
    if t > 0:
        print(f"  dist {bucket}: {s}/{t} = {s/t*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# TEST 3: Full system
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Full hierarchical system")
print("=" * 60)

full_success = 0
full_total   = 0

env.reset()
walkable     = find_walkable()
walkable_set = set(walkable)

for _ in range(N_TESTS):
    env.reset()
    walkable     = find_walkable()
    walkable_set = set(walkable)

    for _ in range(200):
        start = walkable[np.random.randint(len(walkable))]
        goal  = walkable[np.random.randint(len(walkable))]
        if start == goal: continue
        path = bfs_path(start, goal, walkable_set)
        if path and len(path)-1 >= 1: break
    else:
        continue

    x_start, y_start = start
    x_goal,  y_goal  = goal
    d_start = np.random.randint(0, 4)
    d_goal  = np.random.randint(0, 4)

    z_goal_v = encode(x_goal, y_goal, d_goal)
    sg_final = build_state(z_goal_v, d_goal)
    pg_final = build_pos(x_goal, y_goal)

    x, y, d   = x_start, y_start, d_start
    max_steps = (len(path)-1) * 6 + 20
    success   = False

    current_subgoal_pos   = None
    current_subgoal_state = None
    subgoal_steps         = 0
    max_subgoal_steps     = 20

    for step in range(max_steps):
        z_curr     = encode(x, y, d)
        state_curr = build_state(z_curr, d)
        pos_curr   = build_pos(x, y)

        if current_subgoal_pos is None or subgoal_steps >= max_subgoal_steps:
            with torch.no_grad():
                pred = highlevel(state_curr, sg_final, pos_curr, pg_final)
            sg_x = int(round(pred[0,0].item() * max_x))
            sg_y = int(round(pred[0,1].item() * max_y))
            sg_x = max(1, min(max_x, sg_x))
            sg_y = max(1, min(max_y, sg_y))
            if (sg_x, sg_y) not in walkable_set:
                nearest = min(walkable_set, key=lambda p: (p[0]-sg_x)**2+(p[1]-sg_y)**2)
                sg_x, sg_y = nearest
            current_subgoal_pos = (sg_x, sg_y)
            z_sg = encode(sg_x, sg_y, d_goal)
            current_subgoal_state = build_state(z_sg, d_goal)
            subgoal_steps = 0

        pos_subgoal = build_pos(*current_subgoal_pos)
        with torch.no_grad():
            action = lowlevel(
                state_curr, current_subgoal_state, pos_curr, pos_subgoal
            ).argmax(dim=-1).item()

        x, y, d = manual_step(x, y, d, action, walkable_set)
        subgoal_steps += 1

        if (x,y) == (x_goal, y_goal):
            success = True
            break
        if (x,y) == current_subgoal_pos:
            current_subgoal_pos = None
            subgoal_steps = 0

    full_total += 1
    if success: full_success += 1

# ═══════════════════════════════════════════════════════════
# TEST 4: Full system — 500 episodes (definitive evaluation)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Full hierarchical system — 500 episodes")
print("=" * 60)

full_success = 0
full_total   = 0
buckets_full = {
    "Short (1-5)":   {"success": 0, "total": 0},
    "Medium (6-12)": {"success": 0, "total": 0},
    "Long (13+)":    {"success": 0, "total": 0},
}

env.reset()
walkable     = find_walkable()
walkable_set = set(walkable)

for _ in range(500):
    env.reset()
    walkable     = find_walkable()
    walkable_set = set(walkable)

    for _ in range(200):
        start = walkable[np.random.randint(len(walkable))]
        goal  = walkable[np.random.randint(len(walkable))]
        if start == goal: continue
        path = bfs_path(start, goal, walkable_set)
        if path and len(path)-1 >= 1: break
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
    buckets_full[bucket]["total"] += 1

    z_goal_v = encode(x_goal, y_goal, d_goal)
    sg_final = build_state(z_goal_v, d_goal)
    pg_final = build_pos(x_goal, y_goal)

    x, y, d   = x_start, y_start, d_start
    max_steps = bfs_dist * 6 + 20
    success   = False

    current_subgoal_pos   = None
    current_subgoal_state = None
    subgoal_steps         = 0
    max_subgoal_steps     = 20

    for step in range(max_steps):
        z_curr     = encode(x, y, d)
        state_curr = build_state(z_curr, d)
        pos_curr   = build_pos(x, y)

        if current_subgoal_pos is None or subgoal_steps >= max_subgoal_steps:
            with torch.no_grad():
                pred = highlevel(state_curr, sg_final, pos_curr, pg_final)
            sg_x = int(round(pred[0,0].item() * max_x))
            sg_y = int(round(pred[0,1].item() * max_y))
            sg_x = max(1, min(max_x, sg_x))
            sg_y = max(1, min(max_y, sg_y))
            if (sg_x, sg_y) not in walkable_set:
                nearest = min(walkable_set, key=lambda p: (p[0]-sg_x)**2+(p[1]-sg_y)**2)
                sg_x, sg_y = nearest
            current_subgoal_pos = (sg_x, sg_y)
            z_sg = encode(sg_x, sg_y, d_goal)
            current_subgoal_state = build_state(z_sg, d_goal)
            subgoal_steps = 0

        pos_subgoal = build_pos(*current_subgoal_pos)
        with torch.no_grad():
            action = lowlevel(
                state_curr, current_subgoal_state, pos_curr, pos_subgoal
            ).argmax(dim=-1).item()

        x, y, d = manual_step(x, y, d, action, walkable_set)
        subgoal_steps += 1

        if (x,y) == (x_goal, y_goal):
            success = True
            break
        if (x,y) == current_subgoal_pos:
            current_subgoal_pos = None
            subgoal_steps = 0

    full_total += 1
    if success:
        full_success += 1
        buckets_full[bucket]["success"] += 1

    if full_total % 100 == 0:
        print(f"  Progress: {full_total}/500  running_sr={full_success/full_total*100:.1f}%")

overall = full_success / full_total * 100
print(f"\nFinal: {overall:.1f}% ({full_success}/{full_total})")
for name, data in buckets_full.items():
    if data["total"] > 0:
        pct = data["success"] / data["total"] * 100
        print(f"  {name}: {pct:.1f}% ({data['success']}/{data['total']})")

print(f"Full system: {full_success}/{full_total} = {full_success/full_total*100:.1f}%")
print("\nDone.")