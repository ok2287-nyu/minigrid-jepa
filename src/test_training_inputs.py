import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)

from encoder_v2 import Encoder
from controller_bc import Controller
import gymnasium as gym
import minigrid
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = Encoder(256).to(device)
ckpt = torch.load("checkpoints/jepa_phase1_final.pt", map_location=device, weights_only=False)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

controller = Controller(state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2).to(device)
optimizer = torch.optim.Adam(controller.parameters(), lr=3e-3)

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env.reset()
max_x, max_y = 14, 14

def preprocess(frame):
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

def encode(frame):
    with torch.no_grad():
        return encoder(preprocess(frame))

def build_state(z, d):
    dir_oh = F.one_hot(torch.tensor([d], dtype=torch.long).to(device), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def build_pos(x, y):
    return torch.tensor([[x / max_x, y / max_y]], dtype=torch.float32).to(device)

def set_state(x, y, d):
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d

def manual_step(x, y, d, action):
    if action == 0: return x, y, (d-1)%4
    elif action == 1: return x, y, (d+1)%4
    else:
        dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
        nx, ny = x+dx, y+dy
        if 1 <= nx <= max_x and 1 <= ny <= max_y: return nx, ny, d
        return x, y, d

def expert_action(x, y, d, xg, yg):
    dx, dy = xg-x, yg-y
    if dx==0 and dy==0: return 2
    if dx==0: desired = 1 if dy>0 else 3
    elif dy==0: desired = 0 if dx>0 else 2
    elif abs(dx)>=abs(dy): desired = 0 if dx>0 else 2
    else: desired = 1 if dy>0 else 3
    if d==desired: return 2
    diff = (desired-d)%4
    if diff==1: return 1
    elif diff==3: return 0
    else: return 1

def collect_episode(min_dist=1, max_dist=14):
    """Collect one episode, return list of (sc, sg, pc, pg, label) tuples."""
    x_start, y_start, d_start = np.random.randint(1,15), np.random.randint(1,15), np.random.randint(0,4)
    while True:
        x_goal, y_goal, d_goal = np.random.randint(1,15), np.random.randint(1,15), np.random.randint(0,4)
        dist = abs(x_start-x_goal) + abs(y_start-y_goal)
        if min_dist <= dist <= max_dist:
            break

    set_state(x_goal, y_goal, d_goal)
    z_goal = encode(env.render())
    sg = build_state(z_goal, d_goal).detach()
    pg = build_pos(x_goal, y_goal).detach()

    x, y, d = x_start, y_start, d_start
    samples = []
    for _ in range(50):
        set_state(x, y, d)
        z = encode(env.render())
        sc = build_state(z, d).detach()
        pc = build_pos(x, y).detach()
        expert = expert_action(x, y, d, x_goal, y_goal)
        samples.append((sc, sg, pc, pg, expert))
        x, y, d = manual_step(x, y, d, expert)
        if (x, y) == (x_goal, y_goal):
            break
    return samples

# ── Phase 1: Fill replay buffer ──────────────────────────────────────
print("Filling replay buffer with 2000 episodes...")
replay = []
for ep in range(2000):
    replay.extend(collect_episode(min_dist=1, max_dist=14))
print(f"Buffer size: {len(replay)} samples")

labels = [s[4] for s in replay]
print(f"Label dist: L={labels.count(0)/len(labels)*100:.1f}%  R={labels.count(1)/len(labels)*100:.1f}%  F={labels.count(2)/len(labels)*100:.1f}%")

# Stack into tensors for fast batch sampling
sc_all = torch.cat([s[0] for s in replay], dim=0)   # (N, 260)
sg_all = torch.cat([s[1] for s in replay], dim=0)
pc_all = torch.cat([s[2] for s in replay], dim=0)
pg_all = torch.cat([s[3] for s in replay], dim=0)
lb_all = torch.tensor([s[4] for s in replay], dtype=torch.long).to(device)
N = len(replay)

# ── Phase 2: Train on batches from buffer ────────────────────────────
print("\nTraining on replay buffer (batch size=256)...")
batch_size = 256
for step in range(1000):
    idxs = np.random.randint(0, N, size=batch_size)
    logits = controller(sc_all[idxs], sg_all[idxs], pc_all[idxs], pg_all[idxs])
    loss = F.cross_entropy(logits, lb_all[idxs])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step+1) % 100 == 0:
        # Evaluate on full buffer
        with torch.no_grad():
            all_logits = controller(sc_all, sg_all, pc_all, pg_all)
            acc = (all_logits.argmax(dim=-1) == lb_all).float().mean().item()
            per_l = (all_logits.argmax(dim=-1)[lb_all==0] == 0).float().mean().item()
            per_r = (all_logits.argmax(dim=-1)[lb_all==1] == 1).float().mean().item()
            per_f = (all_logits.argmax(dim=-1)[lb_all==2] == 2).float().mean().item()
        print(f"  Step {step+1}: loss={loss.item():.4f}  acc={acc*100:.1f}%  "
              f"L={per_l*100:.0f}%  R={per_r*100:.0f}%  F={per_f*100:.0f}%")