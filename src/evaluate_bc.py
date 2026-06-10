import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder
from controller_bc import Controller
from data_collector_v2 import ReplayBuffer

# ── Load everything ───────────────────────────────────────────────────
buffer = ReplayBuffer(capacity=200_000)
buffer.load("data/replay_buffer_phase1.pkl")

ckpt    = torch.load(
    "checkpoints/jepa_phase1_final.pt",
    map_location="cpu", weights_only=False
)
encoder = Encoder(256)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

cont_ckpt   = torch.load(
    "checkpoints/controller_bc_final.pt",
    map_location="cpu", weights_only=False
)
controller = Controller(state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2)
controller.load_state_dict(cont_ckpt["controller"])
controller.eval()

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

ACTION_NAMES = {0: "Left", 1: "Right", 2: "Forward"}
DIR_NAMES    = {0: "East", 1: "South", 2: "West", 3: "North"}

def preprocess(frame):
    img = Image.fromarray(frame).resize((64,64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr.transpose(2,0,1)).unsqueeze(0)

def build_state(z, direction):
    dir_oh = F.one_hot(torch.tensor([direction]), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def set_state(x, y, d):
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d

# ── Evaluate on buffer transitions ───────────────────────────────────
print("Evaluating BC controller on one-step goals from buffer...")
print("=" * 60)

n_test   = 1000
idxs     = np.random.randint(0, buffer.size, size=n_test)

correct       = 0
per_action    = {0: {"correct": 0, "total": 0},
                 1: {"correct": 0, "total": 0},
                 2: {"correct": 0, "total": 0}}

env.reset()

for idx in idxs:
    # Get buffer entry
    x, y       = buffer.positions[idx].astype(int)
    direction  = int(buffer.directions[idx])
    next_dir   = int(buffer.next_directions[idx])
    action     = int(buffer.actions[idx])

    # Encode current state
    set_state(x, y, direction)
    frame_t    = env.render()
    z_current  = encoder(preprocess(frame_t))
    state_curr = build_state(z_current, direction)

    # Execute action to get real next state
    set_state(x, y, direction)
    env.step(action)
    nx, ny    = env.unwrapped.agent_pos
    next_dir_real = env.unwrapped.agent_dir
    frame_t1  = env.render()

    # Encode goal state (the actual next state)
    z_goal     = encoder(preprocess(frame_t1))
    state_goal = build_state(z_goal, next_dir_real)

    # Build position tensors
    pos_curr = torch.tensor(
        [[x / 14.0, y / 14.0]], dtype=torch.float32
    )
    pos_goal_t = torch.tensor(
        [[nx / 14.0, ny / 14.0]], dtype=torch.float32
    )

    # Controller predicts action
    with torch.no_grad():
        action_probs = controller(state_curr, state_goal, pos_curr, pos_goal_t)
        predicted    = action_probs.argmax(dim=-1).item()

    is_correct = predicted == action
    correct   += int(is_correct)
    per_action[action]["total"]   += 1
    per_action[action]["correct"] += int(is_correct)

env.close()

# ── Results ───────────────────────────────────────────────────────────
print(f"\nOverall accuracy: {correct}/{n_test} = {correct/n_test*100:.1f}%")
print()
for a, name in ACTION_NAMES.items():
    t = per_action[a]["total"]
    c = per_action[a]["correct"]
    if t > 0:
        print(f"  {name:7s}: {c}/{t} = {c/t*100:.1f}%")