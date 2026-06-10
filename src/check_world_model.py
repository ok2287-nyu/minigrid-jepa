import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
import gymnasium as gym
import minigrid
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder, Predictor

# ── Load model ────────────────────────────────────────────────────────
ckpt = torch.load(
    "checkpoints/jepa_phase1_final.pt",
    map_location="cpu",
    weights_only=False
)
encoder   = Encoder(256)
predictor = Predictor(256, n_actions=3)
encoder.load_state_dict(ckpt["online_encoder"])
predictor.load_state_dict(ckpt["predictor"])
encoder.eval()
predictor.eval()

# ── Environment ───────────────────────────────────────────────────────
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

DIR_NAMES    = {0: "East", 1: "South", 2: "West", 3: "North"}
ACTION_NAMES = {0: "Left", 1: "Right", 2: "Forward"}

def encode_state(x, y, d):
    """Encode a specific (x, y, direction) state."""
    env.reset()
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img   = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32) / 255.0
    t     = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        z = encoder(t)
    return z

def build_state(z, d):
    dir_oh = F.one_hot(torch.tensor([d]), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def cosine_dist(a, b):
    return (1 - F.cosine_similarity(
        F.normalize(a, dim=-1),
        F.normalize(b, dim=-1),
        dim=-1
    )).item()

def predict_next(x, y, d, action):
    """Predict next state from (x,y,d) after action."""
    z     = encode_state(x, y, d)
    state = build_state(z, d)
    with torch.no_grad():
        action_t   = torch.tensor([action])
        state_next = predictor(state, action_t)
    return state_next

def actual_next(x, y, d, action):
    """Get actual next state after taking action."""
    env.reset()
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    env.step(action)
    nx, ny = env.unwrapped.agent_pos
    nd     = env.unwrapped.agent_dir
    z_next = encode_state(nx, ny, nd)
    return build_state(z_next, nd), nx, ny, nd

print("=" * 60)
print("WORLD MODEL TEST")
print("=" * 60)
print()

# ── Test 1: Forward action changes position correctly ─────────────────
print("TEST 1: Forward action — does position change correctly?")
print("-" * 60)

test_cases = [
    (7, 7, 0),   # East
    (7, 7, 1),   # South
    (7, 7, 2),   # West
    (7, 7, 3),   # North
]

for x, y, d in test_cases:
    # Predicted next state
    state_pred = predict_next(x, y, d, action=2)  # forward

    # Actual next state
    state_actual, nx, ny, nd = actual_next(x, y, d, action=2)

    # Wrong position (perpendicular — should be far)
    wrong_x = x + (1 if d == 1 else -1 if d == 3 else 0)
    wrong_y = y + (1 if d == 0 else -1 if d == 2 else 0)
    wrong_x = max(1, min(14, wrong_x))
    wrong_y = max(1, min(14, wrong_y))
    z_wrong  = encode_state(wrong_x, wrong_y, d)
    state_wrong = build_state(z_wrong, d)

    dist_correct = cosine_dist(state_pred, state_actual)
    dist_wrong   = cosine_dist(state_pred, state_wrong)
    dist_same    = cosine_dist(state_pred, build_state(encode_state(x,y,d), d))

    correct = "✓" if dist_correct < dist_wrong else "✗"
    print(f"{correct} ({x},{y}) facing {DIR_NAMES[d]:5s} + Forward")
    print(f"     → actual next: ({nx},{ny}) facing {DIR_NAMES[nd]}")
    print(f"     dist to CORRECT next state: {dist_correct:.4f}")
    print(f"     dist to WRONG   next state: {dist_wrong:.4f}")
    print(f"     dist to SAME    state:      {dist_same:.4f}")
    print()

# ── Test 2: Turn actions keep position same ───────────────────────────
print("TEST 2: Turn actions — does position stay same?")
print("-" * 60)

for action, aname in [(0, "Left"), (1, "Right")]:
    state_pred = predict_next(7, 7, 0, action=action)
    state_actual, nx, ny, nd = actual_next(7, 7, 0, action=action)

    # Encode same position different direction
    z_same_pos = encode_state(nx, ny, nd)
    state_same_pos = build_state(z_same_pos, nd)

    # Encode different position same direction
    z_diff_pos = encode_state(8, 7, nd)
    state_diff_pos = build_state(z_diff_pos, nd)

    dist_correct = cosine_dist(state_pred, state_same_pos)
    dist_wrong   = cosine_dist(state_pred, state_diff_pos)

    correct = "✓" if dist_correct < dist_wrong else "✗"
    print(f"{correct} (7,7) facing East + {aname}")
    print(f"     → actual next: ({nx},{ny}) facing {DIR_NAMES[nd]}")
    print(f"     dist to CORRECT (same pos, new dir): {dist_correct:.4f}")
    print(f"     dist to WRONG   (diff pos, new dir): {dist_wrong:.4f}")
    print()

# ── Test 3: Forward into wall keeps position same ─────────────────────
print("TEST 3: Forward into wall — does position stay same?")
print("-" * 60)

# Place agent next to wall, facing wall
wall_cases = [
    (14, 7, 0),   # East wall
    (7,  1, 3),   # North wall
    (1,  7, 2),   # West wall
    (7, 14, 1),   # South wall
]

for x, y, d in wall_cases:
    state_pred = predict_next(x, y, d, action=2)  # forward into wall
    state_actual, nx, ny, nd = actual_next(x, y, d, action=2)

    # Actual next should be same position (wall blocked)
    z_same = encode_state(x, y, d)
    state_same = build_state(z_same, d)

    dist_to_same = cosine_dist(state_pred, state_same)
    moved = (nx != x or ny != y)

    print(f"  ({x},{y}) facing {DIR_NAMES[d]:5s} + Forward into wall")
    print(f"     → actual next: ({nx},{ny}) {'MOVED (boundary issue)' if moved else 'stayed (correct)'}")
    print(f"     predicted dist to same state: {dist_to_same:.4f}")
    print(f"     (lower = predictor thinks agent stayed)")
    print()

env.close()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("Test 1: predictor dist to correct < dist to wrong → world model knows where forward goes")
print("Test 2: predictor dist to same pos < dist to diff pos → world model knows turns don't move")
print("Test 3: low dist to same state → world model knows walls block movement")