"""
Fine-tune JEPA encoder on FourRooms with hierarchical triplet loss.

Goal: Teach z-space a two-level hierarchy:
  Level 1 (room):     different rooms → far apart in z
  Level 2 (position): within a room, close positions → closer in z than far positions

This preserves position structure within rooms while making rooms distinct.

Loss:
  l_room = triplet(anchor, far_same_room, diff_room, margin=4.0)
  l_pos  = triplet(anchor, close_same_room, far_same_room, margin=1.0)
  total  = vicreg + λ_room * l_room + λ_pos * l_pos + λ_pred * l_pred

Usage:
    python src/finetune_room_aware_v2.py

Reads:  checkpoints/jepa_fourrooms_final.pt
        data/replay_buffer_fourrooms_with_nextpos.pkl
Writes: checkpoints/jepa_roomaware_v2_final.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import pickle
import gymnasium as gym
import minigrid
from pathlib import Path
from PIL import Image
import os

from encoder_v2 import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
Path("checkpoints").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = "checkpoints/jepa_fourrooms_final.pt"
CHECKPOINT_OUT = "checkpoints/jepa_roomaware_v2_final.pt"
BUFFER_PATH    = "data/replay_buffer_fourrooms_with_nextpos.pkl"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

N_FINETUNE_STEPS = 10_000
BATCH_SIZE       = 256
LR               = 3e-5
EMA_MOMENTUM     = 0.99

# Hierarchical triplet hyperparameters
CLOSE_THRESHOLD  = 3      # cells — "nearby" within same room
FAR_THRESHOLD    = 6      # cells — "far" within same room

MARGIN_ROOM      = 4.0    # gap between far_same_room and diff_room
MARGIN_POS       = 1.0    # gap between close_same_room and far_same_room

LAMBDA_ROOM      = 0.05   # gentle — just enough to separate rooms
LAMBDA_POS       = 0.05   # gentle — just enough to preserve position
LAMBDA_VICREG    = 1.0    # main loss — prevents collapse
LAMBDA_PRED      = 0.1    # keeps dynamics accurate

LOG_EVERY        = 200
SAVE_EVERY       = 5_000


# ── Room helper ────────────────────────────────────────────────────────────────
def get_room(x, y):
    """Quadrant-based room assignment. No special casing for walls."""
    if x <= 9 and y <= 9:   return 0
    elif x >= 9 and y <= 9: return 1
    elif x <= 9 and y >= 9: return 2
    else:                    return 3


def euclidean_dist(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


# ── Buffer ─────────────────────────────────────────────────────────────────────
class ReplayBufferWithNextPos:
    def __init__(self, capacity=200_000, obs_shape=(3, 64, 64)):
        self.capacity        = capacity
        self.obs_shape       = obs_shape
        self.ptr             = 0
        self.size            = 0
        self.obs             = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs        = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions         = np.zeros((capacity,),            dtype=np.int64)
        self.positions       = np.zeros((capacity, 2),          dtype=np.float32)
        self.next_positions  = np.zeros((capacity, 2),          dtype=np.float32)
        self.directions      = np.zeros((capacity,),            dtype=np.int64)
        self.next_directions = np.zeros((capacity,),            dtype=np.int64)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs             = torch.tensor(self.obs[idxs],            dtype=torch.float32) / 255.0,
            next_obs        = torch.tensor(self.next_obs[idxs],       dtype=torch.float32) / 255.0,
            actions         = torch.tensor(self.actions[idxs],        dtype=torch.long),
            positions       = torch.tensor(self.positions[idxs],      dtype=torch.float32),
            next_positions  = torch.tensor(self.next_positions[idxs], dtype=torch.float32),
            directions      = torch.tensor(self.directions[idxs],     dtype=torch.long),
            next_directions = torch.tensor(self.next_directions[idxs],dtype=torch.long),
        )

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"Buffer loaded: {path} ({self.size} transitions)")

    def __len__(self):
        return self.size


# ── Room + position index ──────────────────────────────────────────────────────
def build_index(buffer):
    """
    Build spatial index for fast hierarchical triplet sampling.
    
    room_index[r] = list of (buffer_idx, x, y)
    Allows us to find:
      - same room pairs
      - close pairs (dist < CLOSE_THRESHOLD)  
      - far pairs   (dist > FAR_THRESHOLD)
    """
    room_index = {0: [], 1: [], 2: [], 3: []}
    for i in range(buffer.size):
        x, y = float(buffer.positions[i, 0]), float(buffer.positions[i, 1])
        r = get_room(x, y)
        room_index[r].append((i, x, y))

    print("Room index built:")
    for r, entries in room_index.items():
        print(f"  Room {r}: {len(entries)} samples")
    return room_index


def sample_hierarchical_triplets(buffer, room_index, batch_size):
    """
    Sample hierarchical triplets:
      anchor:     any position in room R
      close_pos:  same room R, euclidean dist < CLOSE_THRESHOLD
      far_pos:    same room R, euclidean dist > FAR_THRESHOLD
      neg:        different room

    Returns obs tensors for all four.
    """
    anchors    = []
    close_poss = []
    far_poss   = []
    negs       = []

    attempts = 0
    max_attempts = batch_size * 20

    while len(anchors) < batch_size and attempts < max_attempts:
        attempts += 1

        # Pick anchor room
        anchor_room = random.choice([0, 1, 2, 3])
        room_entries = room_index[anchor_room]
        if len(room_entries) < 10:
            continue

        # Anchor
        anchor_entry = random.choice(room_entries)
        anchor_idx, ax, ay = anchor_entry

        # Find close and far candidates from same room
        close_candidates = []
        far_candidates   = []
        # Sample a subset to avoid O(n²) search
        candidates = random.sample(room_entries, min(50, len(room_entries)))
        for idx, cx, cy in candidates:
            if idx == anchor_idx:
                continue
            d = euclidean_dist((ax, ay), (cx, cy))
            if d < CLOSE_THRESHOLD:
                close_candidates.append(idx)
            elif d > FAR_THRESHOLD:
                far_candidates.append(idx)

        if not close_candidates or not far_candidates:
            continue

        close_idx = random.choice(close_candidates)
        far_idx   = random.choice(far_candidates)

        # Negative: different room
        other_rooms = [r for r in range(4) if r != anchor_room]
        neg_room    = random.choice(other_rooms)
        if not room_index[neg_room]:
            continue
        neg_idx = random.choice(room_index[neg_room])[0]

        anchors.append(   torch.tensor(buffer.obs[anchor_idx], dtype=torch.float32) / 255.0)
        close_poss.append(torch.tensor(buffer.obs[close_idx],  dtype=torch.float32) / 255.0)
        far_poss.append(  torch.tensor(buffer.obs[far_idx],    dtype=torch.float32) / 255.0)
        negs.append(      torch.tensor(buffer.obs[neg_idx],    dtype=torch.float32) / 255.0)

    if len(anchors) < batch_size:
        print(f"  Warning: got {len(anchors)}/{batch_size} triplets")

    return (
        torch.stack(anchors).to(DEVICE),
        torch.stack(close_poss).to(DEVICE),
        torch.stack(far_poss).to(DEVICE),
        torch.stack(negs).to(DEVICE),
    )


# ── Losses ─────────────────────────────────────────────────────────────────────
def triplet_loss(z_a, z_pos, z_neg, margin):
    """L = max(0, ||z_a - z_pos||² - ||z_a - z_neg||² + margin)"""
    d_pos = (z_a - z_pos).pow(2).sum(dim=1)
    d_neg = (z_a - z_neg).pow(2).sum(dim=1)
    loss  = F.relu(d_pos - d_neg + margin)
    return loss.mean(), d_pos.mean().item(), d_neg.mean().item()


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2):
    B, D = z1.shape
    sim  = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var  = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean()) / 2
    z1c  = z1 - z1.mean(dim=0)
    z2c  = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    cov  = (off_diagonal(cov1).pow(2).sum() / D +
            off_diagonal(cov2).pow(2).sum() / D)
    return 25.0 * sim + 25.0 * var + 1.0 * cov


# ── Room separation metric ─────────────────────────────────────────────────────
@torch.no_grad()
def measure_separation(encoder, buffer, room_index, n_samples=300):
    encoder.eval()
    samples, rooms = [], []
    for r in range(4):
        idxs = random.sample(room_index[r], min(n_samples // 4, len(room_index[r])))
        for idx, x, y in idxs:
            obs = torch.tensor(buffer.obs[idx], dtype=torch.float32).unsqueeze(0) / 255.0
            samples.append(obs)
            rooms.append(r)

    zs = []
    for obs in samples:
        zs.append(encoder(obs.to(DEVICE)).cpu())
    zs = torch.cat(zs, dim=0)

    within, between = [], []
    for i in range(len(zs)):
        for j in range(i + 1, min(i + 30, len(zs))):
            d = (zs[i] - zs[j]).pow(2).sum().item()
            if rooms[i] == rooms[j]: within.append(d)
            else:                    between.append(d)

    w     = np.mean(within)
    b     = np.mean(between)
    ratio = b / w if w > 0 else float("inf")
    print(f"  Within-room:  {w:.2f}")
    print(f"  Between-room: {b:.2f}")
    print(f"  Ratio: {ratio:.2f}x  (want 2-10x, not 1779x)")
    encoder.train()
    return ratio


# ── Main ───────────────────────────────────────────────────────────────────────
def finetune():
    print(f"Device: {DEVICE}\n")

    # ── Load buffer ────────────────────────────────────────────────────────────
    buffer = ReplayBufferWithNextPos(capacity=300_000)
    buffer.load(BUFFER_PATH)
    room_index = build_index(buffer)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\nLoading: {CHECKPOINT_IN}")
    ckpt      = torch.load(CHECKPOINT_IN, map_location=DEVICE, weights_only=False)
    encoder   = Encoder(latent_dim=256).to(DEVICE)
    predictor = Predictor(latent_dim=256, n_actions=3).to(DEVICE)
    target    = copy.deepcopy(encoder).to(DEVICE)

    encoder.load_state_dict(ckpt["online_encoder"])
    target.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    for p in target.parameters():
        p.requires_grad = False
    print("Checkpoint loaded.")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )

    # ── Baseline ───────────────────────────────────────────────────────────────
    print("\n── Separation BEFORE fine-tuning ──")
    measure_separation(encoder, buffer, room_index)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nFine-tuning for {N_FINETUNE_STEPS} steps...")
    loss_log = []

    for step in range(N_FINETUNE_STEPS):
        encoder.train()
        predictor.train()

        # Sample hierarchical triplets
        obs_a, obs_close, obs_far, obs_neg = sample_hierarchical_triplets(
            buffer, room_index, BATCH_SIZE
        )

        z_a     = encoder(obs_a)
        z_close = encoder(obs_close)
        z_far   = encoder(obs_far)
        z_neg   = encoder(obs_neg)

        # Loss 1 — room separation: far_same_room vs diff_room
        # Enforces: ||z_a - z_far|| < ||z_a - z_neg|| + margin_room
        l_room, d_far, d_neg = triplet_loss(z_a, z_far, z_neg, MARGIN_ROOM)

        # Loss 2 — position preservation: close vs far within same room
        # Enforces: ||z_a - z_close|| < ||z_a - z_far|| + margin_pos
        l_pos, d_close, _ = triplet_loss(z_a, z_close, z_far, MARGIN_POS)

        # VICReg on anchor + close_pos (similar observations, prevent collapse)
        l_vic = vicreg_loss(z_a, z_close)

        # Predictor loss
        batch   = buffer.sample(BATCH_SIZE)
        obs_t   = batch["obs"].to(DEVICE)
        obs_t1  = batch["next_obs"].to(DEVICE)
        actions = batch["actions"].to(DEVICE)
        dirs    = batch["directions"].to(DEVICE)
        dirs_t1 = batch["next_directions"].to(DEVICE)

        z_t = encoder(obs_t)
        with torch.no_grad():
            z_t1 = target(obs_t1)

        dir_oh  = F.one_hot(dirs,    num_classes=4).float()
        dir1_oh = F.one_hot(dirs_t1, num_classes=4).float()
        state_t  = torch.cat([z_t,  dir_oh],  dim=-1)
        state_t1 = torch.cat([z_t1, dir1_oh], dim=-1)
        pred_t1  = predictor(state_t, actions)
        l_pred   = F.mse_loss(pred_t1, state_t1)

        # Total
        total = (LAMBDA_VICREG * l_vic +
                 LAMBDA_ROOM   * l_room +
                 LAMBDA_POS    * l_pos +
                 LAMBDA_PRED   * l_pred)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # EMA target update
        with torch.no_grad():
            for op, tp in zip(encoder.parameters(), target.parameters()):
                tp.data = EMA_MOMENTUM * tp.data + (1 - EMA_MOMENTUM) * op.data

        if step % LOG_EVERY == 0:
            entry = {
                "step":   step,
                "total":  total.item(),
                "l_room": l_room.item(),
                "l_pos":  l_pos.item(),
                "l_vic":  l_vic.item(),
                "l_pred": l_pred.item(),
                "d_close": d_close,
                "d_far":   d_far,
                "d_neg":   d_neg,
            }
            loss_log.append(entry)
            print(f"Step {step:5d} | "
                  f"room={l_room.item():.4f} pos={l_pos.item():.4f} | "
                  f"d_close={d_close:.2f} d_far={d_far:.2f} d_neg={d_neg:.2f} | "
                  f"vic={l_vic.item():.3f} pred={l_pred.item():.4f}")

        if (step + 1) % SAVE_EVERY == 0:
            print(f"\n── Separation at step {step+1} ──")
            measure_separation(encoder, buffer, room_index)

    # ── Final ──────────────────────────────────────────────────────────────────
    print("\n── Separation AFTER fine-tuning ──")
    encoder.eval()
    measure_separation(encoder, buffer, room_index)

    torch.save({
        "online_encoder": encoder.state_dict(),
        "target_encoder": target.state_dict(),
        "predictor":      predictor.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "step":           ckpt["step"] + N_FINETUNE_STEPS,
        "loss_log":       loss_log,
    }, CHECKPOINT_OUT)
    print(f"\nSaved: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    finetune()