"""
Fine-tune JEPA encoder on FourRooms with triplet contrastive loss.

Goal: Teach the encoder that crossing a doorway = significant context switch.
      Same-room z vectors pulled together, cross-room z vectors pushed apart.

Signal: purely self-supervised
  - crossed_doorway = (get_room(x_t) != get_room(x_t1))
  - No room ID labels, no map-specific knowledge
  - Generalizes to any layout where rooms exist

Usage:
    python src/finetune_room_aware.py

Reads:  checkpoints/jepa_fourrooms_final.pt
        data/replay_buffer_fourrooms_with_nextpos.pkl  (re-collected)
Writes: checkpoints/jepa_roomaware_final.pt
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
from minigrid.core.constants import DIR_TO_VEC
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
CHECKPOINT_OUT = "checkpoints/jepa_roomaware_final.pt"
BUFFER_PATH    = "data/replay_buffer_fourrooms_with_nextpos.pkl"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

N_COLLECT_RESETS = 30       # env resets for data collection
N_FINETUNE_STEPS = 10_000   # gradient steps
BATCH_SIZE       = 256
LR               = 3e-5     # lower LR for fine-tuning (don't destroy existing knowledge)
EMA_MOMENTUM     = 0.99
TRIPLET_MARGIN   = 0.5     # minimum gap between same-room and diff-room distances
LAMBDA_TRIPLET   = 0.1      # weight of triplet loss
LAMBDA_VICREG    = 1.0      # weight of VICReg (collapse prevention)
LAMBDA_PRED      = 0.1      # keep predictor accurate
LOG_EVERY        = 200
SAVE_EVERY       = 5_000


# ── Room helper ────────────────────────────────────────────────────────────────
def get_room(x, y):
    """
    Simple quadrant assignment. Wall cells will never be 
    in the buffer as starting positions since agent can't 
    stand in a wall — only in walkable cells.
    """
    if x <= 9 and y <= 9:
        return 0   # top-left
    elif x >= 9 and y <= 9:
        return 1   # top-right
    elif x <= 9 and y >= 9:
        return 2   # bottom-left
    else:
        return 3   # bottom-right

# ── Extended replay buffer (adds next_positions) ───────────────────────────────
class ReplayBufferWithNextPos:
    """
    Same as ReplayBuffer but also stores next_positions.
    Needed to detect doorway crossings from stored data.
    """
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

    def add(self, obs, action, next_obs, pos, next_pos, direction, next_direction):
        self.obs[self.ptr]             = obs
        self.next_obs[self.ptr]        = next_obs
        self.actions[self.ptr]         = action
        self.positions[self.ptr]       = pos
        self.next_positions[self.ptr]  = next_pos
        self.directions[self.ptr]      = direction
        self.next_directions[self.ptr] = next_direction
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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

    def get_room_index(self):
        """
        Build index: room_id → list of buffer indices.
        Used for fast triplet sampling.
        """
        room_index = {0: [], 1: [], 2: [], 3: []}
        for i in range(self.size):
            x, y = int(self.positions[i, 0]), int(self.positions[i, 1])
            r = get_room(x, y)
            if r != -1:   # skip doorway cells
                room_index[r].append(i)
        total = sum(len(v) for v in room_index.values())
        print(f"Room index built: {total} valid samples")
        for r, idxs in room_index.items():
            print(f"  Room {r}: {len(idxs)} samples")
        return room_index

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"Buffer saved: {path} ({self.size} transitions)")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"Buffer loaded: {path} ({self.size} transitions)")

    def __len__(self):
        return self.size


# ── Data collection ────────────────────────────────────────────────────────────
class FourRoomsCollector:
    """
    Systematic data collection for FourRooms.
    Stores next_positions so we can detect doorway crossings.
    """
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.env      = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        print(f"FourRooms grid: {self.width} x {self.height}")

    def preprocess(self, frame):
        img = Image.fromarray(frame).resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(img, dtype=np.uint8).transpose(2, 0, 1)   # CHW uint8

    def set_agent_state(self, x, y, direction):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def collect(self, buffer, n_resets=30):
        crossings    = 0
        same_room    = 0
        wall_bumps   = 0

        # walkable cells: skip boundary walls (col 0, col 18, row 0, row 18)
        # also skip the wall columns/rows (col 9, row 9) — those are walls
        # Fixed — include col 9 and row 9 (doorway cells are walkable)
        x_range = list(range(1, self.width  - 1))
        y_range = list(range(1, self.height - 1))

        print(f"Collecting {n_resets} resets x {len(x_range)*len(y_range)*4*3} "
              f"transitions each...")

        for reset_idx in range(n_resets):
            self.env.reset()

            for x in x_range:
                for y in y_range:
                    for direction in range(4):
                        self.set_agent_state(x, y, direction)
                        frame_t = self.env.render()
                        obs_t   = self.preprocess(frame_t)
                        pos_t   = np.array([x, y], dtype=np.float32)

                        for action in range(3):  # left, right, forward
                            self.set_agent_state(x, y, direction)
                            self.env.step(action)
                            # if action == 2:  # forward only
                            #     print(f"  ({x},{y}) dir={direction} → ({int(x1)},{int(y1)}) "
                            #         f"room {get_room(x,y)} → {get_room(int(x1),int(y1))}")
                            x1, y1   = self.env.unwrapped.agent_pos
                            dir1     = self.env.unwrapped.agent_dir
                            frame_t1 = self.env.render()
                            obs_t1   = self.preprocess(frame_t1)
                            pos_t1   = np.array([x1, y1], dtype=np.float32)

                            # Compute next direction from action
                            if action == 0:   next_dir = (direction - 1) % 4
                            elif action == 1: next_dir = (direction + 1) % 4
                            else:             next_dir = direction

                            buffer.add(obs_t, action, obs_t1,
                                       pos_t, pos_t1,
                                       direction, next_dir)

                            r_t  = get_room(x,        y)
                            r_t1 = get_room(int(x1), int(y1))
                            if r_t != r_t1:
                                crossings += 1
                            else:
                                same_room += 1

            if (reset_idx + 1) % 5 == 0:
                print(f"  Reset {reset_idx+1}/{n_resets} | "
                      f"buffer={len(buffer)} | "
                      f"crossings={crossings} | "
                      f"same_room={same_room}")

        print(f"\nCollection done: {len(buffer)} transitions")
        print(f"  Doorway crossings: {crossings}")
        print(f"  Same-room steps:   {same_room}")
        print(f"  Wall bumps:        {wall_bumps}")
        return buffer


# ── Losses ─────────────────────────────────────────────────────────────────────
def triplet_loss(z_anchor, z_pos, z_neg, margin):
    """
    L = max(0, ||z_a - z_p||² - ||z_a - z_n||² + margin)
    Pulls same-room together, pushes diff-room apart.
    """
    dist_pos = (z_anchor - z_pos).pow(2).sum(dim=1)
    dist_neg = (z_anchor - z_neg).pow(2).sum(dim=1)
    loss     = F.relu(dist_pos - dist_neg + margin)
    return loss.mean(), dist_pos.mean().item(), dist_neg.mean().item()


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2):
    """Prevents encoder collapse while fine-tuning."""
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


# ── Triplet sampling ───────────────────────────────────────────────────────────
def sample_triplets(buffer, room_index, batch_size):
    """
    Sample (anchor, positive, negative) triplets.
    Anchor & positive: same room, different steps.
    Negative: different room from anchor.
    """
    anchors, positives, negatives = [], [], []
    attempts = 0

    while len(anchors) < batch_size and attempts < batch_size * 10:
        attempts += 1

        # Pick anchor room
        anchor_room = random.choice([0, 1, 2, 3])
        if len(room_index[anchor_room]) < 2:
            continue

        # Anchor
        anchor_idx = random.choice(room_index[anchor_room])

        # Positive: same room, different index
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = random.choice(room_index[anchor_room])

        # Negative: different room
        other_rooms = [r for r in range(4) if r != anchor_room]
        neg_room    = random.choice(other_rooms)
        if len(room_index[neg_room]) == 0:
            continue
        neg_idx = random.choice(room_index[neg_room])

        anchors.append(torch.tensor(buffer.obs[anchor_idx], dtype=torch.float32) / 255.0)
        positives.append(torch.tensor(buffer.obs[pos_idx],  dtype=torch.float32) / 255.0)
        negatives.append(torch.tensor(buffer.obs[neg_idx],  dtype=torch.float32) / 255.0)

    if len(anchors) < batch_size:
        print(f"Warning: only got {len(anchors)} triplets from {attempts} attempts")

    return (
        torch.stack(anchors).to(DEVICE),
        torch.stack(positives).to(DEVICE),
        torch.stack(negatives).to(DEVICE),
    )


# ── Room separation metric ─────────────────────────────────────────────────────
@torch.no_grad()
def measure_room_separation(encoder, buffer, room_index, n_samples=200):
    encoder.eval()

    # Sample from each room
    samples = []
    for r in range(4):
        idxs = random.sample(room_index[r], min(n_samples // 4, len(room_index[r])))
        for i in idxs:
            obs = torch.tensor(buffer.obs[i], dtype=torch.float32).unsqueeze(0) / 255.0
            samples.append((obs, r))

    # Encode
    zs    = []
    rooms = []
    for obs, r in samples:
        z = encoder(obs.to(DEVICE))
        zs.append(z.cpu())
        rooms.append(r)
    zs = torch.cat(zs, dim=0)

    # Compute within vs between distances
    within, between = [], []
    for i in range(len(zs)):
        for j in range(i + 1, min(i + 30, len(zs))):
            d = (zs[i] - zs[j]).pow(2).sum().item()
            if rooms[i] == rooms[j]:
                within.append(d)
            else:
                between.append(d)

    w = np.mean(within)
    b = np.mean(between)
    ratio = b / w if w > 0 else float("inf")

    print(f"  Within-room distance:  {w:.2f}")
    print(f"  Between-room distance: {b:.2f}")
    print(f"  Ratio: {ratio:.2f}x  (baseline ~1.15x, want >2.0x)")
    if ratio > 2.0:
        print(f"  ✓ Room structure learned!")
    else:
        print(f"  ✗ Still weak — may need more steps or higher λ_triplet")

    encoder.train()
    return ratio


# ── Main ───────────────────────────────────────────────────────────────────────
def finetune():
    print(f"Device: {DEVICE}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    buffer = ReplayBufferWithNextPos(capacity=300_000)
    if Path(BUFFER_PATH).exists():
        buffer.load(BUFFER_PATH)
    else:
        collector = FourRoomsCollector(img_size=64)
        collector.collect(buffer, n_resets=N_COLLECT_RESETS)
        buffer.save(BUFFER_PATH)

    room_index = buffer.get_room_index()

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

    # ── Optimizer ──────────────────────────────────────────────────────────────
    params    = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    # ── Baseline measurement ───────────────────────────────────────────────────
    print("\n── Room separation BEFORE fine-tuning ──")
    measure_room_separation(encoder, buffer, room_index)

    # ── Fine-tuning loop ───────────────────────────────────────────────────────
    print(f"\nFine-tuning for {N_FINETUNE_STEPS} steps...")
    loss_log = []

    for step in range(N_FINETUNE_STEPS):
        encoder.train()
        predictor.train()

        # ── Triplet loss: room structure ───────────────────────────────────────
        obs_a, obs_p, obs_n = sample_triplets(buffer, room_index, BATCH_SIZE)
        z_a = encoder(obs_a)
        z_p = encoder(obs_p)
        z_n = encoder(obs_n)
        l_trip, d_pos, d_neg = triplet_loss(z_a, z_p, z_n, TRIPLET_MARGIN)

        # ── VICReg: prevent collapse ───────────────────────────────────────────
        l_vic = vicreg_loss(z_a, z_p)

        # ── Predictor: keep dynamics accurate ─────────────────────────────────
        batch   = buffer.sample(BATCH_SIZE)
        obs_t   = batch["obs"].to(DEVICE)
        obs_t1  = batch["next_obs"].to(DEVICE)
        actions = batch["actions"].to(DEVICE)
        dirs    = batch["directions"].to(DEVICE)
        dirs_t1 = batch["next_directions"].to(DEVICE)

        z_t  = encoder(obs_t)
        with torch.no_grad():
            z_t1 = target(obs_t1)

        dir_oh   = F.one_hot(dirs,    num_classes=4).float()
        dir1_oh  = F.one_hot(dirs_t1, num_classes=4).float()
        state_t  = torch.cat([z_t,  dir_oh],  dim=-1)
        state_t1 = torch.cat([z_t1, dir1_oh], dim=-1)
        pred_t1  = predictor(state_t, actions)
        l_pred   = F.mse_loss(pred_t1, state_t1)

        # ── Total loss ─────────────────────────────────────────────────────────
        total = LAMBDA_VICREG * l_vic + LAMBDA_TRIPLET * l_trip + LAMBDA_PRED * l_pred

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # EMA update target encoder
        with torch.no_grad():
            for op, tp in zip(encoder.parameters(), target.parameters()):
                tp.data = EMA_MOMENTUM * tp.data + (1 - EMA_MOMENTUM) * op.data

        if step % LOG_EVERY == 0:
            entry = {
                "step":    step,
                "total":   total.item(),
                "triplet": l_trip.item(),
                "vicreg":  l_vic.item(),
                "pred":    l_pred.item(),
                "d_pos":   d_pos,
                "d_neg":   d_neg,
            }
            loss_log.append(entry)
            print(f"Step {step:5d} | "
                  f"total={total.item():.3f} | "
                  f"triplet={l_trip.item():.4f} | "
                  f"d_pos={d_pos:.2f} d_neg={d_neg:.2f} | "
                  f"vicreg={l_vic.item():.3f} | "
                  f"pred={l_pred.item():.4f}")

        if (step + 1) % SAVE_EVERY == 0:
            measure_room_separation(encoder, buffer, room_index)

    # ── Final measurement ──────────────────────────────────────────────────────
    print("\n── Room separation AFTER fine-tuning ──")
    encoder.eval()
    final_ratio = measure_room_separation(encoder, buffer, room_index)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "online_encoder": encoder.state_dict(),
        "target_encoder": target.state_dict(),
        "predictor":      predictor.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "step":           ckpt["step"] + N_FINETUNE_STEPS,
        "loss_log":       loss_log,
        "final_ratio":    final_ratio,
    }, CHECKPOINT_OUT)
    print(f"\nSaved: {CHECKPOINT_OUT}")


if __name__ == "__main__":
    finetune()