"""
Fine-tune JEPA encoder so z-distance respects GEODESIC (walking) distance.

Core idea:
  Instead of euclidean/visual distance, make ||z_i - z_j|| proportional to
  BFS shortest-path distance between cell i and cell j.

Why this solves cross-room navigation:
  - Two cells separated by a wall have LARGE geodesic distance (must go around)
    → pushed far apart in z  → rooms naturally separate
  - Two cells connected through a doorway have SMALL geodesic distance
    → kept close in z        → doorways become bridges
  - Within an open room, geodesic ≈ euclidean
    → position structure preserved automatically

This is ONE coherent objective. No fighting losses, no hinge collapse.
A regression loss keeps producing gradient until z-distances match targets.

Usage:
    python src/finetune_geodesic.py

Reads:  checkpoints/jepa_fourrooms_final.pt
        data/replay_buffer_fourrooms_with_nextpos.pkl
Writes: checkpoints/jepa_geodesic_final.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import pickle
from collections import deque
import gymnasium as gym
import minigrid
from pathlib import Path
import os

from encoder_v2 import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
Path("checkpoints").mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = "checkpoints/jepa_fourrooms_final.pt"
CHECKPOINT_OUT = "checkpoints/jepa_geodesic_final.pt"
BUFFER_PATH    = "data/replay_buffer_fourrooms_with_nextpos.pkl"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

N_FINETUNE_STEPS = 10_000
BATCH_SIZE       = 128      # number of cells per batch (we do pairwise within batch)
LR               = 3e-5
EMA_MOMENTUM     = 0.99

LAMBDA_GEO       = 1.0      # geodesic matching loss
LAMBDA_VICREG    = 0.5      # collapse prevention (lower — geodesic does most work)
LAMBDA_PRED      = 0.1      # keep dynamics accurate
GEO_SCALE        = 1.0      # z_dist target = GEO_SCALE * geodesic_dist (learnable alt below)

LOG_EVERY        = 200
SAVE_EVERY       = 5_000


# ── BFS geodesic distance ──────────────────────────────────────────────────────
def bfs_path(start, goal, walkable_set):
    if start == goal:
        return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (x+dx, y+dy)
            if nxt in walkable_set and nxt not in visited:
                if nxt == goal:
                    return path + [nxt]
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def compute_all_pairs_geodesic(walkable_set):
    """
    BFS from every walkable cell to get all-pairs shortest path distances.
    Returns dict: (cell_a, cell_b) -> geodesic distance.
    For a 19x19 FourRooms (~250 walkable cells) this is fast.
    """
    cells = sorted(walkable_set)
    dist_map = {}

    for start in cells:
        # Single-source BFS
        queue   = deque([start])
        dists   = {start: 0}
        while queue:
            cur = queue.popleft()
            cx, cy = cur
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nxt = (cx+dx, cy+dy)
                if nxt in walkable_set and nxt not in dists:
                    dists[nxt] = dists[cur] + 1
                    queue.append(nxt)
        for goal, d in dists.items():
            dist_map[(start, goal)] = d

    print(f"All-pairs geodesic computed: {len(cells)} cells, "
          f"{len(dist_map)} pairs")
    return dist_map


def find_walkable(env):
    grid = env.unwrapped.grid
    w, h = env.unwrapped.width, env.unwrapped.height
    cells = set()
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            cell = grid.get(x, y)
            if cell is None or cell.type in ('goal', 'door'):
                cells.add((x, y))
    return cells


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


def build_pos_index(buffer):
    """
    Map each walkable position -> list of buffer indices at that position.
    Lets us sample observations for a given cell.
    """
    pos_index = {}
    for i in range(buffer.size):
        x, y = int(buffer.positions[i, 0]), int(buffer.positions[i, 1])
        pos_index.setdefault((x, y), []).append(i)
    print(f"Position index: {len(pos_index)} distinct cells")
    return pos_index


# ── Geodesic batch sampler ─────────────────────────────────────────────────────
def sample_geodesic_batch(buffer, pos_index, dist_map, batch_size):
    """
    Sample a batch of cells. For each, grab one observation.
    Return obs + the pairwise geodesic distance matrix for the batch.
    The loss then matches z-distances to these geodesic distances.
    """
    cells = list(pos_index.keys())
    # Only keep cells that appear in dist_map (all walkable do)
    chosen = random.sample(cells, min(batch_size, len(cells)))

    obs_list = []
    valid_cells = []
    for c in chosen:
        idx = random.choice(pos_index[c])
        obs_list.append(torch.tensor(buffer.obs[idx], dtype=torch.float32) / 255.0)
        valid_cells.append(c)

    obs_batch = torch.stack(obs_list).to(DEVICE)   # (B, 3, 64, 64)

    # Build geodesic distance matrix (B, B)
    B = len(valid_cells)
    geo = torch.zeros(B, B, dtype=torch.float32)
    for i in range(B):
        for j in range(B):
            d = dist_map.get((valid_cells[i], valid_cells[j]), None)
            if d is None:
                d = dist_map.get((valid_cells[j], valid_cells[i]), 50)  # fallback
            geo[i, j] = d

    return obs_batch, geo.to(DEVICE)


# ── Geodesic matching loss ─────────────────────────────────────────────────────
def geodesic_loss(z, geo_target, scale):
    """
    Match pairwise z-distances to geodesic distances.

    z:          (B, D)
    geo_target: (B, B) geodesic distances

    We compute pairwise euclidean distances in z, then regress them
    onto (scale * geodesic). Using a correlation-friendly normalized
    form so absolute scale doesn't dominate.
    """
    B = z.shape[0]
    # Pairwise z distances (B, B)
    z_dist = torch.cdist(z, z, p=2)            # euclidean

    # Target: scale * geodesic
    target = scale * geo_target

    # Only use upper triangle (avoid double-count + diagonal)
    mask = torch.triu(torch.ones(B, B, device=z.device), diagonal=1).bool()

    z_vals      = z_dist[mask]
    target_vals = target[mask]

    # MSE regression — keeps gradient flowing, won't collapse to 0
    loss = F.mse_loss(z_vals, target_vals)

    # Diagnostic: correlation between z_dist and geodesic
    with torch.no_grad():
        zc = z_vals - z_vals.mean()
        tc = target_vals - target_vals.mean()
        corr = (zc * tc).sum() / (zc.norm() * tc.norm() + 1e-8)

    return loss, corr.item(), z_vals.mean().item(), target_vals.mean().item()


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_var_cov(z):
    """Variance + covariance only (no invariance term — geodesic handles structure)."""
    B, D = z.shape
    std  = torch.sqrt(z.var(dim=0) + 1e-4)
    var  = F.relu(1 - std).mean()
    zc   = z - z.mean(dim=0)
    cov  = (zc.T @ zc) / (B - 1)
    cov_loss = off_diagonal(cov).pow(2).sum() / D
    return 25.0 * var + 1.0 * cov_loss


# ── Metrics ────────────────────────────────────────────────────────────────────
def get_room(x, y):
    if x <= 9 and y <= 9:   return 0
    elif x >= 9 and y <= 9: return 1
    elif x <= 9 and y >= 9: return 2
    else:                    return 3


@torch.no_grad()
def measure_separation(encoder, buffer, pos_index, n_samples=300):
    encoder.eval()
    cells = list(pos_index.keys())
    chosen = random.sample(cells, min(n_samples, len(cells)))

    obs_list, rooms = [], []
    for c in chosen:
        idx = random.choice(pos_index[c])
        obs_list.append(torch.tensor(buffer.obs[idx], dtype=torch.float32) / 255.0)
        rooms.append(get_room(*c))
    obs_batch = torch.stack(obs_list).to(DEVICE)

    zs = []
    for i in range(0, len(obs_batch), 64):
        zs.append(encoder(obs_batch[i:i+64]).cpu())
    zs = torch.cat(zs, dim=0)

    within, between = [], []
    for i in range(len(zs)):
        for j in range(i + 1, min(i + 30, len(zs))):
            d = (zs[i] - zs[j]).pow(2).sum().item()
            if rooms[i] == rooms[j]: within.append(d)
            else:                    between.append(d)

    w, b = np.mean(within), np.mean(between)
    ratio = b / w if w > 0 else float("inf")
    print(f"  Within-room:  {w:.2f}")
    print(f"  Between-room: {b:.2f}")
    print(f"  Ratio: {ratio:.2f}x")
    encoder.train()
    return ratio


# ── Main ───────────────────────────────────────────────────────────────────────
def finetune():
    print(f"Device: {DEVICE}\n")

    # ── Compute geodesic distances ─────────────────────────────────────────────
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env.reset()
    walkable = find_walkable(env)
    env.close()
    print(f"Walkable cells: {len(walkable)}")
    dist_map = compute_all_pairs_geodesic(walkable)

    # ── Load buffer ────────────────────────────────────────────────────────────
    buffer = ReplayBufferWithNextPos(capacity=300_000)
    buffer.load(BUFFER_PATH)
    pos_index = build_pos_index(buffer)

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

    print("\n── Separation BEFORE ──")
    measure_separation(encoder, buffer, pos_index)

    print(f"\nFine-tuning for {N_FINETUNE_STEPS} steps...")
    loss_log = []

    for step in range(N_FINETUNE_STEPS):
        encoder.train()
        predictor.train()

        # ── Geodesic matching ──────────────────────────────────────────────────
        obs_batch, geo = sample_geodesic_batch(buffer, pos_index, dist_map, BATCH_SIZE)
        z = encoder(obs_batch)
        l_geo, corr, z_mean, geo_mean = geodesic_loss(z, geo, GEO_SCALE)

        # ── VICReg (var+cov) to prevent collapse ──────────────────────────────
        l_vic = vicreg_var_cov(z)

        # ── Predictor ──────────────────────────────────────────────────────────
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

        total = (LAMBDA_GEO    * l_geo +
                 LAMBDA_VICREG * l_vic +
                 LAMBDA_PRED   * l_pred)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        with torch.no_grad():
            for op, tp in zip(encoder.parameters(), target.parameters()):
                tp.data = EMA_MOMENTUM * tp.data + (1 - EMA_MOMENTUM) * op.data

        if step % LOG_EVERY == 0:
            entry = {
                "step": step, "total": total.item(),
                "geo": l_geo.item(), "corr": corr,
                "vic": l_vic.item(), "pred": l_pred.item(),
                "z_mean": z_mean, "geo_mean": geo_mean,
            }
            loss_log.append(entry)
            print(f"Step {step:5d} | "
                  f"geo={l_geo.item():.3f} corr={corr:.3f} | "
                  f"z_d={z_mean:.2f} geo_d={geo_mean:.2f} | "
                  f"vic={l_vic.item():.3f} pred={l_pred.item():.4f}")

        if (step + 1) % SAVE_EVERY == 0:
            print(f"\n── Separation at step {step+1} ──")
            measure_separation(encoder, buffer, pos_index)

    print("\n── Separation AFTER ──")
    encoder.eval()
    measure_separation(encoder, buffer, pos_index)

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