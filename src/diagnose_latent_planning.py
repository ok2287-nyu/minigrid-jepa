"""
Latent-space planning diagnostic.

THE question this answers:
  With the geodesic-finetuned encoder, can we pick the correct doorway
  to route through using ONLY latent distances — no high-level controller?

Method (the human "find the next passage" move, done in z-space):
  subgoal_door = argmin over doorways of [ dist(z_start, z_door) + dist(z_door, z_goal) ]

We compare that latent choice against:
  (a) GROUND TRUTH: the doorway BFS actually routes through first
  (b) GEODESIC ORACLE: argmin using TRUE walking distances
      (isolates "is the representation good" from "is doorway-choice ambiguous")

If latent ≈ oracle ≈ ground truth → navigation is derivable from z alone.

Usage:
    python src/diagnose_latent_planning.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym
import minigrid
from pathlib import Path
from PIL import Image
import os

from encoder_v2 import Encoder

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

CHECKPOINT   = "checkpoints/jepa_geodesic_final.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 0           # fix layout so geodesic + z are consistent
N_PAIRS      = 300         # cross-room test pairs
WALL_LINE    = 9           # col/row of the dividing walls in 19x19 FourRooms


# ── BFS ────────────────────────────────────────────────────────────────────────
def bfs_path(start, goal, walkable_set):
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (x, y), path = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (x+dx, y+dy)
            if nxt in walkable_set and nxt not in seen:
                if nxt == goal:
                    return path + [nxt]
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return None


def single_source_geo(start, walkable_set):
    """BFS distances from start to all reachable cells."""
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        cx, cy = cur
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cx+dx, cy+dy)
            if nxt in walkable_set and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


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


def find_doorways(walkable_set):
    """
    Doorways = walkable cells sitting on a dividing wall line.
    In 19x19 FourRooms the walls are at x==9 and y==9; the walkable
    cells on those lines are the gaps (doorways) connecting rooms.
    """
    doors = [c for c in walkable_set if c[0] == WALL_LINE or c[1] == WALL_LINE]
    return sorted(doors)


def get_room(x, y):
    if x <= WALL_LINE and y <= WALL_LINE:   return 0
    elif x >= WALL_LINE and y <= WALL_LINE: return 1
    elif x <= WALL_LINE and y >= WALL_LINE: return 2
    else:                                    return 3


# ── Encoding helpers ───────────────────────────────────────────────────────────
def preprocess(frame):
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def encode_cell(encoder, env, x, y):
    """
    Direction-invariant place embedding: average z over the 4 agent
    directions at this cell. Gives a 'where am I' vector independent
    of which way the agent faces.
    """
    zs = []
    for d in range(4):
        env.unwrapped.agent_pos = np.array([x, y])
        env.unwrapped.agent_dir = d
        frame = env.render()
        z = encoder(preprocess(frame).to(DEVICE))
        zs.append(z)
    return torch.stack(zs).mean(dim=0)   # (1, 256)


def first_doorway_on_path(path):
    """The first cell on the BFS path that is a doorway (on a wall line)."""
    for (x, y) in path:
        if x == WALL_LINE or y == WALL_LINE:
            return (x, y)
    return None


# ── Main diagnostic ────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Fixed layout ───────────────────────────────────────────────────────────
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env.reset(seed=SEED)
    walkable = find_walkable(env)
    walkable_set = walkable
    doors = find_doorways(walkable_set)
    print(f"Walkable cells: {len(walkable)}")
    print(f"Doorways found: {len(doors)} -> {doors}")
    for d in doors:
        print(f"   door {d}: connects rooms touching it")

    # ── Load encoder ───────────────────────────────────────────────────────────
    encoder = Encoder(latent_dim=256).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder.eval()
    print(f"Encoder loaded: {CHECKPOINT}")

    # ── Pre-encode all doorways ────────────────────────────────────────────────
    print("\nEncoding doorway place-vectors...")
    door_z = {d: encode_cell(encoder, env, d[0], d[1]) for d in doors}

    # ── Build test pairs (cross-room only) ─────────────────────────────────────
    cells = sorted(walkable_set)
    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
        attempts += 1
        s = random.choice(cells)
        g = random.choice(cells)
        if s == g:
            continue
        if get_room(*s) == get_room(*g):
            continue   # cross-room only
        if s in door_z or g in door_z:
            continue   # don't start/end on a door
        path = bfs_path(s, g, walkable_set)
        if path is None:
            continue
        pairs.append((s, g, path))

    print(f"Test pairs (cross-room): {len(pairs)}\n")

    # ── Run the three planners ─────────────────────────────────────────────────
    latent_correct   = 0   # latent argmin matches BFS first doorway
    oracle_correct   = 0   # geodesic argmin matches BFS first doorway
    latent_eq_oracle = 0   # latent matches geodesic oracle
    latent_on_path   = 0   # latent-picked door lies on an optimal path

    for (s, g, path) in pairs:
        gt_door = first_doorway_on_path(path)
        if gt_door is None:
            continue

        # Geodesic distances from start and to goal
        geo_from_s = single_source_geo(s, walkable_set)
        geo_from_g = single_source_geo(g, walkable_set)

        # (a) GEODESIC ORACLE: argmin true [geo(s,d) + geo(d,g)]
        oracle_door = min(
            doors,
            key=lambda d: geo_from_s.get(d, 1e9) + geo_from_g.get(d, 1e9)
        )

        # (b) LATENT: argmin [||z_s - z_d|| + ||z_d - z_g||]
        z_s = encode_cell(encoder, env, s[0], s[1])
        z_g = encode_cell(encoder, env, g[0], g[1])
        latent_door = min(
            doors,
            key=lambda d: (
                (z_s - door_z[d]).norm().item() +
                (door_z[d] - z_g).norm().item()
            )
        )

        # Score
        if latent_door == gt_door:
            latent_correct += 1
        if oracle_door == gt_door:
            oracle_correct += 1
        if latent_door == oracle_door:
            latent_eq_oracle += 1

        # On-path check: is latent_door on the optimal route?
        # (it's on an optimal path if geo(s,d)+geo(d,g) == geo(s,g))
        geo_sg = geo_from_s.get(g, 1e9)
        if abs((geo_from_s.get(latent_door,1e9) +
                geo_from_g.get(latent_door,1e9)) - geo_sg) < 1e-6:
            latent_on_path += 1

    n = len(pairs)
    print("══════════════════════════════════════════════")
    print("  LATENT-SPACE PLANNING RESULTS")
    print("══════════════════════════════════════════════")
    print(f"  Pairs tested: {n}")
    print()
    print(f"  Latent == BFS first doorway:   {100*latent_correct/n:.1f}%")
    print(f"  Geodesic oracle == BFS first:  {100*oracle_correct/n:.1f}%")
    print(f"  Latent == Geodesic oracle:     {100*latent_eq_oracle/n:.1f}%")
    print(f"  Latent door lies on optimal:   {100*latent_on_path/n:.1f}%")
    print("══════════════════════════════════════════════")
    print()
    print("  How to read this:")
    print("  - 'Latent == oracle' high  -> representation captures geodesic")
    print("    structure well enough to plan (THE key number).")
    print("  - 'oracle == BFS first' < 100% just means single-doorway")
    print("    decomposition is ambiguous for diagonal rooms; compare")
    print("    latent against ORACLE, not against BFS, to judge the encoder.")
    print("  - 'Latent on optimal path' high -> picked door is a valid")
    print("    next passage even when it isn't the unique BFS choice.")

    env.close()


if __name__ == "__main__":
    main()