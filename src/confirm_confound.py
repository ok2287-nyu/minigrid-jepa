"""
Confirm WHY latent doorway routing failed despite corr=0.97.

Two competing hypotheses:
  H1 (layout confound): encoder learned POSITION, not connectivity, because
      the buffer mixed many doorway layouts while geodesic was from one layout.
      -> On a fixed layout, z-distance will track EUCLIDEAN better than GEODESIC.
  H2 (corr too coarse): encoder did learn fixed-layout geodesic, but corr is
      too blunt to reflect doorway-level routing.
      -> On a fixed layout, z-geodesic corr stays high but routing still fails.

This script measures, on ONE fixed layout (seed=0), with observations rendered
FRESH from that layout (not pulled from the mixed-layout buffer):
    corr(z_dist, geodesic_dist)   <- connectivity
    corr(z_dist, euclidean_dist)  <- raw position

Read-out:
  z-euclid >> z-geo, and z-geo modest  -> H1 confirmed (position fallback)
  z-geo high (~0.95) but routing was 20% -> H2 (need finer objective)

Usage:
    python src/confirm_confound.py
"""

import torch
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

CHECKPOINT = "checkpoints/jepa_geodesic_final.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 0


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


def single_source_geo(start, walkable_set):
    q = deque([start]); dist = {start: 0}
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (cx+dx, cy+dy)
            if nxt in walkable_set and nxt not in dist:
                dist[nxt] = dist[(cx,cy)] + 1
                q.append(nxt)
    return dist


def preprocess(frame):
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)


def corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum() / (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + 1e-9))


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env.reset(seed=SEED)
    walkable = find_walkable(env)
    cells = sorted(walkable)
    print(f"Fixed layout (seed={SEED}): {len(cells)} walkable cells")

    encoder = Encoder(latent_dim=256).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder.eval()

    # Encode every cell two ways: averaged-over-directions, and single-direction
    z_avg = {}
    z_dir0 = {}
    for (x, y) in cells:
        zs = []
        for d in range(4):
            env.unwrapped.agent_pos = np.array([x, y])
            env.unwrapped.agent_dir = d
            z = encoder(preprocess(env.render()).to(DEVICE))
            zs.append(z)
            if d == 0:
                z_dir0[(x,y)] = z
        z_avg[(x,y)] = torch.stack(zs).mean(dim=0)

    # Precompute geodesic from each cell
    geo_cache = {c: single_source_geo(c, walkable) for c in cells}

    # Sample pairs and gather distances
    def gather(zmap):
        zd, gd, ed = [], [], []
        for _ in range(8000):
            a = random.choice(cells); b = random.choice(cells)
            if a == b:
                continue
            g = geo_cache[a].get(b)
            if g is None:
                continue
            zdist = (zmap[a] - zmap[b]).norm().item()
            edist = np.hypot(a[0]-b[0], a[1]-b[1])
            zd.append(zdist); gd.append(g); ed.append(edist)
        return zd, gd, ed

    print("\n── Averaged-over-directions place vectors ──")
    zd, gd, ed = gather(z_avg)
    print(f"  corr(z_dist, geodesic):  {corr(zd, gd):.3f}")
    print(f"  corr(z_dist, euclidean): {corr(zd, ed):.3f}")
    print(f"  corr(euclidean, geodesic) [baseline]: {corr(ed, gd):.3f}")

    print("\n── Single-direction (dir=0) vectors ──")
    zd, gd, ed = gather(z_dir0)
    print(f"  corr(z_dist, geodesic):  {corr(zd, gd):.3f}")
    print(f"  corr(z_dist, euclidean): {corr(zd, ed):.3f}")

    print("\n── Interpretation ──")
    print("  If z-euclid > z-geo  -> encoder learned POSITION, not walls (H1).")
    print("  If z-geo ~0.95 here  -> fixed-layout structure IS present;")
    print("     the training corr was NOT confounded, routing needs finer signal (H2).")
    print("  Compare z-geo here vs training's 0.97: a big drop = layout confound.")

    env.close()


if __name__ == "__main__":
    main()