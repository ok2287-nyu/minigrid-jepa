"""
Position Embedder — maps normalized (x,y) → approximate z vector

This is the differentiable bridge needed for joint training:
    high-level outputs (x,y) → position_embedder → z_approx
    z_approx feeds into low-level as subgoal state
    gradients flow back through embedder to high-level

Training: supervised regression
    input:  normalized (x,y) position
    target: actual z = encoder(render(x,y))
    loss:   MSE

After training, position_embedder(x,y) ≈ encoder(render(x,y))
but is fully differentiable with respect to (x,y).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid
import sys
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)

from encoder_v2 import Encoder


class PositionEmbedder(nn.Module):
    """
    Maps normalized (x,y) position → approximate z vector (256 dims)

    Input:  (B, 2) normalized position [x/max_x, y/max_y]
    Output: (B, 256) approximate latent vector

    Trained to approximate encoder(render(x,y)).
    Used during joint training to make subgoal encoding differentiable.
    """
    def __init__(self, pos_dim=2, n_dirs=4, latent_dim=256, hidden_dim=512):
        super().__init__()
        input_dim = pos_dim + n_dirs   # 2 + 4 = 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, pos):
        """
        pos: (B, 2) normalized (x,y)
        returns: (B, 256) approximate z
        """
        return self.net(pos)


def train_position_embedder(
    jepa_checkpoint,
    env_id     = "MiniGrid-FourRooms-v0",
    latent_dim = 256,
    n_steps    = 10000,
    batch_size = 256,
    lr         = 1e-3,
    device     = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training position embedder on {device}")

    # Load frozen encoder
    encoder = Encoder(latent_dim).to(device)
    ckpt = torch.load(jepa_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Environment
    env = gym.make(env_id, render_mode="rgb_array")
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

    def encode_pos(x, y, d=0):
        env.unwrapped.agent_pos = np.array([x, y])
        env.unwrapped.agent_dir = d
        frame = env.render()
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            return encoder(t)

    # ── Precompute ALL z vectors upfront ──────────────────────────
    print("Precomputing z vectors for all positions...")
    env.reset()
    walkable = find_walkable()

    all_pos = []
    all_z   = []

    for (x, y) in walkable:
        for d in range(4):
            z = encode_pos(x, y, d)
            dir_oh = [1 if i==d else 0 for i in range(4)]
            all_pos.append([x/max_x, y/max_y] + dir_oh)  # 6 values
            all_z.append(z)

    pos_tensor = torch.tensor(all_pos, dtype=torch.float32).to(device)
    z_tensor   = torch.cat(all_z, dim=0)
    N          = len(all_pos)
    print(f"Precomputed {N} z vectors. Starting training...")

    # ── Train on precomputed data ──────────────────────────────────
    embedder  = PositionEmbedder(pos_dim=2, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(embedder.parameters(), lr=lr)
    losses    = []

    for step in range(n_steps):
        idxs  = torch.randint(0, N, (batch_size,))
        pos_b = pos_tensor[idxs]
        z_b   = z_tensor[idxs]

        z_pred = embedder(pos_b)
        loss   = F.mse_loss(z_pred, z_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg_loss = np.mean(losses[-500:])
            print(f"  Step [{step+1:>6}/{n_steps}]  loss: {avg_loss:.6f}")

    print("\nTraining complete.")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "embedder" : embedder.state_dict(),
        "max_x"    : max_x,
        "max_y"    : max_y,
    }, "checkpoints/position_embedder.pt")
    print("Saved: checkpoints/position_embedder.pt")

    # Verify quality
    print("\nVerification — encoder vs embedder distances:")
    env.reset()
    walkable = find_walkable()
    dists = []
    for _ in range(50):
        x, y = walkable[np.random.randint(len(walkable))]
        d    = np.random.randint(0, 4)
        z_real = encode_pos(x, y, d)
        dir_oh = [1 if i==d else 0 for i in range(4)]
        pos    = torch.tensor(
            [[x/max_x, y/max_y] + dir_oh],
            dtype=torch.float32
        ).to(device)   # (1, 6)
        with torch.no_grad():
            z_approx = embedder(pos)
        dist = (z_real - z_approx).norm().item()
        dists.append(dist)

    print(f"  Mean L2 distance: {np.mean(dists):.4f}")
    print(f"  Max  L2 distance: {np.max(dists):.4f}")
    print(f"  (Adjacent cell distance in encoder space: ~4.0)")
    print(f"  Want: mean distance << 4.0 for useful approximation")

    return embedder

if __name__ == "__main__":
    train_position_embedder(
        jepa_checkpoint = "checkpoints/jepa_fourrooms_final.pt",
        env_id          = "MiniGrid-FourRooms-v0",
        latent_dim      = 256,
        n_steps         = 10000,
        batch_size      = 256,
        lr              = 1e-3,
    )