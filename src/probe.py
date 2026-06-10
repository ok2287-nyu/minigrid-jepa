import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from encoder import Encoder
from data_collector import ReplayBuffer, DataCollector

import os
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class LinearProbe(nn.Module):
    """
    A single linear layer trained on top of a FROZEN encoder.

    Task: predict agent (x, y) position from latent vector z.

    If this works well, it proves the world model learned
    spatial structure without ever being shown position labels.
    """
    def __init__(self, latent_dim=256, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.linear(z)  # (B, 2) predicted (x, y)


class ProbeEvaluator:
    """
    Loads a trained encoder, freezes it, trains a linear probe,
    and visualizes what the latent space learned.
    """
    def __init__(self, checkpoint_path, latent_dim=256, device=None):
        self.device     = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.latent_dim = latent_dim

        # ── Load trained encoder ─────────────────────────────────────
        self.encoder = Encoder(latent_dim).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["online_encoder"])

        # Freeze completely — zero gradients, eval mode
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        print(f"Encoder loaded from {checkpoint_path}")
        print(f"Training on: {self.device}")

        # ── Linear probe (only this gets trained) ────────────────────
        self.probe     = LinearProbe(latent_dim, output_dim=2).to(self.device)
        self.optimizer = torch.optim.Adam(self.probe.parameters(), lr=1e-3)

    @torch.no_grad()
    def encode_buffer(self, buffer, n_samples=10_000):
        """
        Pass observations through frozen encoder to get latent vectors.
        Returns z vectors and their corresponding ground truth positions.

        This is the dataset the linear probe trains on.
        """
        print(f"Encoding {n_samples} observations...")

        # Sample from buffer
        idxs = np.random.randint(0, buffer.size, size=n_samples)
        obs  = torch.tensor(
            buffer.obs[idxs], dtype=torch.float32
        ) / 255.0                           # (N, 3, 64, 64)
        positions = torch.tensor(
            buffer.positions[idxs], dtype=torch.float32
        )                                   # (N, 2)

        # Encode in batches to avoid OOM
        z_list = []
        batch_size = 512
        for i in range(0, n_samples, batch_size):
            batch = obs[i:i+batch_size].to(self.device)
            z     = self.encoder(batch)     # (batch, 256)
            z_list.append(z.cpu())

        z_all = torch.cat(z_list, dim=0)    # (N, 256)
        print(f"Encoded. z shape: {z_all.shape}")
        return z_all, positions

    def train_probe(self, z, positions, n_epochs=50):
        """
        Train the linear probe to predict (x,y) from z.

        z:         (N, 256) latent vectors  — input
        positions: (N, 2)   ground truth    — target

        Only the linear probe's weights update here.
        Encoder is completely frozen.
        """
        print(f"\nTraining linear probe for {n_epochs} epochs...")

        # Train/val split — 80/20
        N         = z.shape[0]
        split     = int(0.8 * N)
        z_train, z_val         = z[:split],         z[split:]
        pos_train, pos_val     = positions[:split],  positions[split:]

        # Normalize positions to 0-1 range for stable training
        pos_max = positions.max()
        pos_train = pos_train / pos_max
        pos_val   = pos_val   / pos_max

        train_losses = []
        val_losses   = []

        for epoch in range(n_epochs):
            # ── Train ────────────────────────────────────────────────
            self.probe.train()
            self.optimizer.zero_grad()

            pred_train = self.probe(z_train.to(self.device))
            loss_train = F.mse_loss(pred_train, pos_train.to(self.device))

            loss_train.backward()
            self.optimizer.step()

            # ── Validate ─────────────────────────────────────────────
            self.probe.eval()
            with torch.no_grad():
                pred_val  = self.probe(z_val.to(self.device))
                loss_val  = F.mse_loss(pred_val, pos_val.to(self.device))

            train_losses.append(loss_train.item())
            val_losses.append(loss_val.item())

            if (epoch + 1) % 10 == 0:
                # Convert MSE back to grid units for interpretability
                mse_grid = loss_val.item() * (pos_max.item() ** 2)
                rmse_grid = mse_grid ** 0.5
                print(f"  Epoch [{epoch+1:>3}/{n_epochs}]  "
                      f"train_loss: {loss_train.item():.4f}  "
                      f"val_loss: {loss_val.item():.4f}  "
                      f"position_error: {rmse_grid:.2f} cells")

        return train_losses, val_losses, pos_max

    def visualize_latent_space(self, z, positions, save_path="latent_space.png"):
        """
        Project latent vectors to 2D using UMAP and color by position.

        If the world model learned spatial structure, nearby grid positions
        should cluster together in the 2D projection.

        This is the visual wow moment.
        """
        print("\nGenerating latent space visualization...")

        try:
            import umap
        except ImportError:
            print("UMAP not installed. Run: pip install umap-learn")
            return

        # UMAP: reduce 256 dimensions to 2
        reducer    = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        z_2d       = reducer.fit_transform(z.numpy())   # (N, 2)

        # Color by x position and y position separately
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Latent Space Structure Learned by World Model\n"
            "(colored by agent position — no position labels used during training)",
            fontsize=12
        )

        for ax, dim, label in zip(axes, [0, 1], ["X position", "Y position"]):
            scatter = ax.scatter(
                z_2d[:, 0], z_2d[:, 1],
                c=positions[:, dim].numpy(),
                cmap="viridis",
                s=2,
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax)
            ax.set_title(f"Colored by {label}")
            ax.set_xlabel("UMAP dim 1")
            ax.set_ylabel("UMAP dim 2")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
        plt.show()

    def plot_probe_loss(self, train_losses, val_losses,
                        save_path="probe_training.png"):
        """Plot probe training curve."""
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train loss", alpha=0.8)
        plt.plot(val_losses,   label="Val loss",   alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Linear Probe Training\n"
                  "(predicting agent position from frozen latent vectors)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved probe curve to {save_path}")
        plt.show()


if __name__ == "__main__":

    # ── Load buffer ───────────────────────────────────────────────────
    buffer = ReplayBuffer(capacity=100_000)
    buffer.load("data/replay_buffer_phase1.pkl")

    # ── Load trained encoder ──────────────────────────────────────────
    evaluator = ProbeEvaluator(
        checkpoint_path = "checkpoints/jepa_phase1_final.pt",
        latent_dim      = 256,
    )

    # ── Encode observations → latent vectors ─────────────────────────
    z, positions = evaluator.encode_buffer(buffer, n_samples=10_000)

    # ── Train linear probe ────────────────────────────────────────────
    train_losses, val_losses, pos_max = evaluator.train_probe(
        z, positions, n_epochs=50
    )

    # ── Plot probe training curve ─────────────────────────────────────
    evaluator.plot_probe_loss(
        train_losses, val_losses,
        save_path="notebooks/probe_training.png"
    )

    # ── Visualize latent space with UMAP ─────────────────────────────
    evaluator.visualize_latent_space(
        z, positions,
        save_path="notebooks/latent_space.png"
    )