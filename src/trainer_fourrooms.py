import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from pathlib import Path
import os

from encoder_v2 import Encoder, Predictor
from data_collector_v2 import ReplayBuffer

Path("data").mkdir(exist_ok=True)
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class JEPATrainer:
    def __init__(
        self,
        latent_dim   = 256,
        n_actions    = 3,
        lr           = 1e-4,
        batch_size   = 256,
        ema_momentum = 0.99,
        device       = None,
    ):
        self.device       = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size   = batch_size
        self.ema_momentum = ema_momentum
        self.loss_log     = []

        print(f"Training on: {self.device}")

        self.online_encoder = Encoder(latent_dim).to(self.device)
        self.predictor      = Predictor(latent_dim, n_actions).to(self.device)
        self.target_encoder = copy.deepcopy(
            self.online_encoder
        ).to(self.device)
        self._freeze(self.target_encoder)

        self.optimizer = torch.optim.Adam(
            list(self.online_encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=lr
        )
        self.step = 0

    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.ema_momentum       * target_param.data +
                (1 - self.ema_momentum) * online_param.data
            )

    def _invariance_loss(self, state_pred, state_actual):
        z_pred   = state_pred[:, :256]
        z_actual = state_actual[:, :256]
        return F.mse_loss(z_pred, z_actual)

    def _variance_loss(self, z_pred, z_actual):
        gamma       = 1.0
        std_pred    = z_pred.std(dim=0)
        std_actual  = z_actual.std(dim=0)
        loss_pred   = F.relu(gamma - std_pred).mean()
        loss_actual = F.relu(gamma - std_actual).mean()
        return (loss_pred + loss_actual) / 2

    def _covariance_loss(self, z_pred, z_actual):
        def off_diagonal_cov(z):
            B, D = z.shape
            z        = z - z.mean(dim=0)
            cov      = (z.T @ z) / (B - 1)
            off_diag = cov ** 2
            off_diag.fill_diagonal_(0)
            return off_diag.sum() / D
        return off_diagonal_cov(z_pred) + off_diagonal_cov(z_actual)

    def _action_covariance_loss(self):
        return self.predictor.action_covariance_loss()

    def _direction_transition_loss(self, state_predicted, next_directions):
        dir_logits = state_predicted[:, 256:]
        return F.cross_entropy(dir_logits, next_directions)

    def compute_loss(self, batch):
        obs             = batch["obs"].to(self.device)
        next_obs        = batch["next_obs"].to(self.device)
        actions         = batch["actions"].to(self.device)
        directions      = batch["directions"].to(self.device)
        next_directions = batch["next_directions"].to(self.device)

        z_current = self.online_encoder(obs)

        with torch.no_grad():
            z_actual = self.target_encoder(next_obs)

        dir_onehot      = F.one_hot(directions,      num_classes=4).float()
        next_dir_onehot = F.one_hot(next_directions, num_classes=4).float()

        state_current = torch.cat([z_current, dir_onehot],      dim=-1)
        state_actual  = torch.cat([z_actual,  next_dir_onehot], dim=-1)

        state_predicted = self.predictor(state_current, actions)

        loss_invariance     = self._invariance_loss(state_predicted, state_actual)
        loss_variance       = self._variance_loss(state_predicted, state_actual)
        loss_covariance     = self._covariance_loss(state_predicted, state_actual)
        loss_action_cov     = self._action_covariance_loss()
        loss_dir_transition = self._direction_transition_loss(
            state_predicted, next_directions
        )

        loss = (
            100.0 * loss_invariance     +
             25.0 * loss_variance       +
              1.0 * loss_covariance     +
             10.0 * loss_action_cov     +
             75.0 * loss_dir_transition
        )

        if self.step % 200 == 0:
            self.loss_log.append({
                "step"       : self.step,
                "total"      : loss.item(),
                "invariance" : loss_invariance.item(),
                "variance"   : loss_variance.item(),
                "covariance" : loss_covariance.item(),
                "action_cov" : loss_action_cov.item(),
                "dir_trans"  : loss_dir_transition.item(),
            })

        return loss

    def train_step(self, buffer):
        self.online_encoder.train()
        self.predictor.train()

        batch = buffer.sample(self.batch_size)
        loss  = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.online_encoder.parameters()) +
            list(self.predictor.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self._update_target_encoder()

        self.step += 1
        return loss.item()

    def train(self, buffer, n_steps=30_000, log_every=200, save_every=5000):
        print(f"\nStarting JEPA FourRooms training for {n_steps} steps...")
        print(f"Environment: FourRooms (walls, rooms, doorways)")
        print(f"Batch size:   {self.batch_size}")
        print(f"EMA momentum: {self.ema_momentum}\n")

        losses = []

        for step in range(n_steps):
            loss = self.train_step(buffer)
            losses.append(loss)

            if (step + 1) % log_every == 0:
                avg_loss = np.mean(losses[-log_every:])
                if self.loss_log:
                    last = self.loss_log[-1]
                    print(
                        f"Step [{step+1:>6}/{n_steps}]  "
                        f"loss: {avg_loss:.4f}  "
                        f"(inv: {last['invariance']:.4f}  "
                        f"var: {last['variance']:.4f}  "
                        f"cov: {last['covariance']:.4f}  "
                        f"act: {last['action_cov']:.4f}  "
                        f"dtr: {last['dir_trans']:.4f})"
                    )

            if (step + 1) % 1000 == 0:
                all_embeds = self.predictor.action_encoder.get_all_embeddings()
                a0, a1, a2 = all_embeds[0], all_embeds[1], all_embeds[2]
                d01 = (1 - F.cosine_similarity(
                    a0.unsqueeze(0), a1.unsqueeze(0)
                )).item()
                d02 = (1 - F.cosine_similarity(
                    a0.unsqueeze(0), a2.unsqueeze(0)
                )).item()
                d12 = (1 - F.cosine_similarity(
                    a1.unsqueeze(0), a2.unsqueeze(0)
                )).item()
                print(f"  Action distances: "
                      f"L-R={d01:.4f}  L-F={d02:.4f}  R-F={d12:.4f}")

            if (step + 1) % save_every == 0:
                self.save_checkpoint(step + 1)

        print("\nFourRooms JEPA training complete.")
        self.save_checkpoint("fourrooms_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/jepa_{tag}.pt"
        torch.save({
            "online_encoder" : self.online_encoder.state_dict(),
            "target_encoder" : self.target_encoder.state_dict(),
            "predictor"      : self.predictor.state_dict(),
            "optimizer"      : self.optimizer.state_dict(),
            "step"           : self.step,
            "loss_log"       : self.loss_log,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.online_encoder.load_state_dict(ckpt["online_encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step     = ckpt["step"]
        self.loss_log = ckpt.get("loss_log", [])
        print(f"Checkpoint loaded: {path} (step {self.step})")


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=70_000)
    buffer.load("data/replay_buffer_fourrooms.pkl")

    trainer = JEPATrainer(
        latent_dim   = 256,
        n_actions    = 3,
        lr           = 1e-4,
        batch_size   = 256,
        ema_momentum = 0.99,
    )
    trainer.load_checkpoint("checkpoints/jepa_10000.pt")
    trainer.train(buffer, n_steps=20_000, log_every=200, save_every=5000)