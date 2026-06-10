import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os

from encoder_v2 import Encoder
from data_collector_v2 import ReplayBuffer

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class Controller(nn.Module):
    """
    Navigation controller — Stage 3 Behavioural Cloning.

    Input:  state_current (260,) + state_goal (260,) = (520,)
            where state = [z (256) | direction_onehot (4)]

    Output: action probabilities (3,) via softmax
            0 = left, 1 = right, 2 = forward

    Trained to imitate expert actions from the buffer.
    Encoder is completely frozen — only controller weights update.
    """
    def __init__(self, state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.pos_dim   = pos_dim

        # input = state_current + state_goal + pos_current + pos_goal
        input_dim = state_dim * 2 + pos_dim * 2   # 260*2 + 2*2 = 524

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),        # 524 → 512
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 → 256
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_actions),   # 256 → 3
        )

    def forward(self, state_current, state_goal, pos_current, pos_goal):
        """
        state_current: (B, 260) = [z_current | dir_current_onehot]
        state_goal:    (B, 260) = [z_goal    | dir_goal_onehot   ]
        pos_current:   (B, 2)   = [x_curr/14, y_curr/14]
        pos_goal:      (B, 2)   = [x_goal/14, y_goal/14]
        returns:       (B, 3)   action probabilities
        """
        x = torch.cat([
            state_current, state_goal,
            pos_current,   pos_goal
        ], dim=-1)                                           # (B, 524)
        return self.net(x)


class BCTrainer:
    """
    Stage 3: Behavioural Cloning

    For each buffer entry (obs_t, action, obs_t+1, direction, next_direction):

        z_current  = encoder(obs_t)                    # frozen
        z_goal     = encoder(obs_t+1)                  # frozen
        dir_curr   = one_hot(direction, 4)
        dir_goal   = one_hot(next_direction, 4)

        state_current = [z_current | dir_curr]         # (260,)
        state_goal    = [z_goal    | dir_goal]         # (260,)

        probs = controller(state_current, state_goal)  # (3,)
        loss  = cross_entropy(probs, action)

    Only controller weights update. Encoder frozen.
    """
    def __init__(
        self,
        checkpoint_path,
        latent_dim  = 256,
        n_dirs      = 4,
        n_actions   = 3,
        lr          = 1e-3,
        batch_size  = 256,
        device      = None,
    ):
        self.device     = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.state_dim  = latent_dim + n_dirs   # 260

        print(f"Training on:  {self.device}")
        print(f"State dim:    {self.state_dim} (z={latent_dim} + dir={n_dirs} + pos=2)")

        # ── Frozen encoder ────────────────────────────────────────────
        self.encoder = Encoder(latent_dim).to(self.device)
        ckpt = torch.load(
            checkpoint_path,
            map_location = self.device,
            weights_only = False
        )
        self.encoder.load_state_dict(ckpt["online_encoder"])

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        print(f"Encoder loaded and frozen.")

        # ── Controller (only this trains) ─────────────────────────────
        self.controller = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = n_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=lr
        )

        n_params = sum(p.numel() for p in self.controller.parameters())
        print(f"Controller parameters: {n_params:,}")

        self.step = 0

    def build_state(self, z, directions):
        """
        Build full state by concatenating z with direction one-hot.

        z:          (B, 256)
        directions: (B,) integer direction indices
        returns:    (B, 260)
        """
        dir_onehot = F.one_hot(
            directions, num_classes=4
        ).float()                                    # (B, 4)
        return torch.cat([z, dir_onehot], dim=-1)   # (B, 260)

    def train_step(self, buffer):
        """One BC gradient update."""
        self.controller.train()

        # ── Sample from buffer ────────────────────────────────────────
        idxs = np.random.randint(0, buffer.size, size=self.batch_size)

        obs_t   = torch.tensor(
            buffer.obs[idxs], dtype=torch.float32
        ).to(self.device) / 255.0                    # (B, 3, 64, 64)

        obs_t1  = torch.tensor(
            buffer.next_obs[idxs], dtype=torch.float32
        ).to(self.device) / 255.0                    # (B, 3, 64, 64)

        actions = torch.tensor(
            buffer.actions[idxs], dtype=torch.long
        ).to(self.device)                            # (B,)

        directions = torch.tensor(
            buffer.directions[idxs], dtype=torch.long
        ).to(self.device)                            # (B,)

        next_directions = torch.tensor(
            buffer.next_directions[idxs], dtype=torch.long
        ).to(self.device)                            # (B,)

        # ── Encode — frozen, no gradients ────────────────────────────
        with torch.no_grad():
            z_current = self.encoder(obs_t)          # (B, 256)
            z_goal    = self.encoder(obs_t1)         # (B, 256)

        # ── Build full states ─────────────────────────────────────────
        # ── Build full states ─────────────────────────────────────────
        state_current = self.build_state(z_current, directions)       # (B, 260)
        state_goal    = self.build_state(z_goal,    next_directions)  # (B, 260)

        # ── Explicit positions ────────────────────────────────────────
        pos_current = torch.tensor(
            buffer.positions[idxs], dtype=torch.float32
        ).to(self.device) / 14.0                         # (B, 2) normalized

        # Goal position: next_obs position
        # For one-step goals, next position is computable from action
        # We use next_obs positions stored implicitly via next_directions
        # For simplicity: goal pos = pos_current + action displacement
        # But buffer.positions only stores current pos
        # So we encode goal pos from the goal direction + position heuristic
        # Simplest: use same position for turns, shifted for forward
        pos_goal = pos_current.clone()
        fwd_mask = actions == 2
        dir_vals = directions.cpu().numpy()

        # Direction vectors: East=(+1,0), South=(0,+1), West=(-1,0), North=(0,-1)
        dir_dx = torch.tensor([1, 0, -1, 0], dtype=torch.float32)
        dir_dy = torch.tensor([0, 1,  0,-1], dtype=torch.float32)

        dx = dir_dx[directions.cpu()] / 14.0   # (B,)
        dy = dir_dy[directions.cpu()] / 14.0   # (B,)

        fwd_mask_cpu = fwd_mask.cpu()
        pos_goal[fwd_mask, 0] = (pos_current[fwd_mask, 0] + dx[fwd_mask_cpu].to(self.device))
        pos_goal[fwd_mask, 1] = (pos_current[fwd_mask, 1] + dy[fwd_mask_cpu].to(self.device))
        pos_goal = pos_goal.clamp(0, 1)

        # ── Controller forward pass ───────────────────────────────────
        action_probs = self.controller(
            state_current, state_goal,
            pos_current,   pos_goal
        )          

        # ── Cross entropy loss ────────────────────────────────────────
        loss = F.cross_entropy(action_probs, actions)

        # ── Update controller only ────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        self.step += 1
        return loss.item()

    def evaluate(self, buffer, n_samples=5000):
        """
        Measure action prediction accuracy on random samples.
        Key metric for BC quality.
        """
        self.controller.eval()

        idxs = np.random.randint(0, buffer.size, size=n_samples)

        obs_t   = torch.tensor(
            buffer.obs[idxs], dtype=torch.float32
        ).to(self.device) / 255.0

        obs_t1  = torch.tensor(
            buffer.next_obs[idxs], dtype=torch.float32
        ).to(self.device) / 255.0

        actions = torch.tensor(
            buffer.actions[idxs], dtype=torch.long
        ).to(self.device)

        directions = torch.tensor(
            buffer.directions[idxs], dtype=torch.long
        ).to(self.device)

        next_directions = torch.tensor(
            buffer.next_directions[idxs], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            z_current = self.encoder(obs_t)
            z_goal    = self.encoder(obs_t1)

            state_current = self.build_state(z_current, directions)
            state_goal    = self.build_state(z_goal,    next_directions)

            pos_current = torch.tensor(
                buffer.positions[idxs], dtype=torch.float32
            ).to(self.device) / 14.0

            pos_goal = pos_current.clone()
            fwd_mask = actions == 2
            dir_dx = torch.tensor([1, 0, -1, 0], dtype=torch.float32)
            dir_dy = torch.tensor([0, 1,  0,-1], dtype=torch.float32)
            dx = dir_dx[directions.cpu()] / 14.0
            dy = dir_dy[directions.cpu()] / 14.0
            fwd_mask_cpu = fwd_mask.cpu()
            pos_goal[fwd_mask, 0] = (pos_current[fwd_mask, 0] + dx[fwd_mask_cpu].to(self.device))
            pos_goal[fwd_mask, 1] = (pos_current[fwd_mask, 1] + dy[fwd_mask_cpu].to(self.device))
            pos_goal = pos_goal.clamp(0, 1)

            action_probs = self.controller(
                state_current, state_goal,
                pos_current,   pos_goal
            )
            predicted    = action_probs.argmax(dim=-1)

        # Overall accuracy
        accuracy = (predicted == actions).float().mean().item()

        # Per-action accuracy
        action_names = ["Left", "Right", "Forward"]
        per_action   = {}
        for a, name in enumerate(action_names):
            mask = actions == a
            if mask.sum() > 0:
                per_action[name] = (
                    predicted[mask] == actions[mask]
                ).float().mean().item()

        return accuracy, per_action

    def train(self, buffer, n_steps=5000, log_every=200,
              save_every=1000):
        print(f"\nStage 3: Behavioural Cloning")
        print(f"Steps:      {n_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Buffer:     {len(buffer)} transitions\n")

        losses = []

        for step in range(n_steps):
            loss = self.train_step(buffer)
            losses.append(loss)

            if (step + 1) % log_every == 0:
                avg_loss = np.mean(losses[-log_every:])
                accuracy, per_action = self.evaluate(buffer)
                print(
                    f"Step [{step+1:>5}/{n_steps}]  "
                    f"loss: {avg_loss:.4f}  "
                    f"accuracy: {accuracy*100:.1f}%  "
                    f"(L:{per_action['Left']*100:.0f}%  "
                    f"R:{per_action['Right']*100:.0f}%  "
                    f"F:{per_action['Forward']*100:.0f}%)"
                )

            if (step + 1) % save_every == 0:
                self.save_checkpoint(step + 1)

        print("\nBC training complete.")
        self.save_checkpoint("bc_final")

        # Final evaluation
        accuracy, per_action = self.evaluate(buffer, n_samples=10_000)
        print(f"\nFinal accuracy: {accuracy*100:.1f}%")
        for name, acc in per_action.items():
            print(f"  {name}: {acc*100:.1f}%")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_{tag}.pt"
        torch.save({
            "controller" : self.controller.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "step"       : self.step,
        }, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.controller.load_state_dict(ckpt["controller"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        print(f"Loaded: {path} (step {self.step})")


if __name__ == "__main__":

    buffer = ReplayBuffer(capacity=200_000)
    buffer.load("data/replay_buffer_phase1.pkl")
    print(f"Buffer: {len(buffer)} transitions")

    trainer = BCTrainer(
        checkpoint_path = "checkpoints/jepa_phase1_final.pt",
        latent_dim      = 256,
        n_dirs          = 4,
        n_actions       = 3,
        lr              = 1e-3,
        batch_size      = 256,
    )
    trainer.train(
        buffer,
        n_steps    = 5000,
        log_every  = 200,
        save_every = 1000,
    )