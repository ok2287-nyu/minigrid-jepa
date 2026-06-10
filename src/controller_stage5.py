import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid

from encoder_v2 import Encoder, Predictor
from controller_bc import Controller
from data_collector_v2 import ReplayBuffer

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class Stage5Trainer:
    """
    Stage 5: Train controller through frozen world model.

    The controller is initialized from BC weights (Stage 3).
    It is then fine-tuned by imagining rollouts through the
    world model and minimizing distance to the goal state.

    Pipeline per training step:
        1. Sample random (start, goal) pair from buffer
        2. Build state_current and state_goal
        3. Roll out H steps through world model using soft action blend
        4. Loss = cosine distance between final imagined state and goal
        5. Backprop → only controller weights update
           encoder and predictor are frozen

    The soft blend keeps the pipeline differentiable:
        state_next = p0*pred(s,0) + p1*pred(s,1) + p2*pred(s,2)
        gradients flow back through blend → action_probs → controller
    """
    def __init__(
        self,
        jepa_checkpoint,
        bc_checkpoint,
        latent_dim   = 256,
        n_dirs       = 4,
        n_actions    = 3,
        horizon      = 3,
        lr           = 1e-3,
        batch_size   = 64,
        device       = None,
    ):
        self.device     = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.horizon    = horizon
        self.batch_size = batch_size
        self.n_actions  = n_actions
        self.state_dim  = latent_dim + n_dirs   # 260

        print(f"Training on:  {self.device}")
        print(f"Horizon:      {horizon} steps")
        print(f"State dim:    {self.state_dim}")

        # ── Load frozen encoder ───────────────────────────────────────
        self.encoder = Encoder(latent_dim).to(self.device)
        jepa_ckpt    = torch.load(
            jepa_checkpoint,
            map_location = self.device,
            weights_only = False
        )
        self.encoder.load_state_dict(jepa_ckpt["online_encoder"])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # ── Load frozen predictor ─────────────────────────────────────
        self.predictor = Predictor(latent_dim, n_actions).to(self.device)
        self.predictor.load_state_dict(jepa_ckpt["predictor"])
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()

        print(f"World model loaded and frozen.")

        # ── Load controller from BC checkpoint (warm start) ───────────
        self.controller = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = n_actions,
        ).to(self.device)

        bc_ckpt = torch.load(
            bc_checkpoint,
            map_location = self.device,
            weights_only = False
        )
        self.controller.load_state_dict(bc_ckpt["controller"])
        print(f"Controller loaded from BC checkpoint.")

        self.optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=lr
        )

        # Pre-compute action index tensors for soft blend
        self.all_actions = [
            torch.full(
                (batch_size,), i,
                dtype=torch.long, device=self.device
            )
            for i in range(n_actions)
        ]

        self.step     = 0
        self.loss_log = []

    def build_state(self, z, directions):
        """
        Build full state: [z (256) | direction_onehot (4)] = (260,)

        z:          (B, 256)
        directions: (B,) integer
        returns:    (B, 260)
        """
        dir_onehot = F.one_hot(
            directions, num_classes=4
        ).float()
        return torch.cat([z, dir_onehot], dim=-1)

    def soft_step(self, state, action_probs):
        """
        Differentiable world model step using soft action blend.

        state:        (B, 260) current full state
        action_probs: (B, 3)   softmax probabilities from controller

        Blends predictions for all three actions weighted by probs.
        Keeps pipeline differentiable — gradients flow back to controller.

        After blending, rebuilds a clean state:
            z part:   weighted blend of predicted z vectors
            dir part: argmax of blended direction logits → clean one-hot
        """
        B = state.shape[0]

        # Predict next state for each action
        states_next = []
        for a in range(self.n_actions):
            actions_a  = self.all_actions[a][:B]
            state_pred = self.predictor(state, actions_a)   # (B, 260)
            states_next.append(state_pred)

        # Stack: (B, n_actions, 260)
        states_stack = torch.stack(states_next, dim=1)

        # Weighted blend: (B, 1, 3) @ (B, 3, 260) → (B, 260)
        probs_exp  = action_probs.unsqueeze(1)              # (B, 1, 3)
        state_next = torch.bmm(probs_exp, states_stack).squeeze(1)  # (B, 260)

        # Rebuild clean state:
        # z part: normalize the blended latent
        z_next     = F.normalize(state_next[:, :256], dim=-1)  # (B, 256)

        # dir part: argmax of blended direction logits → clean one-hot
        # Note: argmax here is only for building next input state
        # gradients still flow through z_next
        dir_logits    = state_next[:, 256:]                 # (B, 4)
        predicted_dir = dir_logits.argmax(dim=-1)           # (B,)
        dir_onehot    = F.one_hot(
            predicted_dir, num_classes=4
        ).float()                                           # (B, 4)

        state_clean = torch.cat([z_next, dir_onehot], dim=-1)  # (B, 260)

        return state_clean

    def get_curriculum_steps(self, training_step, n_steps):
        progress = training_step / n_steps
        if progress < 0.20:
            return 1
        elif progress < 0.40:
            return 3
        elif progress < 0.60:
            return 5
        elif progress < 0.80:
            return 8
        else:
            return 12
        

    def compute_loss(self, state_start, state_goal):
        """
        Roll out controller through world model for H steps.
        Loss = cosine distance between final imagined state and goal.

        Progressive weighting: later steps weighted more
        because being close to goal at step H matters most.

        Entropy bonus: encourages exploration early in training
        prevents controller from collapsing to one action.
        """
        state_current  = state_start
        state_goal_norm = F.normalize(state_goal, dim=-1)

        total_dist  = 0.0
        entropy_sum = 0.0

        for h in range(self.horizon):
            # Controller proposes action probabilities
            probs = self.controller(state_current, state_goal_norm)  # (B, 3)

            # Entropy bonus — prevents action collapse
            entropy      = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            entropy_sum += entropy

            # Soft step through frozen world model
            state_current = self.soft_step(state_current, probs)

            # Distance to goal at this step
            dist = 1 - F.cosine_similarity(
                F.normalize(state_current, dim=-1),
                state_goal_norm,
                dim=-1
            ).mean()

            # Later steps weighted more
            weight      = (h + 1) / self.horizon
            total_dist += weight * dist

        dist_loss     = total_dist / self.horizon
        entropy_bonus = entropy_sum / self.horizon

        loss = dist_loss - 0.01 * entropy_bonus

        return loss, dist_loss.item(), entropy_bonus.item()

    def train_step(self, buffer, max_goal_steps=1):
        self.controller.train()

        B    = self.batch_size
        size = buffer.size
        idxs = np.random.randint(0, size, size=B)

        if max_goal_steps == 1:
            # One-step goals: use buffer directly
            # Correct action is already stored — no CEM needed
            start_obs = torch.tensor(
                buffer.obs[idxs], dtype=torch.float32
            ).to(self.device) / 255.0

            goal_obs = torch.tensor(
                buffer.next_obs[idxs], dtype=torch.float32
            ).to(self.device) / 255.0

            start_dirs = torch.tensor(
                buffer.directions[idxs], dtype=torch.long
            ).to(self.device)

            goal_dirs = torch.tensor(
                buffer.next_directions[idxs], dtype=torch.long
            ).to(self.device)

            correct_actions = torch.tensor(
                buffer.actions[idxs], dtype=torch.long
            ).to(self.device)

            with torch.no_grad():
                z_start = self.encoder(start_obs)
                z_goal  = self.encoder(goal_obs)

            state_start = self.build_state(z_start, start_dirs)
            state_goal  = self.build_state(z_goal,  goal_dirs)

        else:
            # Multi-step goals: chain real buffer transitions
            # Goal is exactly max_goal_steps real transitions away
            # Correct first action is always buffer.actions[start_idx]
            start_obs = torch.tensor(
                buffer.obs[idxs], dtype=torch.float32
            ).to(self.device) / 255.0

            start_dirs = torch.tensor(
                buffer.directions[idxs], dtype=torch.long
            ).to(self.device)

            # Correct action is the first action in the chain
            correct_actions = torch.tensor(
                buffer.actions[idxs], dtype=torch.long
            ).to(self.device)

            # Walk forward max_goal_steps by chaining next_obs
            current_idxs = idxs.copy()
            for _ in range(max_goal_steps):
                # Each buffer entry has 3 actions per state
                # Step forward by 3 to get next position's transitions
                current_idxs = (current_idxs + 3) % size

            goal_obs = torch.tensor(
                buffer.obs[current_idxs], dtype=torch.float32
            ).to(self.device) / 255.0

            goal_dirs = torch.tensor(
                buffer.directions[current_idxs], dtype=torch.long
            ).to(self.device)

            with torch.no_grad():
                z_start = self.encoder(start_obs)
                z_goal  = self.encoder(goal_obs)

            state_start = self.build_state(z_start, start_dirs)
            state_goal  = self.build_state(z_goal,  goal_dirs)

        # Controller predicts action
        action_probs = self.controller(state_start, state_goal)
        loss         = F.cross_entropy(action_probs, correct_actions)

        self.optimizer.zero_grad()
        loss.backward()

        total_grad = 0.0
        n_params   = 0
        for p in self.controller.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().mean().item()
                n_params   += 1
        avg_grad = total_grad / max(n_params, 1)

        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(), max_norm=1.0
        )
        self.optimizer.step()
        self.step += 1
        return loss.item(), avg_grad

    def train(self, buffer, n_steps=10_000, log_every=200,
          save_every=2000):
        print(f"\nStage 5: Curriculum Chain Training")
        print(f"Steps:      {n_steps}")
        print(f"Batch size: {self.batch_size}\n")

        losses = []
        grads  = []

        for step in range(n_steps):
            max_goal_steps = self.get_curriculum_steps(step, n_steps)
            loss, grad     = self.train_step(
                buffer, max_goal_steps=max_goal_steps
            )
            losses.append(loss)
            grads.append(grad)

            if (step + 1) % log_every == 0:
                avg_loss = np.mean(losses[-log_every:])
                avg_grad = np.mean(grads[-log_every:])
                print(
                    f"Step [{step+1:>5}/{n_steps}]  "
                    f"loss: {avg_loss:.4f}  "
                    f"grad: {avg_grad:.6f}  "
                    f"goal_steps: {max_goal_steps}"
                )

            if (step + 1) % save_every == 0:
                self.save_checkpoint(step + 1)

        print("\nCEM-guided training complete.")
        self.save_checkpoint("stage5_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_stage5_{tag}.pt"
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

    trainer = Stage5Trainer(
        jepa_checkpoint = "checkpoints/jepa_phase1_final.pt",
        bc_checkpoint   = "checkpoints/controller_bc_final.pt",
        latent_dim      = 256,
        n_dirs          = 4,
        n_actions       = 3,
        horizon         = 3,
        lr              = 1e-3,
        batch_size      = 64,
    )

    trainer.train(
        buffer,
        n_steps    = 10_000,
        log_every  = 200,
        save_every = 2000,
    )