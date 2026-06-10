"""
Joint Training: High-level + Low-level through World Model

Both controllers trained simultaneously via latent rollout.
Gradients flow: loss → predictor → low-level → embedder → high-level

Architecture:
    High-level: (state_curr, state_goal, pos_curr, pos_goal) → subgoal_pos (x,y)
    Embedder:   (subgoal_pos, dir) → z_subgoal  (differentiable)
    Low-level:  (state_curr, state_subgoal, pos_curr, pos_subgoal) → action_probs
    Predictor:  (state_curr, action) → state_next  (frozen)
    Loss:       MSE(state_final[:256], z_goal)
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
from collections import deque 
import sys
sys.path.insert(0, 'src')
os.chdir(Path(__file__).parent.parent)

from encoder_v2 import Encoder, Predictor
from controller_bc import Controller
from controller_highlevel import HighLevelController, bfs_path, find_doorways, extract_subgoals
from position_embedder import PositionEmbedder


class JointTrainer:
    def __init__(
        self,
        jepa_checkpoint,
        highlevel_checkpoint,
        lowlevel_checkpoint,
        embedder_checkpoint,
        env_id     = "MiniGrid-FourRooms-v0",
        latent_dim = 256,
        n_dirs     = 4,
        lr         = 1e-3,
        device     = None,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = latent_dim + n_dirs   # 260
        self.latent_dim = latent_dim
        print(f"Training on: {self.device}")

        # ── Environment ───────────────────────────────────────────────
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.max_x  = self.width  - 2
        self.max_y  = self.height - 2

        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        print(f"Walkable cells: {len(self.walkable)}")

        # ── Frozen encoder ────────────────────────────────────────────
        self.encoder = Encoder(latent_dim).to(self.device)
        ckpt = torch.load(jepa_checkpoint, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["online_encoder"])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # ── Frozen predictor ──────────────────────────────────────────
        self.predictor = Predictor(latent_dim, 3).to(self.device)
        self.predictor.load_state_dict(ckpt["predictor"])
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()
        print("Encoder + Predictor loaded and frozen.")

        # ── Frozen position embedder ──────────────────────────────────
        self.embedder = PositionEmbedder(pos_dim=2, latent_dim=latent_dim).to(self.device)
        emb_ckpt = torch.load(embedder_checkpoint, map_location=self.device, weights_only=False)
        self.embedder.load_state_dict(emb_ckpt["embedder"])
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.embedder.eval()
        print("Position embedder loaded and frozen.")

        # ── Trainable low-level controller ────────────────────────────
        self.lowlevel = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = 3,
            pos_dim    = 2,
        ).to(self.device)
        ll_ckpt = torch.load(lowlevel_checkpoint, map_location=self.device, weights_only=False)
        self.lowlevel.load_state_dict(ll_ckpt["controller"])
        print(f"Low-level loaded from: {lowlevel_checkpoint}")

        # ── Trainable high-level controller ───────────────────────────
        self.highlevel = HighLevelController(
            state_dim  = self.state_dim,
            hidden_dim = 256,
            pos_dim    = 2,
        ).to(self.device)
        hl_ckpt = torch.load(highlevel_checkpoint, map_location=self.device, weights_only=False)
        self.highlevel.load_state_dict(hl_ckpt["highlevel"])
        print(f"High-level loaded from: {highlevel_checkpoint}")

        # ── Joint optimizer ───────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            list(self.highlevel.parameters()) +
            list(self.lowlevel.parameters()),
            lr = lr
        )
        self.episode = 0

    # ── Utilities ──────────────────────────────────────────────────────

    def _find_walkable(self):
        grid = self.env.unwrapped.grid
        cells = []
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                cell = grid.get(x, y)
                if cell is None or cell.type in ('goal', 'door'):
                    cells.append((x, y))
        return cells

    def preprocess(self, frame):
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def encode_obs(self, frame):
        with torch.no_grad():
            return self.encoder(self.preprocess(frame))

    def build_state(self, z, d):
        dir_oh = F.one_hot(
            torch.tensor([d], dtype=torch.long).to(self.device),
            num_classes=4
        ).float()
        return torch.cat([z, dir_oh], dim=-1)

    def build_pos(self, x, y):
        return torch.tensor(
            [[x / self.max_x, y / self.max_y]],
            dtype=torch.float32
        ).to(self.device)

    def set_agent_state(self, x, y, d):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = d

    def manual_step(self, x, y, d, action):
        if action == 0: return x, y, (d-1)%4
        elif action == 1: return x, y, (d+1)%4
        else:
            dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
            nx, ny = x+dx, y+dy
            if (nx,ny) in self.walkable_set:
                return nx, ny, d
            return x, y, d

    def embed_subgoal(self, subgoal_pos, d_goal):
        """
        Convert high-level's predicted (x,y) position to z_approx.
        Differentiable — gradients flow back to high-level.

        subgoal_pos: (1, 2) normalized position from high-level
        d_goal:      integer direction
        returns:     (1, 256) approximate z vector
        """
        dir_oh = F.one_hot(
            torch.tensor([d_goal], dtype=torch.long).to(self.device),
            num_classes=4
        ).float()   # (1, 4)

        # Concatenate pos + direction → (1, 6)
        embedder_input = torch.cat([subgoal_pos, dir_oh], dim=-1)

        # Embedder is frozen but gradients flow through it to subgoal_pos
        # We need to temporarily allow gradients through embedder
        z_approx = self.embedder(embedder_input)   # (1, 256)
        return z_approx

    def predict_weighted(self, state_t, d_t):
        """
        Differentiable next state prediction via weighted sum.
        Gradients flow through action_probs to low-level.
        """
        next_states = []
        for a in range(3):
            action_t = torch.tensor([a], dtype=torch.long).to(self.device)
            next_s   = self.predictor(state_t, action_t)
            next_states.append(next_s)

        return torch.stack(next_states, dim=1)   # (1, 3, 260)

    # ── Joint rollout episode ──────────────────────────────────────────

    def run_episode(self, n_lowlevel_steps=6):
        """
        One joint training episode.

        1. Sample start and goal
        2. High-level predicts subgoal position
        3. Embedder converts subgoal position → z_subgoal (differentiable)
        4. Low-level rolls out N steps toward subgoal through predictor
        5. Loss = MSE(final z, z_goal)
        6. Gradients flow to both high-level and low-level
        """
        self.highlevel.train()
        self.lowlevel.train()
        self.env.reset()

        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)

        # Sample start and goal — prefer paths that cross rooms
        for _ in range(200):
            start = self.walkable[np.random.randint(len(self.walkable))]
            goal  = self.walkable[np.random.randint(len(self.walkable))]
            if start == goal: continue
            path = bfs_path(start, goal, self.walkable_set)
            if path and len(path)-1 >= 3:
                break
        else:
            return None

        x_start, y_start = start
        x_goal,  y_goal  = goal
        d_start = np.random.randint(0, 4)
        d_goal  = np.random.randint(0, 4)

        # Encode start and goal (no grad — encoder frozen)
        self.set_agent_state(x_start, y_start, d_start)
        z_start = self.encode_obs(self.env.render())
        state_start = self.build_state(z_start, d_start)

        self.set_agent_state(x_goal, y_goal, d_goal)
        z_goal_vec  = self.encode_obs(self.env.render())
        state_goal  = self.build_state(z_goal_vec, d_goal)
        pos_goal    = self.build_pos(x_goal, y_goal)
        pos_start   = self.build_pos(x_start, y_start)

        # ── Step 1: High-level predicts subgoal ───────────────────────
        subgoal_pos = self.highlevel(
            state_start, state_goal, pos_start, pos_goal
        )   # (1, 2) — has gradients

        # ── Step 2: Embedder converts subgoal → z (differentiable) ───
        # Temporarily unfreeze embedder for gradient flow
        for p in self.embedder.parameters():
            p.requires_grad = True

        z_subgoal = self.embed_subgoal(subgoal_pos, d_goal)  # (1, 256)
        dir_oh_sg = F.one_hot(
            torch.tensor([d_goal], dtype=torch.long).to(self.device),
            num_classes=4
        ).float()
        state_subgoal = torch.cat([z_subgoal, dir_oh_sg], dim=-1)  # (1, 260)

        # Refreeze embedder weights (gradients flow through but weights don't update)
        for p in self.embedder.parameters():
            p.requires_grad = False

        # ── Step 3: Low-level rolls out toward subgoal ────────────────
        state_t = state_start.detach().clone()
        x_t, y_t, d_t = x_start, y_start, d_start

        episode_losses = []

        for step in range(n_lowlevel_steps):
            pos_curr = self.build_pos(x_t, y_t)

            # Low-level action probabilities
            action_logits = self.lowlevel(
                state_t, state_subgoal, pos_curr, pos_goal
            )   # (1, 3)
            action_probs = F.softmax(action_logits, dim=-1)   # (1, 3)

            # Weighted next state (differentiable)
            next_states_stack = self.predict_weighted(state_t, d_t)  # (1, 3, 260)
            state_t1 = (action_probs.unsqueeze(-1) * next_states_stack).sum(dim=1)

            # Loss at every step: predicted z vs goal z
            z_predicted = state_t1[:, :self.latent_dim]
            loss_t = F.mse_loss(z_predicted, z_goal_vec.detach())
            episode_losses.append(loss_t)

            # Advance position using argmax (for position tracking)
            with torch.no_grad():
                best_action = action_probs.argmax(dim=-1).item()
            x_t, y_t, d_t = self.manual_step(x_t, y_t, d_t, best_action)

            # Next state
            state_t = state_t1
 
        # ── Update both controllers ───────────────────────────────────
        if episode_losses:
            # Latent rollout loss
            loss_rollout = torch.stack(episode_losses).mean()

            # High-level supervised loss — BFS subgoal as target
            doorways    = find_doorways(self.env.unwrapped.grid, self.width, self.height)
            doorway_set = set(doorways)
            subgoals    = extract_subgoals(path, doorway_set, (x_goal, y_goal))
            sg_x, sg_y  = subgoals[0]
            tgt_pos     = torch.tensor(
                [[sg_x / self.max_x, sg_y / self.max_y]],
                dtype=torch.float32
            ).to(self.device)
            loss_hl = F.mse_loss(subgoal_pos, tgt_pos)

            # Combined loss — supervised anchors + rollout coordination
            total_loss = 0.1 * loss_rollout + 1.0 * loss_hl

            self.optimizer.zero_grad()
            total_loss.backward()

            # Check gradient magnitudes
            hl_grad = np.mean([
                p.grad.abs().mean().item()
                for p in self.highlevel.parameters()
                if p.grad is not None
            ])
            ll_grad = np.mean([
                p.grad.abs().mean().item()
                for p in self.lowlevel.parameters()
                if p.grad is not None
            ])

            torch.nn.utils.clip_grad_norm_(
                list(self.highlevel.parameters()) +
                list(self.lowlevel.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            self.episode += 1
            return total_loss.item(), hl_grad, ll_grad

        return None

    # ── Evaluate full hierarchical system ─────────────────────────────

    def evaluate(self, n_episodes=50):
        self.highlevel.eval()
        self.lowlevel.eval()
        self.env.reset()
        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        successes = 0

        for _ in range(n_episodes):
            for _ in range(200):
                start = self.walkable[np.random.randint(len(self.walkable))]
                goal  = self.walkable[np.random.randint(len(self.walkable))]
                if start == goal: continue
                path = bfs_path(start, goal, self.walkable_set)
                if path and len(path)-1 >= 1: break
            else:
                continue

            x_start, y_start = start
            x_goal,  y_goal  = goal
            d_start = np.random.randint(0, 4)
            d_goal  = np.random.randint(0, 4)

            self.set_agent_state(x_goal, y_goal, d_goal)
            z_goal_v   = self.encode_obs(self.env.render()).detach()
            sg_final   = self.build_state(z_goal_v, d_goal)
            pg_final   = self.build_pos(x_goal, y_goal)

            x, y, d    = x_start, y_start, d_start
            max_steps  = (len(path)-1) * 6 + 20
            success    = False
            current_subgoal_pos   = None
            current_subgoal_state = None
            subgoal_steps         = 0
            max_subgoal_steps     = 20

            for step in range(max_steps):
                self.set_agent_state(x, y, d)
                z_curr     = self.encode_obs(self.env.render()).detach()
                state_curr = self.build_state(z_curr, d)
                pos_curr   = self.build_pos(x, y)

                if current_subgoal_pos is None or subgoal_steps >= max_subgoal_steps:
                    with torch.no_grad():
                        pred = self.highlevel(state_curr, sg_final, pos_curr, pg_final)
                    sg_x = int(round(pred[0,0].item() * self.max_x))
                    sg_y = int(round(pred[0,1].item() * self.max_y))
                    sg_x = max(1, min(self.max_x, sg_x))
                    sg_y = max(1, min(self.max_y, sg_y))
                    if (sg_x, sg_y) not in self.walkable_set:
                        nearest = min(
                            self.walkable_set,
                            key=lambda p: (p[0]-sg_x)**2 + (p[1]-sg_y)**2
                        )
                        sg_x, sg_y = nearest
                    current_subgoal_pos = (sg_x, sg_y)
                    self.set_agent_state(sg_x, sg_y, d_goal)
                    z_sg = self.encode_obs(self.env.render()).detach()
                    current_subgoal_state = self.build_state(z_sg, d_goal)
                    subgoal_steps = 0

                pos_subgoal = self.build_pos(*current_subgoal_pos)
                with torch.no_grad():
                    action = self.lowlevel(
                        state_curr, current_subgoal_state,
                        pos_curr, pos_subgoal
                    ).argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, action)
                subgoal_steps += 1

                if (x,y) == (x_goal, y_goal):
                    success = True
                    break
                if (x,y) == current_subgoal_pos:
                    current_subgoal_pos = None
                    subgoal_steps = 0

            if success:
                successes += 1

        return successes / n_episodes

    # ── Training loop ─────────────────────────────────────────────────

    def train(self, n_episodes=3000, log_every=100, save_every=500):
        print(f"\nJoint Training: High-level + Low-level")
        print(f"Episodes: {n_episodes}\n")

        losses   = []
        hl_grads = []
        ll_grads = []

        for ep in range(n_episodes):
            result = self.run_episode()
            if result is None:
                continue

            loss, hl_grad, ll_grad = result
            losses.append(loss)
            hl_grads.append(hl_grad)
            ll_grads.append(ll_grad)

            if (ep + 1) % log_every == 0:
                avg_loss    = np.mean(losses[-log_every:])
                avg_hl_grad = np.mean(hl_grads[-log_every:])
                avg_ll_grad = np.mean(ll_grads[-log_every:])
                eval_sr     = self.evaluate(n_episodes=50)
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"loss: {avg_loss:.4f}  "
                    f"hl_grad: {avg_hl_grad:.6f}  "
                    f"ll_grad: {avg_ll_grad:.6f}  "
                    f"eval_sr: {eval_sr*100:.1f}%"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nJoint training complete.")
        self.save_checkpoint("joint_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/joint_{tag}.pt"
        torch.save({
            "highlevel" : self.highlevel.state_dict(),
            "lowlevel"  : self.lowlevel.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "episode"   : self.episode,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    trainer = JointTrainer(
        jepa_checkpoint      = "checkpoints/jepa_fourrooms_final.pt",
        highlevel_checkpoint = "checkpoints/highlevel_hierarchical_final.pt",
        lowlevel_checkpoint  = "checkpoints/controller_fourrooms_fourrooms_final.pt",
        embedder_checkpoint  = "checkpoints/position_embedder.pt",
        env_id               = "MiniGrid-FourRooms-v0",
        latent_dim           = 256,
        lr                   = 1e-3,
    )
    trainer.train(
        n_episodes  = 3000,
        log_every   = 100,
        save_every  = 500,
    )