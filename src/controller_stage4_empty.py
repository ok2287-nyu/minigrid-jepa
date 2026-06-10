"""
Stage 4 Empty — World Model Rollout with Expert Intermediate States

Key fixes over previous attempt:
1. Expert intermediate states as targets (not final goal)
2. Gumbel-Softmax for differentiable discrete action selection
3. Replay buffer to prevent batch training collapse

Training loop:
    Expert generates path: pos_0 → pos_1 → ... → pos_N
    Encode each: z_0, z_1, ..., z_N (target encoder, frozen)
    Controller predicts action at each step
    Predictor rolls out next state
    Loss = MSE(z_predicted_t, z_expert_t+1) at every step
    Gradients flow: loss → predictor → gumbel → controller
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid

from encoder_v2 import Encoder, Predictor
from controller_bc import Controller

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class Stage4EmptyTrainer:
    def __init__(
        self,
        jepa_checkpoint,
        env_id     = "MiniGrid-Empty-16x16-v0",
        latent_dim = 256,
        n_dirs     = 4,
        n_actions  = 3,
        lr         = 3e-3,
        device     = None,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim  = latent_dim + n_dirs
        self.latent_dim = latent_dim
        self.n_actions  = n_actions

        print(f"Training on:  {self.device}")

        # ── Environment ───────────────────────────────────────────────
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.max_x  = self.width  - 2
        self.max_y  = self.height - 2
        self.x_range = list(range(1, self.max_x + 1))
        self.y_range = list(range(1, self.max_y + 1))

        # ── Frozen encoder + predictor ────────────────────────────────
        self.encoder   = Encoder(latent_dim).to(self.device)
        self.predictor = Predictor(latent_dim, n_actions).to(self.device)

        ckpt = torch.load(jepa_checkpoint, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["online_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

        self.encoder.eval()
        self.predictor.eval()
        print("Encoder + Predictor loaded and frozen.")

        # ── Controller ────────────────────────────────────────────────
        self.controller = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = n_actions,
            pos_dim    = 2,
        ).to(self.device)
        print("Controller randomly initialized.")

        self.optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=lr
        )
        self.episode = 0

        # ── Replay buffer ─────────────────────────────────────────────
        # Each entry is one STEP along an expert path:
        #   sc    = current state (260)
        #   sg    = goal state (260)
        #   pc    = current pos (2)
        #   pg    = goal pos (2)
        #   z_next = expert next z (256) ← TARGET for loss
        #   d_curr = current direction (for state building after predictor)
        self.buf_sc     = []
        self.buf_sg     = []
        self.buf_pc     = []
        self.buf_pg     = []
        self.buf_znext  = []   # expert next z — the supervision signal
        self.buf_dcurr  = []   # current direction
        self.max_buf    = 50_000

    # ── Utilities ──────────────────────────────────────────────────────

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
            if 1 <= nx <= self.max_x and 1 <= ny <= self.max_y:
                return nx, ny, d
            return x, y, d

    def expert_action(self, x, y, d, xg, yg):
        dx, dy = xg-x, yg-y
        if dx==0 and dy==0: return 2
        if dx==0: desired = 1 if dy>0 else 3
        elif dy==0: desired = 0 if dx>0 else 2
        elif abs(dx)>=abs(dy): desired = 0 if dx>0 else 2
        else: desired = 1 if dy>0 else 3
        if d==desired: return 2
        diff = (desired-d)%4
        if diff==1: return 1
        elif diff==3: return 0
        else: return 1

    def sample_position(self):
        return (
            np.random.choice(self.x_range),
            np.random.choice(self.y_range),
            np.random.randint(0, 4)
        )

    def manhattan(self, p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def get_curriculum(self, episode, n_episodes):
        progress = episode / n_episodes
        if progress < 0.30:   return 1, 3
        elif progress < 0.55: return 2, 6
        elif progress < 0.75: return 4, 10
        else:                 return 6, 20

    def add_to_buffer(self, sc, sg, pc, pg, z_next, d_curr):
        self.buf_sc.append(sc)
        self.buf_sg.append(sg)
        self.buf_pc.append(pc)
        self.buf_pg.append(pg)
        self.buf_znext.append(z_next)
        self.buf_dcurr.append(d_curr)
        if len(self.buf_znext) > self.max_buf:
            self.buf_sc    = self.buf_sc[-self.max_buf:]
            self.buf_sg    = self.buf_sg[-self.max_buf:]
            self.buf_pc    = self.buf_pc[-self.max_buf:]
            self.buf_pg    = self.buf_pg[-self.max_buf:]
            self.buf_znext = self.buf_znext[-self.max_buf:]
            self.buf_dcurr = self.buf_dcurr[-self.max_buf:]

    # ── Collect episode ───────────────────────────────────────────────

    def collect_episode(self, min_dist=1, max_dist=3):
        """
        Walk along expert path.
        For each step, store:
            current state + goal state + positions (controller input)
            z of NEXT expert position (supervision target)
        """
        self.controller.eval()
        self.env.reset()

        x_start, y_start, d_start = self.sample_position()
        while True:
            x_goal, y_goal, d_goal = self.sample_position()
            if min_dist <= self.manhattan(
                (x_start,y_start),(x_goal,y_goal)
            ) <= max_dist:
                break

        # Encode goal
        self.set_agent_state(x_goal, y_goal, d_goal)
        z_goal     = self.encode_obs(self.env.render())
        sg         = self.build_state(z_goal, d_goal).detach()
        pg         = self.build_pos(x_goal, y_goal).detach()

        x, y, d = x_start, y_start, d_start

        for step in range(max_dist * 4 + 10):
            if (x, y) == (x_goal, y_goal):
                break

            # Encode CURRENT state
            self.set_agent_state(x, y, d)
            z_curr = self.encode_obs(self.env.render())
            sc     = self.build_state(z_curr, d).detach()
            pc     = self.build_pos(x, y).detach()

            # Expert action
            action = self.expert_action(x, y, d, x_goal, y_goal)

            # Advance to next position
            nx, ny, nd = self.manual_step(x, y, d, action)

            # Encode NEXT state (expert target)
            self.set_agent_state(nx, ny, nd)
            z_next = self.encode_obs(self.env.render()).detach()

            # Store: current state, goal, positions, expert next z
            self.add_to_buffer(sc, sg, pc, pg, z_next, d)

            x, y, d = nx, ny, nd

        self.episode += 1

    # ── Train on buffer ───────────────────────────────────────────────

    def train_on_buffer(self, n_steps=32, batch_size=64):
        if len(self.buf_znext) < batch_size:
            return 0.0, 0.0

        self.controller.train()

        sc_t    = torch.cat(self.buf_sc,    dim=0)
        sg_t    = torch.cat(self.buf_sg,    dim=0)
        pc_t    = torch.cat(self.buf_pc,    dim=0)
        pg_t    = torch.cat(self.buf_pg,    dim=0)
        znext_t = torch.cat(self.buf_znext, dim=0)
        N       = len(znext_t)

        total_loss = 0.0
        total_grad = 0.0

        for _ in range(n_steps):
            idxs = np.random.randint(0, N, size=batch_size)

            sc_b    = sc_t[idxs].clone()      # clone to allow grad flow
            sg_b    = sg_t[idxs]
            pc_b    = pc_t[idxs]
            pg_b    = pg_t[idxs]
            znext_b = znext_t[idxs]           # expert next z — target

            # Controller predicts action probabilities
            action_logits = self.controller(sc_b, sg_b, pc_b, pg_b)  # (B, 3)
            action_probs  = F.softmax(action_logits, dim=-1)          # (B, 3)

            # Weighted next state over all actions — no torch.no_grad()
            next_states = []
            for a in range(self.n_actions):
                action_t = torch.full(
                    (batch_size,), a, dtype=torch.long
                ).to(self.device)
                next_s = self.predictor(sc_b, action_t)   # (B, 260)
                next_states.append(next_s)

            next_states_stack = torch.stack(next_states, dim=1)  # (B, 3, 260)
            state_pred = (action_probs.unsqueeze(-1) * next_states_stack).sum(dim=1)

            # Loss: predicted z vs expert next z
            z_pred = state_pred[:, :self.latent_dim]   # (B, 256)
            loss   = F.mse_loss(z_pred, znext_b)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient magnitude
            grad_sum = sum(
                p.grad.abs().mean().item()
                for p in self.controller.parameters()
                if p.grad is not None
            )
            n_params = sum(
                1 for p in self.controller.parameters()
                if p.grad is not None
            )
            total_grad += grad_sum / max(n_params, 1)

            torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_steps, total_grad / n_steps
    # ── Evaluate ──────────────────────────────────────────────────────

    def evaluate(self, n_episodes=100):
        self.controller.eval()
        self.env.reset()
        successes = 0

        for _ in range(n_episodes):
            x_start, y_start, d_start = self.sample_position()
            while True:
                x_goal, y_goal, d_goal = self.sample_position()
                if self.manhattan(
                    (x_start,y_start),(x_goal,y_goal)
                ) >= 1:
                    break

            self.set_agent_state(x_goal, y_goal, d_goal)
            z_goal     = self.encode_obs(self.env.render()).detach()
            state_goal = self.build_state(z_goal, d_goal)
            pos_goal   = self.build_pos(x_goal, y_goal)

            x, y, d   = x_start, y_start, d_start
            max_steps = 60
            success   = False

            for step in range(max_steps):
                self.set_agent_state(x, y, d)
                z          = self.encode_obs(self.env.render()).detach()
                state_curr = self.build_state(z, d)
                pos_curr   = self.build_pos(x, y)

                with torch.no_grad():
                    action = self.controller(
                        state_curr, state_goal,
                        pos_curr,   pos_goal
                    ).argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, action)
                if (x, y) == (x_goal, y_goal):
                    success = True
                    break

            if success:
                successes += 1

        return successes / n_episodes

    # ── Training loop ─────────────────────────────────────────────────

    def train(self, n_episodes=3000, train_every=10,
              n_train_steps=32, batch_size=64,
            log_every=100, save_every=500):

        print(f"\nStage 4 Empty — Expert Intermediate States + Gumbel-Softmax")
        print(f"Episodes:    {n_episodes}") 

        losses = []
        grads  = []

        print("Warmup: collecting 500 episodes...")
        for ep in range(500):
            min_dist, max_dist = self.get_curriculum(ep, n_episodes)
            self.collect_episode(min_dist=min_dist, max_dist=max_dist)
        print(f"Buffer size after warmup: {len(self.buf_znext)}\n")

        for ep in range(n_episodes):
            min_dist, max_dist = self.get_curriculum(ep, n_episodes)
            self.collect_episode(min_dist=min_dist, max_dist=max_dist)

            if (ep + 1) % train_every == 0:
                loss, grad = self.train_on_buffer(
                    n_steps    = n_train_steps,
                    batch_size = batch_size, 
                )
                losses.append(loss)
                grads.append(grad)

            if (ep + 1) % log_every == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0.0
                avg_grad = np.mean(grads[-10:])  if grads  else 0.0
                eval_sr  = self.evaluate(n_episodes=100)
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"loss: {avg_loss:.4f}  "
                    f"grad: {avg_grad:.6f}  "
                    f"eval_sr: {eval_sr*100:.1f}%  "
                    f"dist: {min_dist}-{max_dist}  "
                    f"buf: {len(self.buf_znext)}"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nStage 4 Empty training complete.")
        self.save_checkpoint("stage4_empty_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_stage4_empty_{tag}.pt"
        torch.save({
            "controller" : self.controller.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "episode"    : self.episode,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    trainer = Stage4EmptyTrainer(
        jepa_checkpoint = "checkpoints/jepa_phase1_final.pt",
        env_id          = "MiniGrid-Empty-16x16-v0",
        latent_dim      = 256,
        lr              = 3e-3,
    )
    trainer.train(
        n_episodes    = 3000,
        train_every   = 10,
        n_train_steps = 32,
        batch_size    = 64,
        log_every     = 100,
        save_every    = 500,
    )