import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid

from encoder_v2 import Encoder
from controller_bc import Controller

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class DAggerTrainer:
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
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = latent_dim + n_dirs
        self.n_actions = n_actions

        print(f"Training on:  {self.device}")

        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width   = self.env.unwrapped.width
        self.height  = self.env.unwrapped.height
        self.x_range = list(range(1, self.width  - 1))
        self.y_range = list(range(1, self.height - 1))
        self.max_x   = self.width  - 2
        self.max_y   = self.height - 2

        self.encoder = Encoder(latent_dim).to(self.device)
        jepa_ckpt    = torch.load(jepa_checkpoint, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(jepa_ckpt["online_encoder"])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        print(f"Encoder loaded and frozen.")

        self.controller = Controller(
            state_dim=self.state_dim, hidden_dim=512,
            n_actions=n_actions, pos_dim=2,
        ).to(self.device)
        print("Controller randomly initialized.")

        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=lr)
        self.episode = 0

        # Replay buffer — stores tensors directly
        self.buf_sc = []
        self.buf_sg = []
        self.buf_pc = []
        self.buf_pg = []
        self.buf_lb = []
        self.max_buf = 50_000

    # ── Utilities ──────────────────────────────────────────────────────

    def preprocess(self, frame):
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def encode_obs(self, frame):
        with torch.no_grad():
            return self.encoder(self.preprocess(frame))

    def build_state(self, z, direction):
        dir_oh = F.one_hot(torch.tensor([direction], dtype=torch.long).to(self.device), num_classes=4).float()
        return torch.cat([z, dir_oh], dim=-1)

    def build_pos(self, x, y):
        return torch.tensor([[x / self.max_x, y / self.max_y]], dtype=torch.float32).to(self.device)

    def manual_step(self, x, y, d, action):
        if action == 0: return x, y, (d-1)%4
        elif action == 1: return x, y, (d+1)%4
        else:
            dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
            nx, ny = x+dx, y+dy
            if 1 <= nx <= self.max_x and 1 <= ny <= self.max_y: return nx, ny, d
            return x, y, d

    def set_agent_state(self, x, y, direction):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def sample_position(self):
        return (np.random.choice(self.x_range),
                np.random.choice(self.y_range),
                np.random.randint(0, 4))

    def manhattan(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def expert_action(self, x, y, d, x_goal, y_goal):
        dx, dy = x_goal-x, y_goal-y
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

    # def get_goal_dist_range(self, episode, n_episodes):
    #     progress = episode / n_episodes
    #     if progress < 0.30:   return 1, 4
    #     elif progress < 0.55: return 2, 7
    #     elif progress < 0.75: return 4, 12
    #     else:                 return 6, 26
    def get_goal_dist_range(self, episode, n_episodes):
        return 6, 26

    # ── Collect one episode into replay buffer ─────────────────────────

    def collect_episode(self, min_dist=1, max_dist=4, beta=1.0):
        self.controller.eval()
        self.env.reset()

        x_start, y_start, d_start = self.sample_position()
        while True:
            x_goal, y_goal, d_goal = self.sample_position()
            if min_dist <= self.manhattan((x_start,y_start),(x_goal,y_goal)) <= max_dist:
                break

        self.set_agent_state(x_goal, y_goal, d_goal)
        z_goal     = self.encode_obs(self.env.render())
        sg         = self.build_state(z_goal, d_goal).detach()
        pg         = self.build_pos(x_goal, y_goal).detach()

        x, y, d    = x_start, y_start, d_start
        success    = False
        new_samples = 0

        for step in range(max_dist * 4 + 10):
            self.set_agent_state(x, y, d)
            z          = self.encode_obs(self.env.render())
            sc         = self.build_state(z, d).detach()
            pc         = self.build_pos(x, y).detach()
            expert     = self.expert_action(x, y, d, x_goal, y_goal)

            # Store in replay buffer
            self.buf_sc.append(sc)
            self.buf_sg.append(sg)
            self.buf_pc.append(pc)
            self.buf_pg.append(pg)
            self.buf_lb.append(expert)
            new_samples += 1

            # Trim buffer if too large
            if len(self.buf_lb) > self.max_buf:
                self.buf_sc = self.buf_sc[-self.max_buf:]
                self.buf_sg = self.buf_sg[-self.max_buf:]
                self.buf_pc = self.buf_pc[-self.max_buf:]
                self.buf_pg = self.buf_pg[-self.max_buf:]
                self.buf_lb = self.buf_lb[-self.max_buf:]

            # Execute: beta=1 → expert, beta=0 → controller
            with torch.no_grad():
                action_probs = self.controller(sc, sg, pc, pg)
                predicted    = action_probs.argmax(dim=-1).item()
            execute = expert if np.random.random() < beta else predicted
            x, y, d = self.manual_step(x, y, d, execute)

            if (x, y) == (x_goal, y_goal):
                success = True
                break

        self.episode += 1
        return success, new_samples

    # ── Train on replay buffer ─────────────────────────────────────────

    def train_on_buffer(self, n_steps=32, batch_size=256):
        if len(self.buf_lb) < batch_size:
            return 0.0

        self.controller.train()

        # Stack buffer into tensors once per training call
        sc_t = torch.cat(self.buf_sc, dim=0)
        sg_t = torch.cat(self.buf_sg, dim=0)
        pc_t = torch.cat(self.buf_pc, dim=0)
        pg_t = torch.cat(self.buf_pg, dim=0)
        lb_t = torch.tensor(self.buf_lb, dtype=torch.long).to(self.device)
        N    = len(lb_t)

        total_loss = 0.0
        for _ in range(n_steps):
            idxs   = np.random.randint(0, N, size=batch_size)
            logits = self.controller(sc_t[idxs], sg_t[idxs], pc_t[idxs], pg_t[idxs])
            loss   = F.cross_entropy(logits, lb_t[idxs])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_steps

    # ── Evaluate controller on fresh episodes ──────────────────────────

    def evaluate(self, n_episodes=100, min_dist=1, max_dist=14):
        self.controller.eval()
        successes = 0
        for _ in range(n_episodes):
            x_start, y_start, d_start = self.sample_position()
            while True:
                x_goal, y_goal, d_goal = self.sample_position()
                if min_dist <= self.manhattan((x_start,y_start),(x_goal,y_goal)) <= max_dist:
                    break

            self.set_agent_state(x_goal, y_goal, d_goal)
            sg = self.build_state(self.encode_obs(self.env.render()), d_goal)
            pg = self.build_pos(x_goal, y_goal)

            x, y, d = x_start, y_start, d_start
            for _ in range(max_dist * 4 + 10):
                self.set_agent_state(x, y, d)
                sc = self.build_state(self.encode_obs(self.env.render()), d)
                pc = self.build_pos(x, y)
                with torch.no_grad():
                    action = self.controller(sc, sg, pc, pg).argmax(dim=-1).item()
                x, y, d = self.manual_step(x, y, d, action)
                if (x, y) == (x_goal, y_goal):
                    successes += 1
                    break
        return successes / n_episodes

    # ── Main training loop ─────────────────────────────────────────────

    def train(self, n_episodes=3000, collect_every=1, train_every=10,
              n_train_steps=32, batch_size=256, log_every=100, save_every=500):

        print(f"\nDAgger + Replay Buffer Training")
        print(f"Episodes:     {n_episodes}")
        print(f"Train every:  {train_every} episodes, {n_train_steps} steps/update\n")

        losses, successes = [], []

        # Warmup: fill buffer before training starts
        print("Warmup: collecting 500 episodes before training...")
        for ep in range(500):
            min_dist, max_dist = self.get_goal_dist_range(ep, n_episodes)
            self.collect_episode(min_dist=min_dist, max_dist=max_dist, beta=1.0)
        print(f"Buffer size after warmup: {len(self.buf_lb)}\n")

        for ep in range(n_episodes):
            min_dist, max_dist = self.get_goal_dist_range(ep, n_episodes)
            beta = max(0.0, 1.0 - ep / (n_episodes * 0.8))

            success, _ = self.collect_episode(
                min_dist=min_dist, max_dist=max_dist, beta=beta
            )
            successes.append(float(success))

            # Train on buffer every train_every episodes
            if (ep + 1) % train_every == 0:
                loss = self.train_on_buffer(n_steps=n_train_steps, batch_size=batch_size)
                losses.append(loss)

            if (ep + 1) % log_every == 0:
                avg_success = np.mean(successes[-log_every:])
                avg_loss    = np.mean(losses[-10:]) if losses else 0.0
                buf_size    = len(self.buf_lb)
                # Quick eval on 50 fresh episodes
                eval_sr = self.evaluate(n_episodes=50, min_dist=1, max_dist=14)
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"collect_success: {avg_success*100:.1f}%  "
                    f"loss: {avg_loss:.4f}  "
                    f"eval_sr: {eval_sr*100:.1f}%  "
                    f"buf: {buf_size}  "
                    f"dist: {min_dist}-{max_dist}  "
                    f"beta: {beta:.2f}"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nDAgger training complete.")
        self.save_checkpoint("dagger_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_dagger_{tag}.pt"
        torch.save({
            "controller": self.controller.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "episode":    self.episode,
        }, path)
        print(f"  Saved: {path}")
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.controller.load_state_dict(ckpt["controller"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode = ckpt["episode"]
        print(f"Resumed from {path} (episode {self.episode})")


if __name__ == "__main__":
    trainer = DAggerTrainer(
        jepa_checkpoint = "checkpoints/jepa_phase1_final.pt",
        env_id          = "MiniGrid-Empty-16x16-v0",
        latent_dim      = 256,
        lr              = 1e-3,
    )
    trainer.load_checkpoint("checkpoints/controller_dagger_dagger_final.pt")
    trainer.train(
        n_episodes    = 2000,
        train_every   = 10,
        n_train_steps = 32,
        batch_size    = 256,
        log_every     = 100,
        save_every    = 500,
    )