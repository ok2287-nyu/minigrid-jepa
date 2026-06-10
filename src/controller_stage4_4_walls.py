import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import os
import gymnasium as gym
import minigrid
from collections import deque

from encoder_v2 import Encoder
from controller_bc import Controller

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


def bfs_path(start, goal, walkable_set):
    """BFS shortest path from start to goal through walkable cells."""
    if start == goal:
        return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if (nx,ny) in walkable_set and (nx,ny) not in visited:
                new_path = path + [(nx,ny)]
                if (nx,ny) == goal:
                    return new_path
                visited.add((nx,ny))
                queue.append(((nx,ny), new_path))
    return None


def bfs_action(x, y, d, x_goal, y_goal, walkable_set):
    path = bfs_path((x,y), (x_goal,y_goal), walkable_set)
    if path is None or len(path) < 2:
        return 2

    nx, ny = path[1]
    dx, dy = nx - x, ny - y

    if   dx ==  1: desired = 0
    elif dx == -1: desired = 2
    elif dy ==  1: desired = 1
    elif dy == -1: desired = 3
    else:          return 2

    if d == desired:
        return 2

    diff = (desired - d) % 4
    if diff == 1:   return 1
    elif diff == 3: return 0
    else:           return 1


class FourRoomsDAgger:
    def __init__(
        self,
        jepa_checkpoint,
        controller_checkpoint = None,
        env_id     = "MiniGrid-FourRooms-v0",
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
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.max_x  = self.width  - 2
        self.max_y  = self.height - 2

        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        print(f"Walkable cells: {len(self.walkable)}")

        self.encoder = Encoder(latent_dim).to(self.device)
        ckpt = torch.load(jepa_checkpoint, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["online_encoder"])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        print("Encoder loaded and frozen.")

        self.controller = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = n_actions,
            pos_dim    = 2,
        ).to(self.device)

        if controller_checkpoint is not None:
            ckpt2 = torch.load(
                controller_checkpoint,
                map_location = self.device,
                weights_only = False
            )
            self.controller.load_state_dict(ckpt2["controller"])
            print(f"Controller loaded from: {controller_checkpoint}")
        else:
            print("Controller randomly initialized.")

        self.optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=lr
        )
        self.episode = 0

        self.buf_sc = []
        self.buf_sg = []
        self.buf_pc = []
        self.buf_pg = []
        self.buf_lb = []
        self.max_buf = 100_000

    # ── Utilities ──────────────────────────────────────────────────────

    def _find_walkable(self):
        grid = self.env.unwrapped.grid
        cells = []
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
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

    def build_state(self, z, direction):
        dir_oh = F.one_hot(
            torch.tensor([direction], dtype=torch.long).to(self.device),
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

    def get_room(self, x, y):
        center_x = self.width  // 2
        center_y = self.height // 2
        if x < center_x and y < center_y:    return 0  # top-left
        elif x >= center_x and y < center_y: return 1  # top-right
        elif x < center_x and y >= center_y: return 2  # bottom-left
        else:                                 return 3  # bottom-right

    def get_curriculum(self, episode, n_episodes):
        progress = episode / n_episodes
        if progress < 0.30:
            return 1, 5, "same_room"
        elif progress < 0.60:
            return 6, 12, "diff_room"
        else:
            return 10, 28, "any"

    def add_to_buffer(self, sc, sg, pc, pg, label):
        self.buf_sc.append(sc)
        self.buf_sg.append(sg)
        self.buf_pc.append(pc)
        self.buf_pg.append(pg)
        self.buf_lb.append(label)
        if len(self.buf_lb) > self.max_buf:
            self.buf_sc = self.buf_sc[-self.max_buf:]
            self.buf_sg = self.buf_sg[-self.max_buf:]
            self.buf_pc = self.buf_pc[-self.max_buf:]
            self.buf_pg = self.buf_pg[-self.max_buf:]
            self.buf_lb = self.buf_lb[-self.max_buf:]

    # ── Collect one episode ───────────────────────────────────────────

    def collect_episode(self, min_dist=1, max_dist=4, beta=1.0, room_constraint="any"):
        self.controller.eval()
        self.env.reset()

        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)

        for _ in range(200):
            start = self.walkable[np.random.randint(len(self.walkable))]
            goal  = self.walkable[np.random.randint(len(self.walkable))]
            if start == goal: continue
            if room_constraint == "same_room":
                if self.get_room(*start) != self.get_room(*goal): continue
            elif room_constraint == "diff_room":
                if self.get_room(*start) == self.get_room(*goal): continue
            path = bfs_path(start, goal, self.walkable_set)
            if path is None: continue
            dist = len(path) - 1
            if min_dist <= dist <= max_dist:
                break
        else:
            return False

        x_start, y_start = start
        x_goal,  y_goal  = goal
        d_start = np.random.randint(0, 4)
        d_goal  = np.random.randint(0, 4)

        self.set_agent_state(x_goal, y_goal, d_goal)
        z_goal     = self.encode_obs(self.env.render())
        sg         = self.build_state(z_goal, d_goal).detach()
        pg         = self.build_pos(x_goal, y_goal).detach()

        x, y, d    = x_start, y_start, d_start
        success    = False
        max_steps  = max_dist * 4 + 10

        for step in range(max_steps):
            self.set_agent_state(x, y, d)
            z          = self.encode_obs(self.env.render())
            sc         = self.build_state(z, d).detach()
            pc         = self.build_pos(x, y).detach()

            expert = bfs_action(x, y, d, x_goal, y_goal, self.walkable_set)
            self.add_to_buffer(sc, sg, pc, pg, expert)

            with torch.no_grad():
                logits    = self.controller(sc, sg, pc, pg)
                predicted = logits.argmax(dim=-1).item()
            execute = expert if np.random.random() < beta else predicted
            x, y, d = self.manual_step(x, y, d, execute)

            if (x, y) == (x_goal, y_goal):
                success = True
                break

        self.episode += 1
        return success

    # ── Train on replay buffer ────────────────────────────────────────

    def train_on_buffer(self, n_steps=32, batch_size=256):
        if len(self.buf_lb) < batch_size:
            return 0.0

        self.controller.train()

        sc_t = torch.cat(self.buf_sc, dim=0)
        sg_t = torch.cat(self.buf_sg, dim=0)
        pc_t = torch.cat(self.buf_pc, dim=0)
        pg_t = torch.cat(self.buf_pg, dim=0)
        lb_t = torch.tensor(self.buf_lb, dtype=torch.long).to(self.device)
        N    = len(lb_t)

        total_loss = 0.0
        for _ in range(n_steps):
            idxs   = np.random.randint(0, N, size=batch_size)
            logits = self.controller(sc_t[idxs], sg_t[idxs],
                                     pc_t[idxs], pg_t[idxs])
            loss   = F.cross_entropy(logits, lb_t[idxs])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_steps

    # ── Evaluate ──────────────────────────────────────────────────────

    def evaluate(self, n_episodes=50):
        self.controller.eval()
        self.env.reset()
        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        successes = 0

        for _ in range(n_episodes):
            for _ in range(100):
                start = self.walkable[np.random.randint(len(self.walkable))]
                goal  = self.walkable[np.random.randint(len(self.walkable))]
                if start == goal: continue
                path = bfs_path(start, goal, self.walkable_set)
                if path and 1 <= len(path)-1 <= 28:
                    break
            else:
                continue

            x_start, y_start = start
            x_goal,  y_goal  = goal
            d_start = np.random.randint(0, 4)
            d_goal  = np.random.randint(0, 4)

            self.set_agent_state(x_goal, y_goal, d_goal)
            z_goal     = self.encode_obs(self.env.render()).detach()
            state_goal = self.build_state(z_goal, d_goal)
            pos_goal   = self.build_pos(x_goal, y_goal)

            x, y, d   = x_start, y_start, d_start
            max_steps = (len(path)-1) * 4 + 10

            for step in range(max_steps):
                self.set_agent_state(x, y, d)
                z          = self.encode_obs(self.env.render()).detach()
                state_curr = self.build_state(z, d)
                pos_curr   = self.build_pos(x, y)

                with torch.no_grad():
                    action = self.controller(
                        state_curr, state_goal, pos_curr, pos_goal
                    ).argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, action)
                if (x, y) == (x_goal, y_goal):
                    successes += 1
                    break

        return successes / n_episodes

    def evaluate_by_dist(self, n_episodes=30, min_d=1, max_d=4):
        self.controller.eval()
        self.env.reset()
        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        successes = 0
        attempts  = 0

        for _ in range(n_episodes):
            for _ in range(100):
                start = self.walkable[np.random.randint(len(self.walkable))]
                goal  = self.walkable[np.random.randint(len(self.walkable))]
                if start == goal: continue
                path = bfs_path(start, goal, self.walkable_set)
                if path and min_d <= len(path)-1 <= max_d:
                    break
            else:
                continue

            x_start, y_start = start
            x_goal,  y_goal  = goal
            d_start = np.random.randint(0, 4)
            d_goal  = np.random.randint(0, 4)

            self.set_agent_state(x_goal, y_goal, d_goal)
            z_goal     = self.encode_obs(self.env.render()).detach()
            state_goal = self.build_state(z_goal, d_goal)
            pos_goal   = self.build_pos(x_goal, y_goal)

            x, y, d   = x_start, y_start, d_start
            max_steps = max_d * 4 + 10
            attempts += 1

            for step in range(max_steps):
                self.set_agent_state(x, y, d)
                z          = self.encode_obs(self.env.render()).detach()
                state_curr = self.build_state(z, d)
                pos_curr   = self.build_pos(x, y)

                with torch.no_grad():
                    action = self.controller(
                        state_curr, state_goal, pos_curr, pos_goal
                    ).argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, action)
                if (x, y) == (x_goal, y_goal):
                    successes += 1
                    break

        return successes / max(attempts, 1)

    # ── Training loop ─────────────────────────────────────────────────

    def train(self, n_episodes=5000, train_every=10,
              n_train_steps=32, batch_size=256,
              log_every=100, save_every=500):

        print(f"\nFourRooms DAgger + Replay Buffer")
        print(f"Episodes:    {n_episodes}")
        print(f"Expert:      BFS planner\n")

        successes = []
        losses    = []

        print("Warmup: collecting 500 episodes...")
        for ep in range(500):
            min_dist, max_dist, room_constraint = self.get_curriculum(ep, n_episodes)
            self.collect_episode(
                min_dist=min_dist, max_dist=max_dist,
                beta=1.0, room_constraint=room_constraint
            )
        print(f"Buffer size after warmup: {len(self.buf_lb)}\n")

        for ep in range(n_episodes):
            min_dist, max_dist, room_constraint = self.get_curriculum(ep, n_episodes)
            beta = max(0.0, 1.0 - ep / (n_episodes * 0.8))

            success = self.collect_episode(
                min_dist=min_dist, max_dist=max_dist,
                beta=beta, room_constraint=room_constraint
            )
            successes.append(float(success))

            if (ep + 1) % train_every == 0:
                loss = self.train_on_buffer(
                    n_steps=n_train_steps, batch_size=batch_size
                )
                losses.append(loss)

            if (ep + 1) % log_every == 0:
                avg_success = np.mean(successes[-log_every:])
                avg_loss    = np.mean(losses[-10:]) if losses else 0.0
                eval_sr     = self.evaluate(n_episodes=50)
                short_sr    = self.evaluate_by_dist(n_episodes=30, min_d=1,  max_d=5)
                medium_sr   = self.evaluate_by_dist(n_episodes=30, min_d=6,  max_d=12)
                long_sr     = self.evaluate_by_dist(n_episodes=30, min_d=13, max_d=28)
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"collect_success: {avg_success*100:.1f}%  "
                    f"loss: {avg_loss:.4f}  "
                    f"eval_sr: {eval_sr*100:.1f}%  "
                    f"(S:{short_sr*100:.0f}% M:{medium_sr*100:.0f}% L:{long_sr*100:.0f}%)  "
                    f"buf: {len(self.buf_lb)}  "
                    f"dist: {min_dist}-{max_dist}  "
                    f"room: {room_constraint}  "
                    f"beta: {beta:.2f}"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nFourRooms DAgger complete.")
        self.save_checkpoint("fourrooms_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_fourrooms_{tag}.pt"
        torch.save({
            "controller" : self.controller.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "episode"    : self.episode,
        }, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.controller.load_state_dict(ckpt["controller"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode = ckpt["episode"]
        print(f"Resumed from {path} (episode {self.episode})")


if __name__ == "__main__":
    trainer = FourRoomsDAgger(
        jepa_checkpoint       = "checkpoints/jepa_fourrooms_final.pt",
        controller_checkpoint = None,
        env_id                = "MiniGrid-FourRooms-v0",
        latent_dim            = 256,
        lr                    = 3e-3,
    )
    trainer.train(
        n_episodes    = 5000,
        train_every   = 10,
        n_train_steps = 32,
        batch_size    = 256,
        log_every     = 100,
        save_every    = 500,
    )