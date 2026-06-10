"""
High-Level Controller for FourRooms Hierarchical Navigation

Architecture:
    High-level: sees (current_state, final_goal_state, current_pos, final_goal_pos)
                outputs next SUBGOAL position (normalized x,y)
                trained with MSE against BFS-extracted doorway positions

    Low-level:  existing Empty DAgger controller (FROZEN, 91% on short paths)
                sees (current_state, subgoal_state, current_pos, subgoal_pos)
                outputs action (left/right/forward)

Together:
    High-level picks next doorway → low-level navigates there → repeat until goal
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

from encoder_v2 import Encoder
from controller_bc import Controller

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


# ── BFS utilities ─────────────────────────────────────────────────────

def bfs_path(start, goal, walkable_set):
    if start == goal: return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (x,y), path = queue.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if (nx,ny) in walkable_set and (nx,ny) not in visited:
                new_path = path + [(nx,ny)]
                if (nx,ny) == goal: return new_path
                visited.add((nx,ny))
                queue.append(((nx,ny), new_path))
    return None


def find_doorways(grid, width, height):
    """
    Find all doorway cells in current grid layout.
    Doorway = floor cell with walls on exactly two opposite sides.
    These change position every reset.
    """
    doorways = []
    for x in range(1, width-1):
        for y in range(1, height-1):
            cell = grid.get(x, y)
            if cell is not None:
                continue
            n = grid.get(x, y-1)
            s = grid.get(x, y+1)
            e = grid.get(x+1, y)
            w = grid.get(x-1, y)
            def is_wall(c): return c is not None and c.type == 'wall'
            if (is_wall(n) and is_wall(s)) or (is_wall(e) and is_wall(w)):
                doorways.append((x, y))
    return doorways


def extract_subgoals(path, doorway_set, final_goal):
    """
    Extract subgoal sequence from BFS path.
    Subgoals are doorway cells the path passes through,
    plus the final goal.

    path:        list of (x,y) from start to goal
    doorway_set: set of (x,y) that are doorways in current layout
    final_goal:  (x,y) of the final destination

    Returns list of (x,y) subgoals in order.
    """
    subgoals = []
    for pos in path[1:]:   # skip start position
        if pos in doorway_set:
            subgoals.append(pos)
    subgoals.append(final_goal)
    return subgoals


# ── High-Level Controller ─────────────────────────────────────────────

class HighLevelController(nn.Module):
    """
    Predicts the next subgoal position given current state and final goal.

    Input:  state_curr(260) + state_goal(260) + pos_curr(2) + pos_goal(2) = 524
    Output: subgoal_pos(2) — normalized (x,y) of next doorway or final goal

    Trained with MSE against BFS-extracted subgoal positions.
    """
    def __init__(self, state_dim=260, hidden_dim=256, pos_dim=2):
        super().__init__()
        input_dim = state_dim * 2 + pos_dim * 2   # 524

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pos_dim),   # output (x,y)
            nn.Sigmoid(),                      # normalized 0-1
        )

    def forward(self, state_curr, state_goal, pos_curr, pos_goal):
        x = torch.cat([state_curr, state_goal, pos_curr, pos_goal], dim=-1)
        return self.net(x)   # (B, 2)


# ── Trainer ───────────────────────────────────────────────────────────

class HierarchicalTrainer:
    def __init__(
        self,
        jepa_checkpoint,
        lowlevel_checkpoint,
        env_id     = "MiniGrid-FourRooms-v0",
        latent_dim = 256,
        n_dirs     = 4,
        lr         = 3e-3,
        device     = None,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = latent_dim + n_dirs   # 260
        print(f"Training on: {self.device}")

        # ── Environment ───────────────────────────────────────────────
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width  = self.env.unwrapped.width    # 19
        self.height = self.env.unwrapped.height   # 19
        self.max_x  = self.width  - 2             # 17
        self.max_y  = self.height - 2             # 17

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
        print("Encoder loaded and frozen.")

        # ── Frozen low-level controller (Empty DAgger) ────────────────
        self.lowlevel = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = 3,
            pos_dim    = 2,
        ).to(self.device)
        ckpt2 = torch.load(lowlevel_checkpoint, map_location=self.device, weights_only=False)
        self.lowlevel.load_state_dict(ckpt2["controller"])
        for p in self.lowlevel.parameters():
            p.requires_grad = False
        self.lowlevel.eval()
        print(f"Low-level controller loaded and frozen: {lowlevel_checkpoint}")

        # ── High-level controller (trainable) ─────────────────────────
        self.highlevel = HighLevelController(
            state_dim  = self.state_dim,
            hidden_dim = 256,
            pos_dim    = 2,
        ).to(self.device)
        print("High-level controller randomly initialized.")

        self.optimizer = torch.optim.Adam(
            self.highlevel.parameters(), lr=lr
        )

        # ── Replay buffer for high-level ──────────────────────────────
        # Stores (sc, sg_final, pc, pg_final, subgoal_pos_target)
        self.buf_sc  = []   # current state
        self.buf_sg  = []   # final goal state
        self.buf_pc  = []   # current pos
        self.buf_pg  = []   # final goal pos
        self.buf_tgt = []   # target subgoal position (x,y) normalized
        self.max_buf = 50_000

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

    def add_to_buffer(self, sc, sg, pc, pg, tgt):
        self.buf_sc.append(sc)
        self.buf_sg.append(sg)
        self.buf_pc.append(pc)
        self.buf_pg.append(pg)
        self.buf_tgt.append(tgt)
        if len(self.buf_tgt) > self.max_buf:
            self.buf_sc  = self.buf_sc[-self.max_buf:]
            self.buf_sg  = self.buf_sg[-self.max_buf:]
            self.buf_pc  = self.buf_pc[-self.max_buf:]
            self.buf_pg  = self.buf_pg[-self.max_buf:]
            self.buf_tgt = self.buf_tgt[-self.max_buf:]

    # ── Collect one episode ───────────────────────────────────────────

    def collect_episode(self):
        """
        Collect training data for the high-level controller.

        For each step along the BFS path:
            - current state is the input
            - next subgoal (doorway or final goal) is the target

        The high-level controller should learn to predict
        which subgoal to navigate toward from any position.
        """
        self.highlevel.eval()
        self.env.reset()

        # Rediscover walkable cells and doorways
        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        doorways          = find_doorways(
            self.env.unwrapped.grid, self.width, self.height
        )
        doorway_set = set(doorways)

        # Sample start and goal — prefer long paths that cross rooms
        for _ in range(200):
            start = self.walkable[np.random.randint(len(self.walkable))]
            goal  = self.walkable[np.random.randint(len(self.walkable))]
            if start == goal: continue
            path = bfs_path(start, goal, self.walkable_set)
            if path and len(path)-1 >= 5:   # at least 5 steps
                break
        else:
            return False

        x_start, y_start = start
        x_goal,  y_goal  = goal
        d_start = np.random.randint(0, 4)
        d_goal  = np.random.randint(0, 4)

        # Extract subgoal sequence from BFS path
        subgoals = extract_subgoals(path, doorway_set, (x_goal, y_goal))

        # Encode final goal — fixed throughout episode
        self.set_agent_state(x_goal, y_goal, d_goal)
        z_goal     = self.encode_obs(self.env.render())
        sg_final   = self.build_state(z_goal, d_goal).detach()
        pg_final   = self.build_pos(x_goal, y_goal).detach()

        # Walk through BFS path, storing training pairs
        # At each position, the target is the NEXT subgoal
        subgoal_idx = 0

        for pos in path[:-1]:   # all positions except final goal
            px, py = pos

            # Advance subgoal index if we've passed current subgoal
            while (subgoal_idx < len(subgoals) - 1 and
                   subgoals[subgoal_idx] == (px, py)):
                subgoal_idx += 1

            # Current subgoal target
            sg_x, sg_y = subgoals[subgoal_idx]
            tgt = torch.tensor(
                [[sg_x / self.max_x, sg_y / self.max_y]],
                dtype=torch.float32
            ).to(self.device).detach()

            # Encode current position
            self.set_agent_state(px, py, d_start)
            z_curr = self.encode_obs(self.env.render())
            sc     = self.build_state(z_curr, d_start).detach()
            pc     = self.build_pos(px, py).detach()

            # Store (current_state, final_goal, current_pos, final_goal_pos, subgoal_target)
            self.add_to_buffer(sc, sg_final, pc, pg_final, tgt)

        return True

    # ── Train high-level on buffer ────────────────────────────────────

    def train_on_buffer(self, n_steps=32, batch_size=256):
        if len(self.buf_tgt) < batch_size:
            return 0.0

        self.highlevel.train()

        sc_t  = torch.cat(self.buf_sc,  dim=0)   # (N, 260)
        sg_t  = torch.cat(self.buf_sg,  dim=0)   # (N, 260)
        pc_t  = torch.cat(self.buf_pc,  dim=0)   # (N, 2)
        pg_t  = torch.cat(self.buf_pg,  dim=0)   # (N, 2)
        tgt_t = torch.cat(self.buf_tgt, dim=0)   # (N, 2)
        N     = len(tgt_t)

        total_loss = 0.0
        for _ in range(n_steps):
            idxs      = np.random.randint(0, N, size=batch_size)
            pred_pos  = self.highlevel(
                sc_t[idxs], sg_t[idxs], pc_t[idxs], pg_t[idxs]
            )   # (B, 2)
            diff = (pred_pos - tgt_t[idxs]).abs()
            weight = torch.where(diff > (2/17), torch.ones_like(diff) * 3.0, torch.ones_like(diff))
            loss = (weight * diff ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.highlevel.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_steps

    # ── Evaluate full hierarchical system ─────────────────────────────

    def evaluate(self, n_episodes=50):
        """
        Evaluate the full hierarchical system:
        High-level predicts subgoal → low-level navigates to it → repeat
        """
        self.highlevel.eval()
        self.lowlevel.eval()
        self.env.reset()
        self.walkable     = self._find_walkable()
        self.walkable_set = set(self.walkable)
        successes = 0

        for _ in range(n_episodes):
            # Sample start and goal
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

            # Encode final goal
            self.set_agent_state(x_goal, y_goal, d_goal)
            z_goal     = self.encode_obs(self.env.render()).detach()
            sg_final   = self.build_state(z_goal, d_goal)
            pg_final   = self.build_pos(x_goal, y_goal)

            x, y, d    = x_start, y_start, d_start
            max_steps  = (len(path)-1) * 6 + 20
            success    = False

            # Subgoal tracking
            current_subgoal_pos = None
            current_subgoal_state = None
            subgoal_steps = 0
            max_subgoal_steps = 20  # max steps to reach each subgoal

            for step in range(max_steps):
                # Encode current state
                self.set_agent_state(x, y, d)
                z_curr     = self.encode_obs(self.env.render()).detach()
                state_curr = self.build_state(z_curr, d)
                pos_curr   = self.build_pos(x, y)

                # High-level: get next subgoal if needed
                if (current_subgoal_pos is None or
                        subgoal_steps >= max_subgoal_steps):
                    with torch.no_grad():
                        pred = self.highlevel(
                            state_curr, sg_final, pos_curr, pg_final
                        )   # (1, 2) normalized position

                    # Convert predicted normalized position to grid coords
                    sg_x = int(round(pred[0, 0].item() * self.max_x))
                    sg_y = int(round(pred[0, 1].item() * self.max_y))
                    sg_x = max(1, min(self.max_x, sg_x))
                    sg_y = max(1, min(self.max_y, sg_y))

                    # Snap to nearest walkable cell
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

                # Low-level: navigate toward current subgoal
                pos_subgoal = self.build_pos(*current_subgoal_pos)
                with torch.no_grad():
                    action = self.lowlevel(
                        state_curr,
                        current_subgoal_state,
                        pos_curr,
                        pos_subgoal
                    ).argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, action)
                subgoal_steps += 1

                # Check final goal
                if (x, y) == (x_goal, y_goal):
                    success = True
                    break

                # If subgoal reached, request new one next step
                if (x, y) == current_subgoal_pos:
                    current_subgoal_pos = None
                    subgoal_steps = 0

            if success:
                successes += 1

        return successes / n_episodes

    # ── Training loop ─────────────────────────────────────────────────

    def train(self, n_episodes=3000, train_every=10,
              n_train_steps=32, batch_size=256,
              log_every=100, save_every=500):

        print(f"\nHierarchical Navigation Training")
        print(f"High-level: learns subgoal prediction")
        print(f"Low-level:  frozen Empty DAgger controller\n")

        losses = []

        # Warmup
        print("Warmup: collecting 500 episodes...")
        for _ in range(500):
            self.collect_episode()
        print(f"Buffer size after warmup: {len(self.buf_tgt)}\n")

        for ep in range(n_episodes):
            self.collect_episode()

            if (ep + 1) % train_every == 0:
                loss = self.train_on_buffer(
                    n_steps=n_train_steps, batch_size=batch_size
                )
                losses.append(loss)

            if (ep + 1) % log_every == 0:
                avg_loss = np.mean(losses[-10:]) if losses else 0.0
                eval_sr  = self.evaluate(n_episodes=50)
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"loss: {avg_loss:.6f}  "
                    f"eval_sr: {eval_sr*100:.1f}%  "
                    f"buf: {len(self.buf_tgt)}"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nHierarchical training complete.")
        self.save_checkpoint("hierarchical_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/highlevel_{tag}.pt"
        torch.save({
            "highlevel" : self.highlevel.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    trainer = HierarchicalTrainer(
        jepa_checkpoint     = "checkpoints/jepa_fourrooms_final.pt",
        lowlevel_checkpoint = "checkpoints/controller_fourrooms_fourrooms_final.pt",
        env_id              = "MiniGrid-FourRooms-v0",
        latent_dim          = 256,
        lr                  = 1e-4,
    )

    # Resume from existing checkpoint
    ckpt = torch.load(
        "checkpoints/highlevel_hierarchical_final.pt",
        map_location = trainer.device,
        weights_only = False
    )
    trainer.highlevel.load_state_dict(ckpt["highlevel"])
    trainer.optimizer = torch.optim.Adam(
        trainer.highlevel.parameters(), lr=1e-4
    )
    print("High-level resumed from checkpoint.")

    trainer.train(
        n_episodes    = 5000,
        train_every   = 10,
        n_train_steps = 32,
        batch_size    = 256,
        log_every     = 100,
        save_every    = 500,
    )