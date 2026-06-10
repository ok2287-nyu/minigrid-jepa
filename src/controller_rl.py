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


class RLTrainer:
    """
    Stage 4: REINFORCE policy gradient training.

    Controller initialized from BC weights (already knows basic controls).
    Fine-tuned through real environment interaction.

    At each step:
        - Controller takes action in real environment
        - Gets reward based on progress toward goal
        - REINFORCE updates controller weights

    Reward:
        +10.0  reached goal
        +0.1   got closer (manhattan distance decreased)
        -0.1   got farther
        -0.5   didn't move (hit wall)
        -0.01  small step penalty (encourages efficiency)
    """
    def __init__(
        self,
        jepa_checkpoint,
        bc_checkpoint,
        env_id      = "MiniGrid-Empty-16x16-v0",
        latent_dim  = 256,
        n_dirs      = 4,
        n_actions   = 3,
        lr          = 1e-4,
        gamma       = 0.99,
        device      = None,
    ):
        self.device    = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma     = gamma
        self.state_dim = latent_dim + n_dirs   # 260

        print(f"Training on:  {self.device}")

        # ── Environment ───────────────────────────────────────────────
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width  = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.x_range = list(range(1, self.width  - 1))
        self.y_range = list(range(1, self.height - 1))

        # ── Frozen encoder ────────────────────────────────────────────
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
        print(f"Encoder loaded and frozen.")

        # ── Controller — initialized from BC ──────────────────────────
        self.controller = Controller(
            state_dim  = self.state_dim,
            hidden_dim = 512,
            n_actions  = n_actions,
            pos_dim    = 2,
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

        self.episode = 0

    # ── Utilities ─────────────────────────────────────────────────────

    def preprocess(self, frame):
        """Raw frame → (1, 3, 64, 64) normalized tensor."""
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr).unsqueeze(0).to(self.device)

    def encode_obs(self, frame):
        """Frame → z (1, 256)."""
        with torch.no_grad():
            return self.encoder(self.preprocess(frame))

    def build_state(self, z, direction):
        """[z (256) | dir_onehot (4)] = (1, 260)."""
        dir_t  = torch.tensor([direction], dtype=torch.long).to(self.device)
        dir_oh = F.one_hot(dir_t, num_classes=4).float()
        return torch.cat([z, dir_oh], dim=-1)

    def build_pos(self, x, y):
        """Normalized position tensor (1, 2)."""
        return torch.tensor(
            [[x / 14.0, y / 14.0]], dtype=torch.float32
        ).to(self.device)

    def set_agent_state(self, x, y, direction):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def sample_position(self):
        x = np.random.choice(self.x_range)
        y = np.random.choice(self.y_range)
        d = np.random.randint(0, 4)
        return x, y, d

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def compute_reward(self, agent_pos, prev_pos, goal_pos):
        """
        Reward based on progress toward goal.
        """
        if tuple(agent_pos) == tuple(goal_pos):
            return 10.0   # reached goal

        curr_dist = self.manhattan(agent_pos, goal_pos)
        prev_dist = self.manhattan(prev_pos,  goal_pos)

        if curr_dist < prev_dist:
            return 0.1    # got closer
        elif curr_dist > prev_dist:
            return -0.1   # got farther
        else:
            return -0.5   # didn't move (wall)

    def compute_returns(self, rewards):
        """
        Discounted cumulative returns for each timestep.
        G_t = r_t + γ*r_t+1 + γ²*r_t+2 + ...

        Normalized to reduce variance.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns — reduces variance, stabilizes training
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    # ── Episode ───────────────────────────────────────────────────────

    def run_episode(self, max_steps=100):
        """
        Run one full episode.

        Returns:
            log_probs: list of log probabilities of actions taken
            rewards:   list of rewards received
            success:   whether goal was reached
            steps:     number of steps taken
        """
        self.controller.train()
        self.env.reset()

        # Sample start and goal positions
        x_start, y_start, d_start = self.sample_position()
        while True:
            x_goal, y_goal, d_goal = self.sample_position()
            if self.manhattan(
                (x_start, y_start), (x_goal, y_goal)
            ) >= 1:   # ensure goal is not trivially close
                break

        # Encode goal state (frozen — no gradient)
        self.set_agent_state(x_goal, y_goal, d_goal)
        goal_frame  = self.env.render()
        z_goal      = self.encode_obs(goal_frame)
        state_goal  = self.build_state(z_goal, d_goal)
        pos_goal    = self.build_pos(x_goal, y_goal)

        # Set agent to start
        self.set_agent_state(x_start, y_start, d_start)
        current_dir = d_start

        log_probs = []
        rewards   = []
        entropies = []
        success   = False

        for step in range(max_steps):
            # Encode current state
            frame      = self.env.render()
            z_current  = self.encode_obs(frame)
            state_curr = self.build_state(z_current, current_dir)
            agent_xy   = tuple(self.env.unwrapped.agent_pos)
            pos_curr   = self.build_pos(agent_xy[0], agent_xy[1])

            # Controller selects action
            action_probs = self.controller(
                state_curr, state_goal,
                pos_curr,   pos_goal
            )                                                        # (1, 3)
            dist         = torch.distributions.Categorical(action_probs)
            action       = dist.sample()                             # scalar
            log_prob     = dist.log_prob(action)                    # scalar
            entropy  = dist.entropy()   # ADD THIS

            # Execute in real environment
            prev_pos  = tuple(self.env.unwrapped.agent_pos)
            self.env.step(action.item())
            agent_pos   = tuple(self.env.unwrapped.agent_pos)
            current_dir = self.env.unwrapped.agent_dir
            if self.episode == 0 and step < 10:
                print(f"  step {step}: pos={agent_pos} dir={current_dir} "
                    f"action={action.item()} probs={action_probs.detach().cpu().numpy().round(2)}")
                        # Compute reward
            reward = self.compute_reward(agent_pos, prev_pos, (x_goal, y_goal))
            reward -= 0.01   # small step penalty

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)   # ADD THIS

            if tuple(agent_pos) == (x_goal, y_goal):
                success = True
                break

        return log_probs, rewards,entropies, success, step + 1

    # ── Training ──────────────────────────────────────────────────────

    def update(self, log_probs, rewards, entropies):
        """
        REINFORCE update.

        loss = -sum(G_t * log_prob(action_t))

        Positive return → increase probability of action taken
        Negative return → decrease probability of action taken
        """
        returns = self.compute_returns(rewards)

        loss = 0.0
        for log_prob, G, entropy in zip(log_probs, returns, entropies):
            loss += -G * log_prob - 0.1 * entropy  # entropy bonus

        self.optimizer.zero_grad()
        loss.backward()

        if self.episode % 100 == 0:
            total_grad = sum(
                p.grad.abs().mean().item()
                for p in self.controller.parameters()
                if p.grad is not None
            )
            print(f"  grad: {total_grad:.6f}")
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes=5000, log_every=100, save_every=500):
        print(f"\nStage 4: REINFORCE Training")
        print(f"Episodes:   {n_episodes}")
        print(f"Gamma:      {self.gamma}\n")

        successes    = []
        total_rewards = []
        steps_list   = []

        for ep in range(n_episodes):
            log_probs, rewards,entropies, success, steps = self.run_episode()
            loss = self.update(log_probs, rewards, entropies)

            successes.append(float(success))
            total_rewards.append(sum(rewards))
            steps_list.append(steps)
            self.episode += 1

            if (ep + 1) % log_every == 0:
                avg_success = np.mean(successes[-log_every:])
                avg_reward  = np.mean(total_rewards[-log_every:])
                avg_steps   = np.mean(steps_list[-log_every:])
                print(
                    f"Episode [{ep+1:>5}/{n_episodes}]  "
                    f"success: {avg_success*100:.1f}%  "
                    f"reward: {avg_reward:.2f}  "
                    f"steps: {avg_steps:.1f}  "
                    f"loss: {loss:.4f}"
                )

            if (ep + 1) % save_every == 0:
                self.save_checkpoint(ep + 1)

        print("\nRL training complete.")
        self.save_checkpoint("rl_final")

    def save_checkpoint(self, tag):
        Path("checkpoints").mkdir(exist_ok=True)
        path = f"checkpoints/controller_rl_{tag}.pt"
        torch.save({
            "controller" : self.controller.state_dict(),
            "optimizer"  : self.optimizer.state_dict(),
            "episode"    : self.episode,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    trainer = RLTrainer(
        jepa_checkpoint = "checkpoints/jepa_phase1_final.pt",
        bc_checkpoint   = "checkpoints/controller_bc_final.pt",
        env_id          = "MiniGrid-Empty-16x16-v0",
        latent_dim      = 256,
        lr              = 1e-4,
        gamma           = 0.99,
    )

    trainer.train(
        n_episodes  = 5000,
        log_every   = 100,
        save_every  = 500,
    )