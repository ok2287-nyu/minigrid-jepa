import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
import minigrid
from pathlib import Path
import os

from encoder import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class WorldModelPlanner:
    """
    Uses the trained world model to navigate to a goal.
    No additional training — pure imagination-based planning.

    At each step:
    1. Encode current observation → z_current
    2. Encode goal observation   → z_goal
    3. Imagine all possible action sequences of length H
    4. Pick the sequence whose final z is closest to z_goal
    5. Execute the first action
    6. Repeat
    """
    def __init__(self, checkpoint_path, latent_dim=256, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load trained encoder and predictor
        self.encoder   = Encoder(latent_dim).to(self.device)
        self.predictor = Predictor(latent_dim, n_actions=3).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["online_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

        self.encoder.eval()
        self.predictor.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

        print(f"World model loaded from {checkpoint_path}")

    @torch.no_grad()
    def encode_obs(self, frame):
        """Convert raw frame to latent vector."""
        from PIL import Image
        img  = Image.fromarray(frame).resize((64, 64))
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = arr.transpose(2, 0, 1)                     # (3, 64, 64)
        t    = torch.tensor(arr).unsqueeze(0).to(self.device)  # (1, 3, 64, 64)
        return self.encoder(t)                             # (1, 256)

    @torch.no_grad()
    def plan(self, z_current, z_goal, horizon=5, n_actions=3):
        """
        Random shooting planner.

        Samples many random action sequences, imagines each one
        in latent space, returns the first action of the sequence
        that ends closest to the goal.

        horizon:   how many steps ahead to imagine
        n_samples: how many random sequences to try
        """
        n_samples = 200

        # Sample random action sequences: (n_samples, horizon)
        action_seqs = torch.randint(
            0, n_actions, (n_samples, horizon)
        ).to(self.device)

        # Expand z_current to match n_samples: (n_samples, 256)
        z = z_current.expand(n_samples, -1)

        # Roll out each sequence in latent space
        for h in range(horizon):
            actions = action_seqs[:, h]          # (n_samples,)
            z = self.predictor(z, actions)        # (n_samples, 256)
            z = F.normalize(z, dim=-1)

        # Measure distance from final imagined z to goal z
        z_goal_expanded = z_goal.expand(n_samples, -1)
        z_goal_norm     = F.normalize(z_goal_expanded, dim=-1)

        distances = F.mse_loss(z, z_goal_norm, reduction="none").mean(dim=1)

        # Pick best sequence
        best_idx    = distances.argmin()
        best_action = action_seqs[best_idx, 0].item()

        return best_action, distances.min().item()


def run_visual_demo(checkpoint_path, n_episodes=3, max_steps=150):
    planner = WorldModelPlanner(checkpoint_path)
    env     = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")

    Path("notebooks").mkdir(exist_ok=True)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        step   = 0
        frames = []
        positions = []

        # ── Collect reference frames by random exploration ────────────
        # We need a goal frame from THIS environment instance
        # so the latent space is consistent
        print(f"\nEpisode {episode+1}: collecting reference frames...")

        ref_frames    = []
        ref_positions = []
        for _ in range(100):
            a = np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.50])
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()
            ref_frames.append(env.render())
            ref_positions.append(tuple(env.unwrapped.agent_pos))

        # Pick a goal that is far from current position
        obs, _ = env.reset()
        start_pos = tuple(env.unwrapped.agent_pos)

        # Find the frame whose position is furthest from start
        distances_from_start = [
            np.sqrt((p[0]-start_pos[0])**2 + (p[1]-start_pos[1])**2)
            for p in ref_positions
        ]
        goal_idx      = np.argmax(distances_from_start)
        goal_frame    = ref_frames[goal_idx]
        goal_position = ref_positions[goal_idx]
        z_goal        = planner.encode_obs(goal_frame)

        print(f"  Start: {start_pos}")
        print(f"  Goal:  {goal_position}")
        print(f"  Straight-line distance: "
              f"{distances_from_start[goal_idx]:.1f} cells")

        # ── Navigate using world model ────────────────────────────────
        frames.append(env.render())
        positions.append(start_pos)

        latent_distances = []

        while not done and step < max_steps:
            frame     = env.render()
            z_current = planner.encode_obs(frame)
            action, ldist = planner.plan(z_current, z_goal, horizon=5)
            latent_distances.append(ldist)

            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            frames.append(env.render())
            positions.append(tuple(env.unwrapped.agent_pos))
            step += 1

        final_pos  = positions[-1]
        final_dist = np.sqrt(
            (final_pos[0] - goal_position[0])**2 +
            (final_pos[1] - goal_position[1])**2
        )
        print(f"  Final position: {final_pos}")
        print(f"  Distance to goal: {final_dist:.1f} cells")
        print(f"  Steps taken: {step}")

        # ── Plot ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(
            f"Episode {episode+1}: World Model Navigation\n"
            f"Start: {start_pos}  |  "
            f"Goal: {goal_position}  |  "
            f"Final: {final_pos}  |  "
            f"Distance: {final_dist:.1f} cells",
            fontsize=11
        )

        # Key frames
        key_indices = [0, len(frames)//3, 2*len(frames)//3, -1]
        key_labels  = ["Start", "Early", "Late", "End"]
        for i, (fidx, label) in enumerate(zip(key_indices, key_labels)):
            ax = fig.add_subplot(2, 5, i+1)
            ax.imshow(frames[fidx])
            ax.set_title(f"{label}\n{positions[fidx]}")
            ax.axis("off")

        # Goal frame
        ax_goal = fig.add_subplot(2, 5, 5)
        ax_goal.imshow(goal_frame)
        ax_goal.set_title(f"Goal\n{goal_position}")
        ax_goal.axis("off")

        # Trajectory
        ax_traj = fig.add_subplot(2, 1, 2)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
        for i in range(len(xs)-1):
            ax_traj.plot(
                [xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=colors[i], linewidth=2, alpha=0.8
            )

        ax_traj.scatter(xs[0],  ys[0],
                        color="green", s=200, zorder=5,
                        label=f"Start {start_pos}")
        ax_traj.scatter(xs[-1], ys[-1],
                        color="blue",  s=200, zorder=5,
                        label=f"End {final_pos}")
        ax_traj.scatter(goal_position[0], goal_position[1],
                        color="red", marker="*", s=400, zorder=5,
                        label=f"Goal {goal_position}")

        ax_traj.set_xlim(0, 19)
        ax_traj.set_ylim(0, 19)
        ax_traj.set_xlabel("X position")
        ax_traj.set_ylabel("Y position")
        ax_traj.set_title(
            "Agent Trajectory  (purple=start → yellow=end)"
        )
        ax_traj.legend(loc="upper right")
        ax_traj.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"notebooks/episode_{episode+1}_navigation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.show()

    env.close()
    print("\nDemo complete.")

if __name__ == "__main__":
    run_visual_demo(
        checkpoint_path = "checkpoints/jepa_final.pt",
        n_episodes      = 3,
        max_steps       = 200,
    )