import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import minigrid
from pathlib import Path
from PIL import Image
import os

from encoder import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class CEMPlanner:
    """
    Cross Entropy Method planner using the trained JEPA world model.

    At each real environment step:
    1. Encode current observation → z_current
    2. Run CEM search in latent space:
       a. Sample N action sequences
       b. Imagine each forward H steps using JEPA predictor
       c. Score by cumulative distance to goal
          + penalize wall hits (no state change)
          + penalize consecutive turns (spinning)
       d. Keep top K elite sequences
       e. Refit distribution from elite set
       f. Repeat for n_iterations
    3. Execute first action of best sequence
    4. Replan from new real state

    No controller network. No learning during planning.
    Pure search using world model as simulator.
    """
    def __init__(
        self,
        checkpoint_path,
        latent_dim   = 256,
        n_actions    = 3,
        horizon      = 20,
        n_samples    = 1000,
        n_elite      = 100,
        n_iterations = 7,
        device       = None,
    ):
        self.device      = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_actions   = n_actions
        self.horizon     = horizon
        self.n_samples   = n_samples
        self.n_elite     = n_elite
        self.n_iterations= n_iterations

        # Load trained world model
        self.encoder   = Encoder(latent_dim).to(self.device)
        self.predictor = Predictor(latent_dim, n_actions).to(self.device)

        ckpt = torch.load(
            checkpoint_path,
            map_location  = self.device,
            weights_only  = False
        )
        self.encoder.load_state_dict(ckpt["online_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

        self.encoder.eval()
        self.predictor.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

        print(f"CEM Planner loaded. Device: {self.device}")
        print(f"Horizon: {horizon} | Samples: {n_samples} | "
              f"Elite: {n_elite} | Iterations: {n_iterations}")

    @torch.no_grad()
    def encode_frame(self, frame):
        """Raw pixel frame → latent vector z."""
        img = Image.fromarray(frame).resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)                         # (3, 64, 64)
        t   = torch.tensor(arr).unsqueeze(0).to(self.device) # (1, 3, 64, 64)
        return self.encoder(t)                                # (1, 256)

    @torch.no_grad()
    def plan(self, z_current, z_goal):
        """
        Run full CEM search to find best action sequence.

        Scoring has three components:
        1. Cumulative distance to goal (discounted across steps)
        2. Movement penalty: penalize sequences where agent hits walls
        3. Turn penalty: penalize consecutive turns (spinning)

        Returns: best_action (int), best_score (float)
        """
        # Initialize distribution: uniform over actions
        # Shape: (H, n_actions) — probability of each action at each step
        action_probs = torch.ones(
            self.horizon, self.n_actions,
            device=self.device
        ) / self.n_actions

        best_sequence = None
        best_score    = float("inf")

        z_goal_norm = F.normalize(z_goal, dim=-1)  # (1, latent_dim)

        for iteration in range(self.n_iterations):

            # ── Sample action sequences from current distribution ─────
            sequences = torch.zeros(
                self.n_samples, self.horizon,
                dtype=torch.long, device=self.device
            )
            for h in range(self.horizon):
                sequences[:, h] = torch.multinomial(
                    action_probs[h].unsqueeze(0).expand(
                        self.n_samples, -1
                    ),
                    num_samples=1
                ).squeeze(1)

            # ── Imagine each sequence, score at every step ────────────
            N  = self.n_samples
            z  = z_current.expand(N, -1).clone()
            cumulative_scores = torch.zeros(N, device=self.device)

            for h in range(self.horizon):
                actions = sequences[:, h]       # (N,)
                z_prev  = z.clone()

                z = self.predictor(z, actions)  # (N, latent_dim)
                z = F.normalize(z, dim=-1)

                # ── 1. Distance to goal (discounted) ──────────────────
                discount  = 0.9 ** (self.horizon - h)
                step_dist = 1 - F.cosine_similarity(
                    z, z_goal_norm.expand(N, -1), dim=-1
                )
                cumulative_scores += discount * step_dist

                # ── 2. Movement penalty ───────────────────────────────
                # If z barely changed, agent likely hit a wall
                # state_change ≈ 0.0 → no movement → penalize
                # state_change ≈ 0.1+ → real movement → no penalty
                state_change = 1 - F.cosine_similarity(
                    z, z_prev, dim=-1
                )
                no_movement_penalty = F.relu(
                    0.05 - state_change
                ) * 3.0
                cumulative_scores += no_movement_penalty

                # ── 3. Turn penalty ───────────────────────────────────
                # Penalize consecutive turns (spinning in place)
                # Two turns in a row = wasted steps
                if h > 0:
                    prev_actions = sequences[:, h-1]
                    both_turns   = (
                        (actions <= 1) & (prev_actions <= 1)
                    ).float()
                    cumulative_scores += both_turns * 0.5

            # ── Elite selection ───────────────────────────────────────
            elite_indices = cumulative_scores.argsort()[:self.n_elite]
            elite_seqs    = sequences[elite_indices]
            elite_scores  = cumulative_scores[elite_indices]

            if elite_scores[0] < best_score:
                best_score    = elite_scores[0].item()
                best_sequence = elite_seqs[0]

            # ── Refit distribution from elite sequences ───────────────
            new_probs = torch.zeros(
                self.horizon, self.n_actions,
                device=self.device
            )
            for h in range(self.horizon):
                for a in range(self.n_actions):
                    new_probs[h, a] = (
                        elite_seqs[:, h] == a
                    ).float().sum()

            # Normalize + smoothing to avoid zero probabilities
            new_probs    = new_probs + 0.1
            new_probs    = new_probs / new_probs.sum(dim=1, keepdim=True)

            # Blend old and new (momentum keeps some exploration)
            action_probs = 0.3 * action_probs + 0.7 * new_probs

        best_action = best_sequence[0].item()
        return best_action, best_score


class CEMNavigator:
    """
    Runs the full navigation loop:
    - Collects reference frames for goal from same env instance
    - Navigates using CEM planner
    - Visualizes results
    """
    def __init__(self, checkpoint_path):
        self.planner = CEMPlanner(
            checkpoint_path = checkpoint_path,
            horizon         = 20,
            n_samples       = 1000,
            n_elite         = 100,
            n_iterations    = 7,
        )
        self.env = gym.make(
            "MiniGrid-FourRooms-v0",
            render_mode="rgb_array"
        )

    def collect_goal_candidates(self, n_steps=150):
        """
        Explore randomly to collect reference frames.
        Returns frames and positions from THIS environment instance
        so latent vectors are consistent with the navigation episode.
        """
        frames    = []
        positions = []
        self.env.reset()

        for _ in range(n_steps):
            action = np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.50])
            self.env.step(action)
            frames.append(self.env.render())
            positions.append(tuple(
                int(x) for x in self.env.unwrapped.agent_pos
            ))

        return frames, positions

    def run_episode(self, goal_distance_range=(8, 20), max_steps=200):
        """
        Run one navigation episode.

        1. Explore to build goal candidates
        2. Pick goal within distance range
        3. Navigate using CEM planner
        4. Return trajectory data
        """
        # ── Collect goal candidates ───────────────────────────────────
        print("  Exploring to find goal candidates...")
        ref_frames, ref_positions = self.collect_goal_candidates(150)

        # Reset to fresh start
        self.env.reset()
        start_pos = tuple(int(x) for x in self.env.unwrapped.agent_pos)

        # Pick goal within distance range
        valid_goals = [
            (i, p) for i, p in enumerate(ref_positions)
            if goal_distance_range[0]
            <= np.sqrt(
                (p[0]-start_pos[0])**2 + (p[1]-start_pos[1])**2
            )
            <= goal_distance_range[1]
        ]

        if not valid_goals:
            print("  No valid goals in range, using furthest point")
            dists    = [
                np.sqrt(
                    (p[0]-start_pos[0])**2 + (p[1]-start_pos[1])**2
                )
                for p in ref_positions
            ]
            goal_idx = int(np.argmax(dists))
        else:
            goal_idx = valid_goals[len(valid_goals)//2][0]

        goal_frame    = ref_frames[goal_idx]
        goal_position = ref_positions[goal_idx]
        z_goal        = self.planner.encode_frame(goal_frame)

        straight_line = np.sqrt(
            (start_pos[0]-goal_position[0])**2 +
            (start_pos[1]-goal_position[1])**2
        )
        print(f"  Start: {start_pos}")
        print(f"  Goal:  {goal_position}")
        print(f"  Straight-line distance: {straight_line:.1f} cells")

        # ── Navigate ──────────────────────────────────────────────────
        frames        = [self.env.render()]
        positions     = [start_pos]
        actions_taken = []
        plan_scores   = []
        done          = False
        step          = 0

        while not done and step < max_steps:
            frame     = self.env.render()
            z_current = self.planner.encode_frame(frame)

            action, score = self.planner.plan(z_current, z_goal)
            plan_scores.append(score)
            actions_taken.append(action)

            _, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            frames.append(self.env.render())
            positions.append(
                tuple(int(x) for x in self.env.unwrapped.agent_pos)
            )
            step += 1

            if step % 25 == 0:
                curr = positions[-1]
                d    = np.sqrt(
                    (curr[0]-goal_position[0])**2 +
                    (curr[1]-goal_position[1])**2
                )
                print(f"  Step {step:>3}: pos={curr}  "
                      f"dist_to_goal={d:.1f}  "
                      f"plan_score={score:.3f}")

        final_pos  = positions[-1]
        final_dist = np.sqrt(
            (final_pos[0]-goal_position[0])**2 +
            (final_pos[1]-goal_position[1])**2
        )
        print(f"  Final distance to goal: {final_dist:.1f} cells "
              f"(started {straight_line:.1f} away)")
        improvement = ((straight_line - final_dist) / straight_line) * 100
        print(f"  Improvement: {improvement:.1f}%")

        return dict(
            frames        = frames,
            positions     = positions,
            goal_frame    = goal_frame,
            goal_position = goal_position,
            start_pos     = start_pos,
            final_dist    = final_dist,
            straight_line = straight_line,
            plan_scores   = plan_scores,
            actions_taken = actions_taken,
            steps         = step,
            improvement   = improvement,
        )

    def visualize(self, result, episode_num, save=True):
        """Full visualization of one navigation episode."""
        positions     = result["positions"]
        goal_position = result["goal_position"]
        start_pos     = result["start_pos"]
        frames        = result["frames"]
        plan_scores   = result["plan_scores"]

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Episode {episode_num} — CEM World Model Navigation\n"
            f"Start: {start_pos}   Goal: {goal_position}   "
            f"Final dist: {result['final_dist']:.1f} cells   "
            f"Improvement: {result['improvement']:.1f}%   "
            f"Steps: {result['steps']}",
            fontsize=11, fontweight="bold"
        )

        # ── Top row: key frames ───────────────────────────────────────
        n_frames    = len(frames)
        key_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4, -1]
        key_labels  = ["Start", "25%", "50%", "75%", "End"]

        for i, (fidx, label) in enumerate(zip(key_indices, key_labels)):
            ax = fig.add_subplot(3, 6, i+1)
            ax.imshow(frames[fidx])
            ax.set_title(f"{label}\n{positions[fidx]}", fontsize=8)
            ax.axis("off")

        ax_goal = fig.add_subplot(3, 6, 6)
        ax_goal.imshow(result["goal_frame"])
        ax_goal.set_title(
            f"GOAL\n{goal_position}",
            fontsize=8, color="red", fontweight="bold"
        )
        ax_goal.axis("off")

        # ── Trajectory ────────────────────────────────────────────────
        ax_traj = fig.add_subplot(3, 2, 3)
        xs      = [p[0] for p in positions]
        ys      = [p[1] for p in positions]
        colors  = plt.cm.viridis(np.linspace(0, 1, len(xs)))
        for i in range(len(xs)-1):
            ax_traj.plot(
                [xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=colors[i], linewidth=2, alpha=0.9
            )
        ax_traj.scatter(
            xs[0], ys[0], color="green", s=200,
            zorder=5, label=f"Start {start_pos}"
        )
        ax_traj.scatter(
            xs[-1], ys[-1], color="blue", s=200,
            zorder=5, label=f"End {positions[-1]}"
        )
        ax_traj.scatter(
            goal_position[0], goal_position[1],
            color="red", marker="*", s=400,
            zorder=5, label=f"Goal {goal_position}"
        )
        ax_traj.set_xlim(0, 19)
        ax_traj.set_ylim(0, 19)
        ax_traj.set_title(
            "Agent Trajectory\n(purple=start, yellow=end)"
        )
        ax_traj.legend(fontsize=7)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")

        # ── Plan score over time ──────────────────────────────────────
        ax_score = fig.add_subplot(3, 2, 4)
        ax_score.plot(plan_scores, color="steelblue", linewidth=1.5)
        if len(plan_scores) > 10:
            z = np.polyfit(range(len(plan_scores)), plan_scores, 1)
            p = np.poly1d(z)
            ax_score.plot(
                range(len(plan_scores)),
                p(range(len(plan_scores))),
                "r--", linewidth=1.5, alpha=0.7,
                label=f"Trend (slope={z[0]:.4f})"
            )
            ax_score.legend(fontsize=8)
        ax_score.set_xlabel("Step")
        ax_score.set_ylabel("CEM score")
        ax_score.set_title(
            "CEM Plan Score Over Time\n"
            "(lower = planner more confident)"
        )
        ax_score.grid(True, alpha=0.3)

        # ── Action distribution ───────────────────────────────────────
        ax_act = fig.add_subplot(3, 2, 5)
        action_names  = ["Turn Left", "Turn Right", "Move Forward"]
        action_counts = [result["actions_taken"].count(a) for a in range(3)]
        bars = ax_act.bar(
            action_names, action_counts,
            color=["#e74c3c", "#3498db", "#2ecc71"]
        )
        ax_act.set_title("Action Distribution")
        ax_act.set_ylabel("Count")
        for bar, count in zip(bars, action_counts):
            ax_act.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(count), ha="center", fontsize=9
            )

        # ── Actual distance to goal over time ─────────────────────────
        ax_gdist = fig.add_subplot(3, 2, 6)
        grid_dists = [
            np.sqrt(
                (p[0]-goal_position[0])**2 +
                (p[1]-goal_position[1])**2
            )
            for p in positions
        ]
        ax_gdist.plot(grid_dists, color="darkorange", linewidth=1.5)
        ax_gdist.axhline(
            y=result["straight_line"], color="gray",
            linestyle="--", alpha=0.5,
            label=f"Start dist: {result['straight_line']:.1f}"
        )
        ax_gdist.set_xlabel("Step")
        ax_gdist.set_ylabel("Grid cells to goal")
        ax_gdist.set_title(
            "Actual Distance to Goal Over Time\n(grid cells — lower is better)"
        )
        ax_gdist.legend(fontsize=8)
        ax_gdist.grid(True, alpha=0.3)

        plt.tight_layout()
        Path("notebooks").mkdir(exist_ok=True)
        if save:
            path = f"notebooks/cem_episode_{episode_num}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.show()


if __name__ == "__main__":
    navigator = CEMNavigator("checkpoints/jepa_final.pt")

    all_improvements = []

    for episode in range(3):
        print(f"\nEpisode {episode+1}/3")
        print("-" * 40)
        result = navigator.run_episode(
            goal_distance_range = (8, 20),
            max_steps           = 200,
        )
        navigator.visualize(result, episode_num=episode+1)
        all_improvements.append(result["improvement"])

    navigator.env.close()

    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    for i, imp in enumerate(all_improvements):
        print(f"Episode {i+1}: {imp:.1f}% improvement")
    print(f"Average:   {np.mean(all_improvements):.1f}% improvement")