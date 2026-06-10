import os
from pathlib import Path
import gymnasium as gym
import minigrid
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from controller_bc import Controller
from data_collector_v2 import ReplayBuffer
from encoder_v2 import Encoder

PROJECT_ROOT = Path(__file__).parent.parent
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)


class DAggerEvaluator:
    """Evaluates the DAgger-trained Controller over multi-step paths

    sampled directly from real transitions preserved inside the ReplayBuffer.
    """

    def __init__(
        self,
        jepa_checkpoint,
        dagger_checkpoint,
        buffer_path="data/replay_buffer_phase1.pkl",
        env_id="MiniGrid-Empty-16x16-v0",
        latent_dim=256,
        n_dirs=4,
        n_actions=3,
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.state_dim = latent_dim + n_dirs

        # 1. Initialize Gym Environment
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.max_x = self.width - 2
        self.max_y = self.height - 2

        # 2. Load and Freeze JEPA Vision Encoder
        self.encoder = Encoder(latent_dim).to(self.device)
        jepa_ckpt = torch.load(
            jepa_checkpoint, map_location=self.device, weights_only=False
        )
        self.encoder.load_state_dict(jepa_ckpt["online_encoder"])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # 3. Load Trained DAgger Controller
        self.controller = Controller(
            state_dim=self.state_dim, hidden_dim=512, n_actions=n_actions, pos_dim=2
        ).to(self.device)
        dagger_ckpt = torch.load(
            dagger_checkpoint, map_location=self.device, weights_only=False
        )
        self.controller.load_state_dict(dagger_ckpt["controller"])
        self.controller.eval()

        # 4. Load the Replay Buffer to draw test cases
        self.buffer = ReplayBuffer(capacity=200_000)
        self.buffer.load(buffer_path)

        print(f"Evaluator initialized on device: {self.device}")
        print(f"Successfully loaded buffer containing {self.buffer.size} steps.")

    def preprocess(self, frame):
        """Converts image matrix array to normalized tensor format."""
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return (
            torch.tensor(arr.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

    def encode_obs(self, frame):
        """Generates visual latent embeddings via frozen network encoder."""
        with torch.no_grad():
            return self.encoder(self.preprocess(frame))

    def build_state(self, z, direction):
        """Appends orientation classes tracking compass vectors onto latent vector z."""
        dir_t = torch.tensor([direction], dtype=torch.long).to(self.device)
        dir_oh = F.one_hot(dir_t, num_classes=4).float()
        return torch.cat([z, dir_oh], dim=-1)

    def build_pos(self, x, y):
        """Generates scale-normalized localization parameters."""
        return torch.tensor(
            [[x / self.max_x, y / self.max_y]], dtype=torch.float32
        ).to(self.device)

    def set_agent_state(self, x, y, direction):
        """Directly injects target coordinates into the engine wrapper."""
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def manual_step(self, x, y, d, action):
        """Simulates directional transitions forward to bypass env.step corruption."""
        if action == 0:  # Left Turn
            return x, y, (d - 1) % 4
        elif action == 1:  # Right Turn
            return x, y, (d + 1) % 4
        else:  # Move Forward
            dx = [1, 0, -1, 0][d]
            dy = [0, 1, 0, -1][d]
            nx, ny = x + dx, y + dy
            if 1 <= nx <= self.max_x and 1 <= ny <= self.max_y:
                return nx, ny, d
            return x, y, d

    def evaluate(self, num_tests=500, max_steps_multiplier=4):
        print(f"\nEvaluating DAgger Policy over {num_tests} test sequences...")
        print("=" * 60)

        success_count = 0
        total_steps_taken = []
        distance_buckets = {
            "Short (1-3)":  {"success": 0, "total": 0},
            "Medium (4-7)": {"success": 0, "total": 0},
            "Long (8+)":    {"success": 0, "total": 0},
        }

        self.env.reset()

        for test_idx in range(num_tests):
            # Sample start — same distribution as training
            x_start = np.random.randint(1, self.max_x + 1)
            y_start = np.random.randint(1, self.max_y + 1)
            d_start = np.random.randint(0, 4)

            # Sample goal with distance >= 1
            while True:
                x_goal = np.random.randint(1, self.max_x + 1)
                y_goal = np.random.randint(1, self.max_y + 1)
                d_goal = np.random.randint(0, 4)
                manhattan_dist = abs(x_start - x_goal) + abs(y_start - y_goal)
                if manhattan_dist >= 1:
                    break

            max_allowed_steps = max(15, manhattan_dist * max_steps_multiplier)

            if manhattan_dist <= 3:
                bucket = "Short (1-3)"
            elif manhattan_dist <= 7:
                bucket = "Medium (4-7)"
            else:
                bucket = "Long (8+)"
            distance_buckets[bucket]["total"] += 1

            # Encode goal
            self.set_agent_state(x_goal, y_goal, d_goal)
            goal_frame = self.env.render()
            z_goal = self.encode_obs(goal_frame)
            state_goal = self.build_state(z_goal, d_goal)
            pos_goal = self.build_pos(x_goal, y_goal)

            # Set start
            x, y, d = x_start, y_start, d_start
            success = False

            for step in range(1, max_allowed_steps + 1):
                self.set_agent_state(x, y, d)
                frame = self.env.render()
                z_current = self.encode_obs(frame)
                state_curr = self.build_state(z_current, d)
                pos_curr = self.build_pos(x, y)

                with torch.no_grad():
                    action_probs = self.controller(
                        state_curr, state_goal, pos_curr, pos_goal
                    )
                    predicted_action = action_probs.argmax(dim=-1).item()

                x, y, d = self.manual_step(x, y, d, predicted_action)

                if (x, y) == (x_goal, y_goal):
                    success = True
                    break

            if success:
                success_count += 1
                total_steps_taken.append(step)
                distance_buckets[bucket]["success"] += 1

        overall_sr = (success_count / num_tests) * 100
        avg_steps = np.mean(total_steps_taken) if total_steps_taken else 0.0

        print("\n" + "=" * 25 + " FINAL RESULTS " + "=" * 25)
        print(f"Overall Success Rate: {overall_sr:.2f}% ({success_count}/{num_tests})")
        print(f"Average Efficiency  : {avg_steps:.1f} steps (on success executions)")
        print("-" * 65)
        print("Performance Breakdown Across Distance Scales:")
        for name, data in distance_buckets.items():
            if data["total"] > 0:
                pct = (data["success"] / data["total"]) * 100
                print(f"  {name:13s} : {pct:>5.1f}% ({data['success']}/{data['total']})")
            else:
                print(f"  {name:13s} : No samples tested.")
        print("=" * 65)

        return {"success_rate": overall_sr, "avg_steps": avg_steps}

if __name__ == "__main__":
    evaluator = DAggerEvaluator(
        jepa_checkpoint="checkpoints/jepa_phase1_final.pt",
        dagger_checkpoint="checkpoints/controller_stage4_empty_stage4_empty_final.pt",
        buffer_path="data/replay_buffer_phase1.pkl",
    )

    evaluator.evaluate(num_tests=500, max_steps_multiplier=6)