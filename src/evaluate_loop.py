import os
from pathlib import Path
import gymnasium as gym
import minigrid
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from controller_bc import Controller
from encoder_v2 import Encoder

# Match file structure paths from context files
PROJECT_ROOT = Path(__file__).parent.parent
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)


class AgentEvaluator:
    """The Evaluation Loop for Stage 3 Goal-Conditioned Behavioural Cloning."""

    def __init__(
        self,
        checkpoint_path,
        encoder_path,
        env_id="MiniGrid-Empty-16x16-v0",
        img_size=64,
        latent_dim=256,
        n_dirs=4,
        n_actions=3,
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.img_size = img_size
        self.n_dirs = n_dirs

        # 1. Initialize gym environment
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width = self.env.unwrapped.width
        self.height = self.env.unwrapped.height

        # Valid layout boundaries (excluding the outer 1-cell thick boundary walls)
        self.x_range = list(range(1, self.width - 1))
        self.y_range = list(range(1, self.height - 1))

        # 2. Load and freeze Encoder
        self.encoder = Encoder(latent_dim).to(self.device)
        enc_ckpt = torch.load(
            encoder_path, map_location=self.device, weights_only=False
        )
        self.encoder.load_state_dict(enc_ckpt["online_encoder"])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3. Load and prepare Controller
        state_dim = latent_dim + n_dirs  # 260
        self.controller = Controller(
            state_dim=state_dim, hidden_dim=512, n_actions=n_actions
        ).to(self.device)
        cont_ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.controller.load_state_dict(cont_ckpt["controller"])
        self.controller.eval()

        print(f"Evaluation pipeline ready on {self.device}.")
        print(f"Loaded controller from: {checkpoint_path}")
        print(f"Loaded encoder from:    {encoder_path}")

    def preprocess(self, frame):
        """Converts raw RGB frame to a (1, 3, 64, 64) normalized float32 tensor."""
        img = Image.fromarray(frame).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
        tensor = (
            torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            / 255.0
        )
        return tensor

    def set_agent_state(self, x, y, direction):
        """Direct state injection matching SystematicDataCollector exactly."""
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def build_state_vector(self, z, direction):
        """Combines latent image embedding vector z (256,) with direction one-hot (4,)."""
        dir_tensor = torch.tensor([direction], dtype=torch.long).to(self.device)
        dir_onehot = F.one_hot(dir_tensor, num_classes=self.n_dirs).float()
        return torch.cat([z, dir_onehot], dim=-1)

    def sample_random_position(self):
        """Samples a valid cell inside the walkable environment parameters."""
        x = np.random.choice(self.x_range)
        y = np.random.choice(self.y_range)
        direction = np.random.randint(0, self.n_dirs)
        return x, y, direction

    def run_evaluation(self, num_episodes=10, max_steps=50):
        """Executes the systematic goal loop evaluation over multiple episodes."""
        print(
            f"\nStarting Evaluation Loop: {num_episodes} episodes (Max steps: {max_steps})"
        )

        successes = 0
        steps_recorded = []

        for episode in range(num_episodes):
            self.env.reset()
            

            # 1. Pick unique start and goal positions
            x_start, y_start, dir_start = self.sample_random_position()
            while True:
                x_goal, y_goal, dir_goal = self.sample_random_position()
                if (x_start, y_start) != (x_goal, y_goal):
                    break

            # 2. Encode Goal Configuration
            with torch.no_grad():
                self.set_agent_state(x_goal, y_goal, dir_goal)
                goal_frame = self.env.render()
                z_goal = self.encoder(self.preprocess(goal_frame))
                state_goal = self.build_state_vector(z_goal, dir_goal)

            # In run_evaluation, after picking goal:
            manhattan_dist = abs(x_goal - x_start) + abs(y_goal - y_start)
            # Skip trivially close goals
            if manhattan_dist < 3:
                continue
            # 3. Teleport agent back to start coordinates
            self.set_agent_state(x_start, y_start, dir_start)
            current_dir = dir_start
            episode_success = False

            # 4. Evaluation roll-out loop
            for step in range(1, max_steps + 1):
                with torch.no_grad():
                    
                    # Render, preprocess and encode current observation
                    current_frame = self.env.render()
                    z_current = self.encoder(self.preprocess(current_frame))
                    state_current = self.build_state_vector(
                        z_current, current_dir
                    )

                    # Query controller strategy
                    action_probs = self.controller(state_current, state_goal)
                    # action = torch.argmax(action_probs, dim=-1).item()
                    # Change sampling to argmax:
                    action = torch.argmax(action_probs, dim=-1).item()
 
                    # action_dist = torch.distributions.Categorical(action_probs)
                    # action = action_dist.sample().item() 
                # Step the environment engine
                _, _, _, _, _ = self.env.step(action)

                # Collect structural updates from the wrapper tracking properties
                agent_pos = tuple(self.env.unwrapped.agent_pos)
                current_dir = self.env.unwrapped.agent_dir
                # In the episode loop, print each step:
                print(f"  Step {step}: pos={agent_pos} dir={current_dir} "
                f"action={action} goal=({x_goal},{y_goal})")

                # 5. Success check logic
                if agent_pos == (x_goal, y_goal):
                    successes += 1
                    steps_recorded.append(step)
                    episode_success = True
                    break

            if not episode_success:
                pass  # Registered as a default step-cap boundary exhaustion failure

        # Metric Analytics Calculations
        success_rate = (successes / num_episodes) * 100
        avg_steps = np.mean(steps_recorded) if steps_recorded else 0.0

        print(f"\n" + "=" * 40)
        print(f"EVALUATION METRICS RESULTS:")
        print(f"=" * 40)
        print(f"Success Rate : {success_rate:.2f}%")
        print(f"Avg Steps    : {avg_steps:.2f} (Calculated over successes)")
        print(f"=" * 40 + "\n")

        return {"success_rate": success_rate, "avg_steps": avg_steps}


if __name__ == "__main__":
    # Point these parameters directly to your phase model parameters
    evaluator = AgentEvaluator(
        checkpoint_path="checkpoints/controller_bc_final.pt",
        encoder_path="checkpoints/jepa_phase1_final.pt",
        env_id="MiniGrid-Empty-16x16-v0",
    )

    evaluator.run_evaluation(num_episodes=100, max_steps=100)