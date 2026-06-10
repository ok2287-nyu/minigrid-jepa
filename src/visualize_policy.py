import os
from pathlib import Path
import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

from encoder_v2 import Encoder
from controller_bc import Controller

PROJECT_ROOT = Path(__file__).parent.parent
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)

class PolicyVisualizer:
    def __init__(self, jepa_checkpoint, dagger_checkpoint, env_id="MiniGrid-Empty-16x16-v0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Environment configuration
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset()
        self.width = self.env.unwrapped.width
        self.height = self.env.unwrapped.height
        self.max_x = self.width - 2
        self.max_y = self.height - 2
        
        # Walkable cells inside boundary walls
        self.x_range = range(1, self.width - 1)
        self.y_range = range(1, self.height - 1)

        # 2. Load Networks
        self.encoder = Encoder(latent_dim=256).to(self.device)
        jepa_ckpt = torch.load(jepa_checkpoint, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(jepa_ckpt["online_encoder"])
        self.encoder.eval()

        self.controller = Controller(state_dim=260, hidden_dim=512, n_actions=3, pos_dim=2).to(self.device)
        dagger_ckpt = torch.load(dagger_checkpoint, map_location=self.device, weights_only=False)
        self.controller.load_state_dict(dagger_ckpt["controller"])
        self.controller.eval()

    def preprocess(self, frame):
        img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def build_state(self, z, direction):
        dir_t = torch.tensor([direction], dtype=torch.long).to(self.device)
        dir_oh = F.one_hot(dir_t, num_classes=4).float()
        return torch.cat([z, dir_oh], dim=-1)

    def build_pos(self, x, y):
        return torch.tensor([[x / self.max_x, y / self.max_y]], dtype=torch.float32).to(self.device)

    def set_agent_state(self, x, y, direction):
        self.env.unwrapped.agent_pos = np.array([x, y])
        self.env.unwrapped.agent_dir = direction

    def generate_policy_map(self, goal_x=7, goal_y=7, goal_dir=0):
        """Generates a 4-panel vector field plot showing the controller's decision

        for every single position on the map, facing all 4 directions.
        """
        print(f"Generating policy map converging on goal: ({goal_x}, {goal_y}) facing {['East','South','West','North'][goal_dir]}...")
        
        # 1. Encode the fixed evaluation target goal
        self.env.reset()
        self.set_agent_state(goal_x, goal_y, goal_dir)
        goal_frame = self.env.render()
        with torch.no_grad():
            z_goal = self.encoder(self.preprocess(goal_frame))
            state_goal = self.build_state(z_goal, goal_dir)
            pos_goal = self.build_pos(goal_x, goal_y)

        # Map directions to string labels and plotting arrow components (dx, dy)
        # MiniGrid tracking convention: 0=East (+x), 1=South (+y), 2=West (-x), 3=North (-y)
        dir_labels = ["Facing East (+X)", "Facing South (+Y)", "Facing West (-X)", "Facing North (-Y)"]
        move_vectors = {0: (0.4, 0), 1: (0, -0.4), 2: (-0.4, 0), 3: (0, 0.4)} # Matplotlib Y is inverted vs grid array

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        # Generate a standalone policy grid map for each compass direction the agent could be facing
        for current_facing in range(4):
            ax = axes[current_facing]
            ax.set_title(dir_labels[current_facing], fontsize=12, fontweight='bold')
            
            # Draw empty grid arena limits
            ax.set_xlim(0, self.width - 1)
            ax.set_ylim(self.height - 1, 0) # Invert Y axis to match grid layout visually
            ax.set_xticks(range(self.width))
            ax.set_yticks(range(self.height))
            ax.grid(True, which='both', color='#ddd', linestyle='-', linewidth=0.5)

            # Mark the static goal spot clearly
            ax.plot(goal_x + 0.5, goal_y + 0.5, 'ro', markersize=12, label="Goal Target")
            
            # Systematically probe every walkable cell matrix position
            for x in self.x_range:
                for y in self.y_range:
                    if x == goal_x and y == goal_y:
                        continue # Skip goal tile itself

                    # Teleport agent to probe this specific state
                    self.set_agent_state(x, y, current_facing)
                    frame = self.env.render()
                    
                    with torch.no_grad():
                        z_curr = self.encoder(self.preprocess(frame))
                        state_curr = self.build_state(z_curr, current_facing)
                        pos_curr = self.build_pos(x, y)
                        
                        # Fetch standalone action probability selection profiles
                        action_probs = self.controller(state_curr, state_goal, pos_curr, pos_goal)
                        action = action_probs.argmax(dim=-1).item()

                    # Render arrows centered on individual tiles
                    cx, cy = x + 0.5, y + 0.5

                    if action == 2: # FORWARD: Draw arrow matching current agent alignment direction
                        dx, dy = move_vectors[current_facing]
                        ax.arrow(cx - dx/2, cy - dy/2, dx, dy, head_width=0.25, head_length=0.15, fc='#2ecc71', ec='#27ae60')
                    elif action == 0: # LEFT TURN: Draw a blue counter-clockwise indicator circle/arc block
                        ax.plot(cx, cy, marker='$\circlearrowleft$', color='#3498db', markersize=14)
                    elif action == 1: # RIGHT TURN: Draw a purple clockwise indicator circle/arc block
                        ax.plot(cx, cy, marker='$\circlearrowright$', color='#9b59b6', markersize=14)

        plt.suptitle(f"DAgger Controller Policy Flow Field\nGoal Target locked at ({goal_x}, {goal_y})", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save output image directly to project workspace
        output_path = "checkpoints/controller_policy_manifold.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Policy manifold map successfully generated and saved to: {output_path}")
        plt.show()

if __name__ == "__main__":
    visualizer = PolicyVisualizer(
        jepa_checkpoint="checkpoints/jepa_phase1_final.pt",
        dagger_checkpoint="checkpoints/controller_dagger_dagger_final.pt"
    )
    # Probe convergence vectors towards an arbitrary mid-grid point scenario
    visualizer.generate_policy_map(goal_x=8, goal_y=5, goal_dir=0)