import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    CNN encoder: maps raw pixel observations to latent vectors.

    Input:  (B, 3, 64, 64)  - batch of RGB images, normalized 0-1
    Output: (B, latent_dim) - batch of latent vectors

    Architecture: 4 conv layers with progressively more filters,
    each halving spatial dimensions, followed by a linear projection.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Each conv layer: learns spatial filters at increasing abstraction
        # (in_channels, out_channels, kernel_size, stride, padding)
        self.conv_layers = nn.Sequential(
            # Layer 1: 3 → 32 channels, 64×64 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 32 → 64 channels, 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 64 → 128 channels, 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: 128 → 256 channels, 8×8 → 4×4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # After 4 conv layers: spatial dims are 4×4, channels are 256
        # Flatten → 256 * 4 * 4 = 4096 values
        self.flatten_dim = 256 * 4 * 4  # = 4096

        # Project from 4096 → latent_dim
        # This is the actual "latent vector" the world model operates on
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim),  # stabilizes training
        )

    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.conv_layers(x)     # (B, 256, 4, 4)
        x = x.flatten(start_dim=1)  # (B, 4096)
        x = self.projection(x)      # (B, latent_dim)
        return x


class Predictor(nn.Module):
    """
    Predicts next latent state given current latent state + action.

    This IS the world model — it learns the transition dynamics:
    "if I'm in state z and take action a, I'll end up in state z'"

    Input:  z (B, latent_dim) + action (B,)
    Output: z_next_predicted (B, latent_dim)
    """
    def __init__(self, latent_dim=256, n_actions=3, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim

        # Embed discrete action into a continuous vector
        # so it can be concatenated with z
        self.action_embedding = nn.Embedding(n_actions, 32)

        # MLP that takes [z, action_embed] and predicts z_next
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        # z:      (B, latent_dim)
        # action: (B,)  — integer action indices

        a = self.action_embedding(action)  # (B, 32)
        x = torch.cat([z, a], dim=1)       # (B, latent_dim + 32)
        return self.net(x)                 # (B, latent_dim)


if __name__ == "__main__":
    # Sanity check — verify shapes through the full forward pass
    B = 32
    latent_dim = 256

    encoder   = Encoder(latent_dim=latent_dim)
    predictor = Predictor(latent_dim=latent_dim, n_actions=3)

    # Simulate a batch from the replay buffer
    obs     = torch.randn(B, 3, 64, 64)   # random "images"
    actions = torch.randint(0, 3, (B,))   # random actions

    # Forward pass
    z           = encoder(obs)             # (B, 256)
    z_predicted = predictor(z, actions)   # (B, 256)

    print(f"Encoder input:     {obs.shape}")
    print(f"Latent vector z:   {z.shape}")
    print(f"Predicted z_next:  {z_predicted.shape}")
    print(f"Latent dim:        {latent_dim}")
    print(f"\nEncoder parameters:   {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")