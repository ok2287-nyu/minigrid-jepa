import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Identical to v1. Untouched.
    CNN encoder: maps raw pixel observations to latent vectors.
    Input:  (B, 3, 64, 64)
    Output: (B, latent_dim)
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = 256 * 4 * 4

        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.projection(x)
        return x


class ActionEncoder(nn.Module):
    """
    Replaces the simple nn.Embedding lookup table.

    Takes a discrete action (0, 1, 2) and produces a rich
    continuous embedding through a small MLP.

    The key difference from a lookup table:
    - Lookup table: 3 vectors, learned independently, no constraint
    - ActionEncoder: 3 vectors produced by a network
                     + covariance loss forces them spatially apart

    This prevents action embedding collapse — where all three
    actions map to nearly identical vectors, causing the predictor
    to ignore action input entirely.
    """
    def __init__(self, n_actions=3, action_dim=64):
        super().__init__()
        self.n_actions  = n_actions
        self.action_dim = action_dim

        # One-hot input → rich embedding via MLP
        self.net = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.LayerNorm(action_dim),
        )

    def forward(self, action):
        """
        action: (B,) integer tensor of action indices
        returns: (B, action_dim) continuous embeddings
        """
        # Convert integer actions to one-hot vectors
        # One-hot gives the network explicit discrete structure to work with
        one_hot = F.one_hot(
            action, num_classes=self.n_actions
        ).float()                           # (B, n_actions)
        return self.net(one_hot)            # (B, action_dim)

    def get_all_embeddings(self):
        """
        Returns embeddings for ALL actions: (n_actions, action_dim)
        Used to compute covariance loss across action vectors.
        """
        all_actions = torch.arange(
            self.n_actions,
            device=next(self.parameters()).device
        )
        return self.forward(all_actions)    # (n_actions, action_dim)

    def covariance_loss(self):
        """
        Forces the three action embeddings to be spatially distinct.

        Gets all 3 action vectors: (3, action_dim)
        Computes covariance matrix across them.
        Penalizes off-diagonal entries — correlation between actions.

        If action embeddings collapse (all similar):
          off-diagonal covariance is high → loss spikes
          gradients push embeddings apart

        If action embeddings are distinct:
          off-diagonal covariance is low → no penalty
        """
        # Get all 3 action embeddings: (3, action_dim)
        A = self.get_all_embeddings()

        # Center the embeddings
        A_centered = A - A.mean(dim=0)     # (3, action_dim)

        # Covariance matrix across the 3 action vectors
        # (action_dim, action_dim)
        cov = (A_centered.T @ A_centered) / (self.n_actions - 1)

        # Penalize off-diagonal entries only
        # Diagonal = each dimension's own variance (fine)
        # Off-diagonal = correlation between dimensions (bad)
        off_diag = cov ** 2
        off_diag.fill_diagonal_(0)

        return off_diag.sum() / self.action_dim


class Predictor(nn.Module):
    """
    Predicts next latent state given current latent state + action.

    Key change from v1:
    Uses ActionEncoder instead of nn.Embedding.
    Action embedding dimension increased from 32 to 64.
    Predictor input: latent_dim + action_dim = 256 + 64 = 320
    """
    def __init__(self, latent_dim=256, n_actions=3,
             hidden_dim=512, action_dim=64, n_dirs=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim  = latent_dim + n_dirs   # 256 + 4 = 260

        self.action_encoder = ActionEncoder(n_actions, action_dim)

        # Input: full state (260) + action embedding (64) = 324
        # Output: full next state (260)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.state_dim),  # predicts 260
        )
    def forward(self, state, action):
        """
        state:  (B, state_dim) = [z (256) | direction_onehot (4)]
        action: (B,) integer action indices
        returns: (B, state_dim) predicted next state
        """
        a = self.action_encoder(action)        # (B, action_dim)
        x = torch.cat([state, a], dim=1)       # (B, state_dim + action_dim)
        return self.net(x)                     # (B, state_dim)

    def action_covariance_loss(self):
        """Delegate to action encoder's covariance loss."""
        return self.action_encoder.covariance_loss()


if __name__ == "__main__":
    B          = 32
    latent_dim = 256

    encoder   = Encoder(latent_dim=latent_dim)
    predictor = Predictor(latent_dim=latent_dim, n_actions=3)

    obs     = torch.randn(B, 3, 64, 64)
    actions = torch.randint(0, 3, (B,))

    z          = encoder(obs)                              # (B, 256)
    directions = torch.randint(0, 4, (B,))
    dir_onehot = F.one_hot(directions, num_classes=4).float()  # (B, 4)
    state      = torch.cat([z, dir_onehot], dim=-1)        # (B, 260)

    state_next = predictor(state, actions)                 # (B, 260)

    print(f"Encoder input:      {obs.shape}")
    print(f"Latent vector z:    {z.shape}")
    print(f"Full state:         {state.shape}")
    print(f"Predicted state:    {state_next.shape}")

    # Verify action embeddings are distinct
    all_embeds = predictor.action_encoder.get_all_embeddings()
    print(f"\nAction embeddings shape: {all_embeds.shape}")

    a0, a1, a2 = all_embeds[0], all_embeds[1], all_embeds[2]
    print(f"Distance left vs right:   "
          f"{(1 - F.cosine_similarity(a0.unsqueeze(0), a1.unsqueeze(0))).item():.4f}")
    print(f"Distance left vs forward: "
          f"{(1 - F.cosine_similarity(a0.unsqueeze(0), a2.unsqueeze(0))).item():.4f}")
    print(f"Distance right vs forward:"
          f"{(1 - F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0))).item():.4f}")

    cov_loss = predictor.action_covariance_loss()
    print(f"\nAction covariance loss: {cov_loss.item():.4f}")
    print(f"(should decrease toward 0 during training)")

    print(f"\nEncoder parameters:   "
          f"{sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor parameters: "
          f"{sum(p.numel() for p in predictor.parameters()):,}")