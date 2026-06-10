import torch
import torch.nn.functional as F
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder, Predictor

ckpt = torch.load(
    "checkpoints/jepa_5000.pt",
    map_location="cpu",
    weights_only=False
)

predictor = Predictor(latent_dim=256, n_actions=3)
predictor.load_state_dict(ckpt["predictor"])
predictor.eval()

z = F.normalize(torch.randn(1, 256), dim=-1).detach()

for action_idx in range(3):
    action = torch.tensor([action_idx])

    # Get action embedding as leaf tensor
    action_embed = predictor.action_encoder(action)
    action_embed.retain_grad()   # ← force grad storage on non-leaf

    # Concatenate with z and run predictor net directly
    x      = torch.cat([z, action_embed], dim=1)
    z_pred = predictor.net(x)
    z_pred = F.normalize(z_pred, dim=-1)

    loss   = z_pred.sum()
    loss.backward()

    grad_mag = action_embed.grad.abs().mean().item()
    print(f"Action {action_idx}: "
          f"embed_grad={grad_mag:.8f}  "
          f"grad_is_zero={grad_mag < 1e-6}")
# Inspect first layer weight magnitudes
first_layer = predictor.net[0]
W = first_layer.weight          # (512, 320)

W_z      = W[:, :256]           # weights for z
W_action = W[:, 256:]           # weights for action

print(f"\nWeight magnitudes:")
print(f"  z input:      {W_z.abs().mean():.6f}")
print(f"  action input: {W_action.abs().mean():.6f}")
print(f"  ratio:        {W_action.abs().mean() / W_z.abs().mean():.4f}")
print(f"\n  ratio=1.0 → action weighted equally to z")
print(f"  ratio=0.1 → action 10x suppressed vs z")
print(f"  ratio=0.0 → action completely ignored")