import torch
import torch.nn.functional as F
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder, Predictor

ckpt = torch.load("checkpoints/jepa_5000.pt", 
                   map_location="cpu", 
                   weights_only=False)

predictor = Predictor(latent_dim=256, n_actions=3)
predictor.load_state_dict(ckpt["predictor"])
predictor.eval()

# Check embedding distances
all_embeds = predictor.action_encoder.get_all_embeddings()
a0, a1, a2 = all_embeds[0], all_embeds[1], all_embeds[2]
d01 = (1 - F.cosine_similarity(a0.unsqueeze(0), a1.unsqueeze(0))).item()
d02 = (1 - F.cosine_similarity(a0.unsqueeze(0), a2.unsqueeze(0))).item()
d12 = (1 - F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0))).item()
print(f"Embedding distances:")
print(f"  L-R={d01:.4f}  L-F={d02:.4f}  R-F={d12:.4f}")

# Check prediction distances from a random z
z = F.normalize(torch.randn(1, 256), dim=-1)
a0t = torch.tensor([0])
a1t = torch.tensor([1])
a2t = torch.tensor([2])

with torch.no_grad():
    z0 = F.normalize(predictor(z, a0t), dim=-1)
    z1 = F.normalize(predictor(z, a1t), dim=-1)
    z2 = F.normalize(predictor(z, a2t), dim=-1)

p01 = (1 - F.cosine_similarity(z0, z1)).item()
p02 = (1 - F.cosine_similarity(z0, z2)).item()
p12 = (1 - F.cosine_similarity(z1, z2)).item()
print(f"Prediction distances:")
print(f"  L-R={p01:.4f}  L-F={p02:.4f}  R-F={p12:.4f}")