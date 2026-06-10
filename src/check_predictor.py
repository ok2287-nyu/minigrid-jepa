import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'src')
from encoder_v2 import Encoder, Predictor
from data_collector_v2 import ReplayBuffer

# Load
ckpt      = torch.load("checkpoints/jepa_phase1_final.pt", map_location="cpu", weights_only=False)
encoder   = Encoder(256)
predictor = Predictor(256, 3)
encoder.load_state_dict(ckpt["online_encoder"])
predictor.load_state_dict(ckpt["predictor"])
encoder.eval()
predictor.eval()

buffer = ReplayBuffer(capacity=200_000)
buffer.load("data/replay_buffer_phase1.pkl")

# Sample 100 transitions
idxs = np.random.randint(0, buffer.size, size=100)

obs_t   = torch.tensor(buffer.obs[idxs],      dtype=torch.float32) / 255.0
obs_t1  = torch.tensor(buffer.next_obs[idxs], dtype=torch.float32) / 255.0
dirs    = torch.tensor(buffer.directions[idxs],      dtype=torch.long)
ndirs   = torch.tensor(buffer.next_directions[idxs], dtype=torch.long)
actions = torch.tensor(buffer.actions[idxs], dtype=torch.long)

with torch.no_grad():
    z_t  = encoder(obs_t)
    z_t1 = encoder(obs_t1)

    dir_oh  = F.one_hot(dirs,  num_classes=4).float()
    ndir_oh = F.one_hot(ndirs, num_classes=4).float()

    state_t  = torch.cat([z_t,  dir_oh],  dim=-1)  # (100, 260)
    state_t1 = torch.cat([z_t1, ndir_oh], dim=-1)  # (100, 260)

    # Predict next state
    state_pred = predictor(state_t, actions)  # (100, 260)

    # Cosine distance between predicted and actual
    dist_pred_actual = 1 - F.cosine_similarity(
        F.normalize(state_pred, dim=-1),
        F.normalize(state_t1,   dim=-1),
        dim=-1
    )

    # Cosine distance between current and actual (baseline)
    dist_curr_actual = 1 - F.cosine_similarity(
        F.normalize(state_t,  dim=-1),
        F.normalize(state_t1, dim=-1),
        dim=-1
    )

print(f"Predictor distance to actual next state: {dist_pred_actual.mean():.4f}")
print(f"Current state distance to actual next:   {dist_curr_actual.mean():.4f}")
print(f"(predictor should be LOWER than current)")
# Add this to check_predictor.py
print(f"\nInvariance loss (should be ~0):")
inv = F.mse_loss(
    F.normalize(state_pred, dim=-1),
    F.normalize(state_t1,   dim=-1)
)
print(f"  {inv.item():.4f}")
# Check direction prediction accuracy
dir_pred    = state_pred[:, 256:].argmax(dim=-1)
dir_actual  = ndirs
dir_acc     = (dir_pred == dir_actual).float().mean()
print(f"\nDirection prediction accuracy: {dir_acc*100:.1f}%")
# Add to check_predictor.py

# Load target encoder too
target_encoder = Encoder(256)
target_encoder.load_state_dict(ckpt["target_encoder"])
target_encoder.eval()

with torch.no_grad():
    z_t1_target = target_encoder(obs_t1)
    ndir_oh_t   = F.one_hot(ndirs, num_classes=4).float()
    state_t1_target = torch.cat([z_t1_target, ndir_oh_t], dim=-1)

    dist_pred_target = 1 - F.cosine_similarity(
        F.normalize(state_pred,       dim=-1),
        F.normalize(state_t1_target,  dim=-1),
        dim=-1
    )

print(f"Predictor vs TARGET encoder next state: {dist_pred_target.mean():.4f}")
print(f"Predictor vs ONLINE encoder next state: {dist_pred_actual.mean():.4f}")

# Check z part only (first 256 dims)
z_pred_only   = state_pred[:, :256]
z_actual_only = state_t1[:, :256]

dist_z_only = 1 - F.cosine_similarity(
    F.normalize(z_pred_only,   dim=-1),
    F.normalize(z_actual_only, dim=-1),
    dim=-1
)

# Check direction part only (last 4 dims)
dir_pred_only   = state_pred[:, 256:]
dir_actual_only = state_t1[:, 256:]

dist_dir_only = 1 - F.cosine_similarity(
    F.normalize(dir_pred_only,   dim=-1),
    F.normalize(dir_actual_only, dim=-1),
    dim=-1
)

print(f"\nZ part distance (256 dims):         {dist_z_only.mean():.4f}")
print(f"Direction part distance (4 dims):   {dist_dir_only.mean():.4f}")
print(f"Full state distance (260 dims):     {dist_pred_actual.mean():.4f}")