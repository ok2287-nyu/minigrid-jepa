import torch
import torch.nn.functional as F
from pathlib import Path
import os
import gymnasium as gym
import minigrid
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

from encoder_v2 import Encoder, Predictor

ckpt = torch.load(
    "checkpoints/jepa_5000.pt",
    map_location="cpu",
    weights_only=False
)

encoder        = Encoder(256)
target_encoder = Encoder(256)
predictor      = Predictor(256, 3)

encoder.load_state_dict(ckpt["online_encoder"])
target_encoder.load_state_dict(ckpt["target_encoder"])
predictor.load_state_dict(ckpt["predictor"])

encoder.eval()
target_encoder.eval()
predictor.eval()

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env.reset()

def encode_frame(frame, enc):
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    t   = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        return enc(t)

# Test all three actions
for action in range(3):
    env.reset()
    frame_t  = env.render()

    z_online = encode_frame(frame_t, encoder)
    z_target = encode_frame(frame_t, target_encoder)

    # Predict
    a      = torch.tensor([action])
    z_pred = predictor(z_online, a)
    z_pred = F.normalize(z_pred, dim=-1)

    # Take action
    env.step(action)
    frame_t1 = env.render()

    z_actual_online = encode_frame(frame_t1, encoder)
    z_actual_target = encode_frame(frame_t1, target_encoder)

    dist_online = (1 - F.cosine_similarity(
        z_pred, F.normalize(z_actual_online, dim=-1)
    )).item()
    dist_target = (1 - F.cosine_similarity(
        z_pred, F.normalize(z_actual_target, dim=-1)
    )).item()

    action_name = ["Left", "Right", "Forward"][action]
    print(f"Action {action_name}:")
    print(f"  dist vs online encoder: {dist_online:.4f}")
    print(f"  dist vs target encoder: {dist_target:.4f}")

env.close()
