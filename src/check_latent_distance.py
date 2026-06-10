import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'src')
from encoder_v2 import Encoder
from data_collector_v2 import ReplayBuffer
import gymnasium as gym
import minigrid
from PIL import Image

# Load
ckpt    = torch.load("checkpoints/jepa_phase1_final.pt", 
                     map_location="cpu", weights_only=False)
encoder = Encoder(256)
encoder.load_state_dict(ckpt["online_encoder"])
encoder.eval()

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

def encode_pos(x, y, d):
    env.reset()
    env.unwrapped.agent_pos = np.array([x, y])
    env.unwrapped.agent_dir = d
    frame = env.render()
    img   = Image.fromarray(frame).resize((64,64), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32) / 255.0
    t     = torch.tensor(arr.transpose(2,0,1)).unsqueeze(0)
    with torch.no_grad():
        return encoder(t)

# Test: does latent distance correlate with real distance?
z_origin = encode_pos(7, 7, 0)  # center

print("Latent distance vs Manhattan distance from center (7,7):")
for x, y in [(7,8), (7,10), (7,14), (1,7), (14,14)]:
    z     = encode_pos(x, y, 0)
    lat_d = (1 - F.cosine_similarity(z_origin, z)).item()
    man_d = abs(x-7) + abs(y-7)
    print(f"  pos=({x},{y})  manhattan={man_d:2d}  latent={lat_d:.4f}")

env.close()