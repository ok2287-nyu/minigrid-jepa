import torch
import minigrid
import gymnasium as gym

env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
obs, _ = env.reset()
frame = env.render()
print("Environment works. Frame shape:", frame.shape)
print("CUDA available:", torch.cuda.is_available())