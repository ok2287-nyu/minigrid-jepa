import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
from PIL import Image
import pygame
import os
from pathlib import Path

from encoder_v2 import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────
WINDOW_W     = 900
WINDOW_H     = 400
CELL_SIZE    = 380
FPS          = 30
IMG_SIZE     = 64

# Colors
BLACK  = (0,   0,   0  )
WHITE  = (255, 255, 255)
GRAY   = (40,  40,  40 )
GREEN  = (0,   200, 100)
YELLOW = (255, 220, 0  )
RED    = (220, 50,  50 )
BLUE   = (50,  150, 255)

ACTION_NAMES = {
    pygame.K_LEFT:  (0, "Turn Left"),
    pygame.K_RIGHT: (1, "Turn Right"),
    pygame.K_UP:    (2, "Move Forward"),
}


def load_model(checkpoint_path, device):
    encoder   = Encoder(latent_dim=256).to(device)
    predictor = Predictor(latent_dim=256, n_actions=3).to(device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    encoder.load_state_dict(ckpt["online_encoder"])
    predictor.load_state_dict(ckpt["predictor"])

    encoder.eval()
    predictor.eval()

    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    print(f"Model loaded from {checkpoint_path}")
    return encoder, predictor


def preprocess_frame(frame, img_size=IMG_SIZE):
    """Raw frame → normalized tensor (1, 3, 64, 64)."""
    img = Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0)


def tensor_to_surface(tensor, size):
    """
    Convert (1, 3, H, W) float tensor → pygame surface.
    Used to display the world model's imagined next frame.
    """
    arr = tensor.squeeze(0).cpu().numpy()      # (3, H, W)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)               # (H, W, 3)
    img = Image.fromarray(arr).resize(
        (size, size), Image.NEAREST
    )
    arr_resized = np.array(img)
    return pygame.surfarray.make_surface(
        arr_resized.transpose(1, 0, 2)
    )


def frame_to_surface(frame, size):
    """Raw env frame → pygame surface."""
    img = Image.fromarray(frame).resize((size, size), Image.NEAREST)
    arr = np.array(img)
    return pygame.surfarray.make_surface(arr.transpose(1, 0, 2))


def draw_text(surface, text, x, y, color=WHITE, size=18):
    font   = pygame.font.SysFont("monospace", size, bold=False)
    render = font.render(text, True, color)
    surface.blit(render, (x, y))


def draw_panel(surface, title, img_surface, x, y, w, h, color):
    """Draw a labeled panel with an image inside."""
    # Border
    pygame.draw.rect(surface, color, (x-2, y-2, w+4, h+4), 2)

    # Title bar
    pygame.draw.rect(surface, color, (x-2, y-22, w+4, 22))
    font   = pygame.font.SysFont("monospace", 14, bold=True)
    render = font.render(title, True, BLACK)
    surface.blit(render, (x + 4, y - 19))

    # Image
    surface.blit(img_surface, (x, y))


def run_interactive(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    encoder, predictor = load_model(checkpoint_path, device)

    # Setup environment
    env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
    obs, _ = env.reset()

    # Setup pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "World Model Interactive Demo — Arrow Keys to Move"
    )
    clock = pygame.time.Clock()

    # State
    current_frame    = env.render()
    last_action      = None
    last_action_name = "None"
    predicted_frame  = None
    step_count       = 0
    prediction_error = None

    # Encode current state
    with torch.no_grad():
        obs_tensor = preprocess_frame(current_frame).to(device)
        z_current  = encoder(obs_tensor)

    running = True
    while running:
        # ── Event handling ────────────────────────────────────────────
        action_taken = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # Reset environment
                    obs, _ = env.reset()
                    current_frame    = env.render()
                    predicted_frame  = None
                    last_action      = None
                    last_action_name = "None"
                    prediction_error = None
                    step_count       = 0
                    with torch.no_grad():
                        obs_tensor = preprocess_frame(
                            current_frame
                        ).to(device)
                        z_current  = encoder(obs_tensor)
                    print("Environment reset.")

                elif event.key in ACTION_NAMES:
                    action_taken, last_action_name = ACTION_NAMES[event.key]

        # ── Take action if key was pressed ────────────────────────────
        if action_taken is not None:
            last_action = action_taken

            # ── World model prediction BEFORE taking action ───────────
            with torch.no_grad():
                action_t  = torch.tensor(
                    [action_taken], device=device
                )
                z_pred    = predictor(z_current, action_t)
                z_pred    = F.normalize(z_pred, dim=-1)

            # ── Take real action in environment ───────────────────────
            prev_frame    = current_frame.copy()
            _, _, term, trunc, _ = env.step(action_taken)
            current_frame = env.render()
            step_count   += 1

            if term or trunc:
                obs, _ = env.reset()
                current_frame = env.render()
                print("Episode ended, resetting...")

            # ── Encode real next state ────────────────────────────────
            with torch.no_grad():
                obs_tensor = preprocess_frame(current_frame).to(device)
                z_actual   = encoder(obs_tensor)
                z_current  = z_actual   # update current state

            # ── Compute prediction error ──────────────────────────────
            with torch.no_grad():
                sim = F.cosine_similarity(
                    F.normalize(z_pred, dim=-1),
                    F.normalize(z_actual, dim=-1),
                    dim=-1
                ).item()
                prediction_error = 1 - sim   # distance (lower = better)

            # ── Decode predicted z back to pixel space ────────────────
            # We can't directly decode z (no decoder)
            # Instead show the actual next frame for comparison
            # and display latent distance as the accuracy metric
            predicted_frame = current_frame   # placeholder

        # ── Draw ──────────────────────────────────────────────────────
        screen.fill(GRAY)

        # Current frame panel
        cur_surf = frame_to_surface(current_frame, CELL_SIZE)
        draw_panel(
            screen, "REAL ENVIRONMENT",
            cur_surf, 20, 40, CELL_SIZE, CELL_SIZE, GREEN
        )

        # Latent space info panel (right side)
        info_x = CELL_SIZE + 40
        info_y = 40

        # Title
        font_big = pygame.font.SysFont("monospace", 20, bold=True)
        title    = font_big.render("World Model Demo", True, WHITE)
        screen.blit(title, (info_x, info_y))

        # Controls
        draw_text(screen, "Controls:", info_x, info_y + 35,
                  YELLOW, 16)
        draw_text(screen, "← Left Arrow  : Turn Left",
                  info_x, info_y + 58,  WHITE, 14)
        draw_text(screen, "→ Right Arrow : Turn Right",
                  info_x, info_y + 76,  WHITE, 14)
        draw_text(screen, "↑ Up Arrow    : Move Forward",
                  info_x, info_y + 94,  WHITE, 14)
        draw_text(screen, "R             : Reset",
                  info_x, info_y + 112, WHITE, 14)
        draw_text(screen, "ESC           : Quit",
                  info_x, info_y + 130, WHITE, 14)

        # Stats
        draw_text(screen, "─" * 32,
                  info_x, info_y + 158, GRAY, 13)
        draw_text(screen, f"Steps:        {step_count}",
                  info_x, info_y + 175, WHITE, 15)
        draw_text(screen, f"Last action:  {last_action_name}",
                  info_x, info_y + 198, WHITE, 15)

        pos = env.unwrapped.agent_pos
        d   = ["East", "South", "West", "North"]
        direction = d[env.unwrapped.agent_dir]
        draw_text(screen,
                  f"Position:     ({int(pos[0])}, {int(pos[1])})",
                  info_x, info_y + 221, WHITE, 15)
        draw_text(screen, f"Facing:       {direction}",
                  info_x, info_y + 244, WHITE, 15)

        # Prediction error
        if prediction_error is not None:
            err_color = (
                GREEN if prediction_error < 0.3 else
                YELLOW if prediction_error < 0.6 else
                RED
            )
            draw_text(screen, "─" * 32,
                      info_x, info_y + 272, GRAY, 13)
            draw_text(screen, "World Model Accuracy:",
                      info_x, info_y + 289, YELLOW, 15)
            draw_text(screen,
                      f"Latent distance: {prediction_error:.4f}",
                      info_x, info_y + 312, err_color, 15)

            # Visual accuracy bar
            bar_w   = 200
            bar_h   = 12
            bar_x   = info_x
            bar_y   = info_y + 335
            fill    = int(bar_w * (1 - min(prediction_error, 1.0)))
            pygame.draw.rect(
                screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h)
            )
            pygame.draw.rect(
                screen, err_color, (bar_x, bar_y, fill, bar_h)
            )
            draw_text(screen,
                      "Bad                    Good",
                      info_x, bar_y + 15, GRAY, 11)

            # Interpretation
            if prediction_error < 0.2:
                interp = "Excellent prediction!"
            elif prediction_error < 0.4:
                interp = "Good prediction"
            elif prediction_error < 0.6:
                interp = "Moderate prediction"
            else:
                interp = "Poor prediction"
            draw_text(screen, interp,
                      info_x, bar_y + 30, err_color, 14)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    env.close()
    print("Demo closed.")


if __name__ == "__main__":
    run_interactive("checkpoints/jepa_5000.pt")