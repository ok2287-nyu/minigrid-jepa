"""
JEPA Navigation Visualizer
==========================
Watch the DAgger-trained controller navigate to random goals in real time.

Usage:
    python visualize_navigation.py

Controls:
    SPACE   — new random goal (anywhere on grid)
    S       — new random goal (short distance 1-4)
    L       — new random goal (long distance 8-26)
    R       — reset to same goal (retry)
    Q       — quit
    +/-     — speed up / slow down
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import minigrid
import pygame

from encoder_v2 import Encoder
from controller_bc import Controller


# ── Config ────────────────────────────────────────────────────────────
JEPA_CKPT    = "checkpoints/jepa_phase1_final.pt"
DAGGER_CKPT  = "checkpoints/controller_dagger_dagger_final.pt"
ENV_ID       = "MiniGrid-Empty-16x16-v0"
LATENT_DIM   = 256
WINDOW_W     = 900
WINDOW_H     = 620
GRID_PIXEL   = 36          # px per cell in our custom grid
GRID_MARGIN  = 20
STEP_DELAY   = 0.18        # seconds between steps


# ── Colors ────────────────────────────────────────────────────────────
BG          = (10,  12,  20)
WALL        = (35,  38,  52)
CELL        = (22,  26,  40)
AGENT_COL   = (80, 200, 120)
GOAL_COL    = (255, 180,  40)
TRAIL_COL   = (50,  90, 140)
TEXT_COL    = (200, 210, 230)
DIM_COL     = (80,  90, 110)
SUCCESS_COL = (80, 200, 120)
FAIL_COL    = (220,  70,  70)
ACCENT      = (100, 160, 255)


# ── Load models ───────────────────────────────────────────────────────
def load_models(device):
    encoder = Encoder(LATENT_DIM).to(device)
    ckpt = torch.load(JEPA_CKPT, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["online_encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    controller = Controller(
        state_dim=LATENT_DIM + 4, hidden_dim=512, n_actions=3, pos_dim=2
    ).to(device)
    ckpt2 = torch.load(DAGGER_CKPT, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt2["controller"])
    controller.eval()

    return encoder, controller


# ── Encode helpers ────────────────────────────────────────────────────
def encode_frame(encoder, frame, device):
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)

def build_state(z, d, device):
    dir_oh = F.one_hot(torch.tensor([d], dtype=torch.long).to(device), num_classes=4).float()
    return torch.cat([z, dir_oh], dim=-1)

def build_pos(x, y, max_x, max_y, device):
    return torch.tensor([[x / max_x, y / max_y]], dtype=torch.float32).to(device)

def manual_step(x, y, d, action, max_x, max_y):
    if action == 0: return x, y, (d-1) % 4
    elif action == 1: return x, y, (d+1) % 4
    else:
        dx, dy = [1,0,-1,0][d], [0,1,0,-1][d]
        nx, ny = x+dx, y+dy
        if 1 <= nx <= max_x and 1 <= ny <= max_y:
            return nx, ny, d
        return x, y, d


# ── Drawing ───────────────────────────────────────────────────────────
def draw_grid(surface, env, agent_pos, goal_pos, trail, max_x, max_y, ox, oy):
    """Draw the grid, trail, goal, and agent."""
    gs = GRID_PIXEL

    # Background cells
    for gx in range(1, max_x + 1):
        for gy in range(1, max_y + 1):
            rx = ox + (gx - 1) * gs
            ry = oy + (gy - 1) * gs
            pygame.draw.rect(surface, CELL, (rx+1, ry+1, gs-2, gs-2), border_radius=3)

    # Trail
    for (tx, ty) in trail:
        rx = ox + (tx - 1) * gs
        ry = oy + (ty - 1) * gs
        alpha_surf = pygame.Surface((gs-4, gs-4), pygame.SRCALPHA)
        alpha_surf.fill((*TRAIL_COL, 120))
        surface.blit(alpha_surf, (rx+2, ry+2))
        pygame.draw.rect(surface, TRAIL_COL, (rx+2, ry+2, gs-4, gs-4), 1, border_radius=2)

    # Goal
    gx_p = ox + (goal_pos[0] - 1) * gs
    gy_p = oy + (goal_pos[1] - 1) * gs
    pygame.draw.rect(surface, GOAL_COL, (gx_p+2, gy_p+2, gs-4, gs-4), border_radius=4)
    # Star on goal
    cx, cy = gx_p + gs//2, gy_p + gs//2
    r = gs//4
    for angle in range(0, 360, 72):
        import math
        a = math.radians(angle - 90)
        b = math.radians(angle - 90 + 36)
        x1 = cx + r * math.cos(a)
        y1 = cy + r * math.sin(a)
        x2 = cx + (r//2) * math.cos(b)
        y2 = cy + (r//2) * math.sin(b)
        pygame.draw.line(surface, BG, (cx, cy), (int(x1), int(y1)), 2)

    # Agent — triangle pointing in direction
    ax = ox + (agent_pos[0] - 1) * gs
    ay = oy + (agent_pos[1] - 1) * gs
    cx, cy = ax + gs//2, ay + gs//2
    d = agent_pos[2]
    r = gs//2 - 4
    import math
    angle_map = {0: 0, 1: 90, 2: 180, 3: 270}  # E S W N
    base_angle = math.radians(angle_map[d] - 90)
    tip   = (cx + r * math.cos(base_angle),       cy + r * math.sin(base_angle))
    left  = (cx + r * 0.6 * math.cos(base_angle + 2.4),
             cy + r * 0.6 * math.sin(base_angle + 2.4))
    right = (cx + r * 0.6 * math.cos(base_angle - 2.4),
             cy + r * 0.6 * math.sin(base_angle - 2.4))
    pygame.draw.polygon(surface, AGENT_COL, [tip, left, right])
    pygame.draw.polygon(surface, (255,255,255), [tip, left, right], 1)

    # Grid border
    grid_w = max_x * gs
    grid_h = max_y * gs
    pygame.draw.rect(surface, WALL, (ox-2, oy-2, grid_w+4, grid_h+4), 2, border_radius=4)


def draw_panel(surface, font_big, font_med, font_sm, state):
    """Draw info panel on the right."""
    px = GRID_MARGIN * 2 + 14 * GRID_PIXEL + 10
    py = GRID_MARGIN

    # Title
    title = font_big.render("JEPA Navigator", True, ACCENT)
    surface.blit(title, (px, py))
    py += 44

    # Status
    status_text = state.get("status", "navigating...")
    status_col = SUCCESS_COL if "success" in status_text else (FAIL_COL if "fail" in status_text else TEXT_COL)
    surf = font_med.render(status_text, True, status_col)
    surface.blit(surf, (px, py))
    py += 34

    # Divider
    pygame.draw.line(surface, WALL, (px, py), (px + 220, py), 1)
    py += 14

    def label_val(label, val, col=TEXT_COL):
        nonlocal py
        ls = font_sm.render(label, True, DIM_COL)
        vs = font_sm.render(str(val), True, col)
        surface.blit(ls, (px, py))
        surface.blit(vs, (px + 130, py))
        py += 22

    label_val("Start", f"({state['x_start']}, {state['y_start']})")
    label_val("Goal", f"({state['x_goal']}, {state['y_goal']})", GOAL_COL)
    label_val("Current", f"({state['x']}, {state['y']})")
    label_val("Manhattan dist", state['manhattan'])
    label_val("Steps taken", state['steps'])
    label_val("Facing", ['East','South','West','North'][state['d']])

    py += 10
    pygame.draw.line(surface, WALL, (px, py), (px + 220, py), 1)
    py += 14

    # Action probs
    ls = font_sm.render("Action probs", True, DIM_COL)
    surface.blit(ls, (px, py))
    py += 22

    probs = state.get("probs", [0.33, 0.33, 0.33])
    action_names = ["Left", "Right", "Fwd"]
    last_action  = state.get("last_action", -1)
    bar_w = 200
    for i, (name, prob) in enumerate(zip(action_names, probs)):
        col = ACCENT if i == last_action else DIM_COL
        ns  = font_sm.render(f"{name}", True, col)
        surface.blit(ns, (px, py))
        filled = int(prob * bar_w)
        pygame.draw.rect(surface, WALL,    (px + 40, py+3, bar_w, 12), border_radius=3)
        pygame.draw.rect(surface, col,     (px + 40, py+3, filled, 12), border_radius=3)
        ps = font_sm.render(f"{prob:.2f}", True, col)
        surface.blit(ps, (px + 40 + bar_w + 6, py+2))
        py += 20

    py += 10
    pygame.draw.line(surface, WALL, (px, py), (px + 220, py), 1)
    py += 14

    # Stats
    label_val("Total episodes", state['total_eps'])
    label_val("Successes", state['successes'], SUCCESS_COL)
    label_val("Failures", state['failures'],   FAIL_COL)
    if state['total_eps'] > 0:
        sr = state['successes'] / state['total_eps'] * 100
        label_val("Success rate", f"{sr:.1f}%", SUCCESS_COL if sr > 80 else TEXT_COL)

    py += 14
    pygame.draw.line(surface, WALL, (px, py), (px + 220, py), 1)
    py += 14

    # Controls
    controls = [
        ("SPACE", "new random goal"),
        ("S",     "short goal (1-4)"),
        ("L",     "long goal (8-26)"),
        ("R",     "retry same goal"),
        ("+/-",   "speed up/down"),
        ("Q",     "quit"),
    ]
    for key, desc in controls:
        ks = font_sm.render(key, True, ACCENT)
        ds = font_sm.render(desc, True, DIM_COL)
        surface.blit(ks, (px, py))
        surface.blit(ds, (px + 40, py))
        py += 19

    # Speed indicator
    py += 6
    spd = font_sm.render(f"Speed: {state.get('speed_label','normal')}", True, DIM_COL)
    surface.blit(spd, (px, py))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    encoder, controller = load_models(device)
    print("Models loaded.")

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.reset()
    max_x = env.unwrapped.width  - 2   # 14
    max_y = env.unwrapped.height - 2   # 14

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("JEPA Navigation Visualizer")
    clock = pygame.time.Clock()

    try:
        font_big = pygame.font.SysFont("consolas", 22, bold=True)
        font_med = pygame.font.SysFont("consolas", 16, bold=True)
        font_sm  = pygame.font.SysFont("consolas", 13)
    except:
        font_big = pygame.font.SysFont(None, 24)
        font_med = pygame.font.SysFont(None, 18)
        font_sm  = pygame.font.SysFont(None, 14)

    ox = GRID_MARGIN
    oy = GRID_MARGIN

    step_delay = STEP_DELAY
    speed_labels = {0.30: "slow", 0.18: "normal", 0.08: "fast", 0.02: "turbo"}
    speed_levels  = [0.30, 0.18, 0.08, 0.02]
    speed_idx     = 1

    total_eps  = 0
    successes  = 0
    failures   = 0

    def sample_goal(x_start, y_start, min_d=1, max_d=26):
        while True:
            xg = np.random.randint(1, max_x+1)
            yg = np.random.randint(1, max_y+1)
            dg = np.random.randint(0, 4)
            md = abs(x_start-xg) + abs(y_start-yg)
            if min_d <= md <= max_d:
                return xg, yg, dg

    def new_episode(min_d=1, max_d=26):
        xs = np.random.randint(1, max_x+1)
        ys = np.random.randint(1, max_y+1)
        ds = np.random.randint(0, 4)
        xg, yg, dg = sample_goal(xs, ys, min_d, max_d)
        return xs, ys, ds, xg, yg, dg

    # Initial episode
    x_start, y_start, d_start, x_goal, y_goal, d_goal = new_episode()

    def encode_goal(xg, yg, dg):
        env.unwrapped.agent_pos = np.array([xg, yg])
        env.unwrapped.agent_dir = dg
        frame = env.render()
        z = encode_frame(encoder, frame, device)
        sg = build_state(z, dg, device)
        pg = build_pos(xg, yg, max_x, max_y, device)
        return sg, pg

    sg, pg = encode_goal(x_goal, y_goal, d_goal)

    x, y, d = x_start, y_start, d_start
    trail    = []
    steps    = 0
    max_steps = 120
    episode_done = False
    status   = "navigating..."
    probs    = [0.33, 0.33, 0.33]
    last_action = -1
    last_step_time = time.time()
    retry_params = (x_start, y_start, d_start, x_goal, y_goal, d_goal)

    running = True
    while running:
        now = time.time()

        # ── Events ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    x_start, y_start, d_start, x_goal, y_goal, d_goal = new_episode()
                    sg, pg = encode_goal(x_goal, y_goal, d_goal)
                    x, y, d = x_start, y_start, d_start
                    trail = []; steps = 0; episode_done = False; status = "navigating..."
                    retry_params = (x_start, y_start, d_start, x_goal, y_goal, d_goal)

                elif event.key == pygame.K_s:
                    x_start, y_start, d_start, x_goal, y_goal, d_goal = new_episode(1, 4)
                    sg, pg = encode_goal(x_goal, y_goal, d_goal)
                    x, y, d = x_start, y_start, d_start
                    trail = []; steps = 0; episode_done = False; status = "navigating..."
                    retry_params = (x_start, y_start, d_start, x_goal, y_goal, d_goal)

                elif event.key == pygame.K_l:
                    x_start, y_start, d_start, x_goal, y_goal, d_goal = new_episode(8, 26)
                    sg, pg = encode_goal(x_goal, y_goal, d_goal)
                    x, y, d = x_start, y_start, d_start
                    trail = []; steps = 0; episode_done = False; status = "navigating..."
                    retry_params = (x_start, y_start, d_start, x_goal, y_goal, d_goal)

                elif event.key == pygame.K_r:
                    x_start, y_start, d_start, x_goal, y_goal, d_goal = retry_params
                    sg, pg = encode_goal(x_goal, y_goal, d_goal)
                    x, y, d = x_start, y_start, d_start
                    trail = []; steps = 0; episode_done = False; status = "navigating..."

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed_idx = min(speed_idx + 1, len(speed_levels)-1)
                    step_delay = speed_levels[speed_idx]

                elif event.key == pygame.K_MINUS:
                    speed_idx = max(speed_idx - 1, 0)
                    step_delay = speed_levels[speed_idx]

        # ── Step controller ───────────────────────────────────────────
        if not episode_done and (now - last_step_time) >= step_delay:
            last_step_time = now
            trail.append((x, y))

            env.unwrapped.agent_pos = np.array([x, y])
            env.unwrapped.agent_dir = d
            frame = env.render()
            z = encode_frame(encoder, frame, device)
            sc = build_state(z, d, device)
            pc = build_pos(x, y, max_x, max_y, device)

            with torch.no_grad():
                logits = controller(sc, sg, pc, pg)
                prob_t = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

            probs = prob_t.tolist()
            last_action = int(np.argmax(probs))
            x, y, d = manual_step(x, y, d, last_action, max_x, max_y)
            steps += 1

            if (x, y) == (x_goal, y_goal):
                episode_done = True
                status = "✓ success!"
                successes += 1
                total_eps += 1
            elif steps >= max_steps:
                episode_done = True
                status = "✗ failed (timeout)"
                failures += 1
                total_eps += 1

        # ── Draw ──────────────────────────────────────────────────────
        screen.fill(BG)

        draw_grid(
            screen, env,
            (x, y, d),
            (x_goal, y_goal),
            trail,
            max_x, max_y, ox, oy
        )

        state_info = {
            "status":      status,
            "x_start":     x_start, "y_start": y_start,
            "x_goal":      x_goal,  "y_goal":  y_goal,
            "x":           x,       "y":       y,
            "d":           d,
            "manhattan":   abs(x_start-x_goal) + abs(y_start-y_goal),
            "steps":       steps,
            "probs":       probs,
            "last_action": last_action,
            "total_eps":   total_eps,
            "successes":   successes,
            "failures":    failures,
            "speed_label": speed_labels.get(step_delay, "custom"),
        }
        draw_panel(screen, font_big, font_med, font_sm, state_info)

        # Auto-advance after success/fail
        if episode_done and (now - last_step_time) >= 1.5:
            x_start, y_start, d_start, x_goal, y_goal, d_goal = new_episode()
            sg, pg = encode_goal(x_goal, y_goal, d_goal)
            x, y, d = x_start, y_start, d_start
            trail = []; steps = 0; episode_done = False; status = "navigating..."
            retry_params = (x_start, y_start, d_start, x_goal, y_goal, d_goal)
            last_step_time = time.time()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()
    print(f"\nFinal: {successes}/{total_eps} episodes successful")


if __name__ == "__main__":
    main()