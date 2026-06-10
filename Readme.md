# JEPA World Model for MiniGrid Navigation

A self-supervised world model built on the **Joint-Embedding Predictive Architecture (JEPA)** paradigm, applied to navigation in MiniGrid environments. This project follows the PPUU pipeline (Henaff et al. 2019) and validates that learned world model dynamics can substitute for ground-truth environment interaction during controller training.

---

## Overview

The core question this project investigates:

> *Can a self-supervised world model learn environment dynamics well enough to train a navigation controller — without any reward signal or environment labels?*

**Short answer: yes, for single-room navigation.** The world model achieves 90% success on Empty environment navigation, matching a DAgger-trained controller (91%) that uses ground-truth environment interaction. Cross-room navigation in FourRooms remains an open problem, with a clear diagnosis of why and a principled direction for future work.

---

## Architecture

```
Observation (3×64×64 pixels)
        ↓
   CNN Encoder          → 256-dim latent vector z
        ↓
   [z | direction]      → 260-dim state
        ↓
   MLP Predictor        → next state (z_{t+1} | dir_{t+1})
        ↓
   MLP Controller       → action logits (left / right / forward)
```

### Components

**Encoder** (`src/encoder_v2.py`)
- 4-layer CNN: 3→32→64→128→256 channels, stride-2 convolutions
- Linear projection → 256-dim latent z with LayerNorm
- ActionEncoder: one-hot action → 64-dim embedding via MLP (prevents action embedding collapse)
- Target encoder: EMA copy of online encoder (momentum=0.99)

**Predictor** (`src/encoder_v2.py`)
- MLP: (z=256 + direction=4 + action_embed=64) → next state (260-dim)
- Predicts both next z and next direction jointly

**Controller** (`src/controller_bc.py`)
- MLP: (state_curr, state_goal, pos_curr, pos_goal) → action logits
- Trained via DAgger with BFS expert and replay buffer

---

## Training Pipeline

### Phase 1 — World Model on Empty Environment
```bash
python src/trainer.py
```
- Systematic data collection: every (x, y, direction) state × 3 actions × 20 resets
- VICReg loss (invariance + variance + covariance) prevents representation collapse
- Direction transition loss forces predictor to learn rotation semantics
- Action covariance loss keeps action embeddings distinct

### Phase 2 — World Model on FourRooms
```bash
python src/trainer_fourrooms.py
```
- Fine-tunes Phase 1 checkpoint on FourRooms observations
- Same VICReg + direction transition objective
- Learns wall/doorway visual features

### Phase 3 — Controller via DAgger (Empty)
```bash
python src/controller_dagger.py
```
- Expert: BFS shortest-path planner
- Replay buffer (100k capacity) prevents catastrophic forgetting
- Beta annealing: starts fully expert, gradually shifts to learned policy

### Phase 4 — World Model Rollout (Empty)
```bash
python src/controller_stage4_empty.py
```
- Controller trained purely on world model imagination
- Loss: MSE between predicted z and expert intermediate states
- Uses expert path as intermediate targets (fixes vanishing gradient problem)

### Phase 5 — FourRooms Controller
```bash
python src/controller_stage4_4_walls.py
```
- DAgger with curriculum learning on FourRooms
- Three stages: same-room → diff-room → any path
- Hierarchical variant: high-level picks subgoals, low-level executes

---

## Results

| Setting | Method | Success Rate |
|---|---|---|
| Empty 16×16 | DAgger (ground truth) | **91%** |
| Empty 16×16 | Stage 4 (world model rollout) | **90%** |
| FourRooms | Hierarchical (high+low level) | **32%** |
| FourRooms | Flat DAgger + curriculum | ~25–32% |

**Key validation:** Stage 4 world model rollout (90%) ≈ DAgger with ground truth (91%). This confirms the learned dynamics model is accurate enough to serve as a differentiable proxy for the real environment — the central claim of the PPUU approach.

---

## Key Technical Insights

### 1. World Model as Differentiable Simulator
Training a controller purely on world model imagination achieves equivalent performance to training with ground truth expert interaction. This validates the JEPA/PPUU approach: learned dynamics can replace real environment rollouts.

### 2. Expert Intermediate States Fix the Gradient Problem
Initial Stage 4 attempts using the final goal as target produced vanishing gradients (~0.000027) because the goal was 10+ steps away. Fix: use expert intermediate states as step-by-step targets. Gradient signal jumped to ~0.0002 and training stabilized immediately.

### 3. Replay Buffer is Critical
Without a replay buffer, the controller sees each state once and forgets it. A 100k-capacity buffer allows the same state to be revisited from multiple angles, preventing catastrophic forgetting when curriculum difficulty increases.

### 4. Curriculum Learning for Cross-Room Navigation
Random sampling of start/goal pairs causes the controller to over-specialize on within-room navigation. A three-stage curriculum (same-room → forced cross-room → any) significantly improves medium-distance success (27% → 40–60%).

### 5. Doorways are Distinctively Encoded
Probing the encoder's representations confirms that doorway cells produce z-vectors that are distinctly different from floor and wall cells. The encoder *sees* doorways — but doesn't understand their navigational significance as connectors between rooms.

---

## Root Cause Analysis: Why Cross-Room Navigation Fails

The central finding of this project's failure analysis:

```
Within-room z-distance:  6.77
Between-room z-distance: 7.81
Ratio:                   1.15x  (want >> 1.0)
```

The encoder organizes z-space by **visual/positional similarity**, not by **navigational connectivity**. Cells in different rooms look almost as similar to each other as cells within the same room. As a result:

- The controller cannot infer which room it is currently in
- The high-level controller cannot identify which doorway to route through
- Cross-room planning requires information that simply isn't encoded in z

### What Was Tried

**Attempt 1 — Triplet contrastive loss** (`src/finetune_room_aware.py`)
Same-room z-vectors pulled together, cross-room z-vectors pushed apart. Result: room ratio jumped to 1779x but within-room position information collapsed (d_pos → 0.04). The encoder learned room identity but lost position structure — trading one problem for another.

**Attempt 2 — Hierarchical triplet loss** (`src/finetune_room_aware_v2.py`)
Added a second loss level: close-position pairs pulled tighter than far-position pairs within the same room. Result: loss satisfied trivially within 200 steps (hinge loss collapsed to 0), no stable gradient signal.

**Attempt 3 — Geodesic distance regression** (`src/finetune_geodesic.py`)
Reshaped z-space so that ||z_i − z_j|| ∝ BFS walking distance between cells i and j. Training correlation reached **0.97** — z-distances tracked walking distances almost perfectly. However, latent-space doorway routing remained at chance level (20% with 4 doorways).

**Root cause of Attempt 3 failure** (`src/confirm_confound.py`):

```
corr(euclidean, geodesic) baseline:  0.949
corr(z_dist, geodesic):              0.677
corr(z_dist, euclidean):             0.648
```

In FourRooms, raw euclidean distance already predicts walking distance with 95% accuracy (most cell pairs are in open rooms with no wall between them). The geodesic loss was dominated by these easy pairs and never learned to represent wall detours specifically. Additionally, training data was collected across 30 random environment resets (randomized doorway positions), while geodesic targets were computed from one fixed layout — creating a confound that prevented the encoder from learning consistent wall structure.

---

## Future Work

The diagnosis points clearly to the next steps:

**Immediate fix (highest priority):**
Train geodesic fine-tuning on a **single fixed layout** with **wall-detour-weighted loss**:
```python
wall_penalty = geodesic_dist - euclidean_dist   # large only at wall crossings
weight = 1.0 + α * wall_penalty                 # upweight informative pairs
loss = weighted_MSE(z_dist, geodesic_dist, weight)
```
This forces the encoder to focus on the 5% of pairs where wall structure actually changes the answer.

**Medium term:**
- Train world model from scratch with geodesic objective as primary loss (rather than fine-tuning)
- Replace MLP controller with Transformer for memory across doorway crossings
- Topological map approach (SNN, Savinov et al. 2018): build explicit graph of visited z-vectors, plan with BFS on the graph

**Research direction:**
The ideal representation encodes **local reachability** — whether adjacent cells are connected or blocked — as a composable primitive. Global room structure would then emerge from chaining local facts, exactly as a human builds a mental map by bumping into walls and walking through gaps.

---

## Repository Structure

```
WorldModel/
├── src/
│   │
│   ├── # Core architecture
│   ├── encoder_v2.py                  Encoder (CNN→256-dim z) + Predictor + ActionEncoder
│   ├── controller_bc.py               Base MLP controller architecture
│   │
│   ├── # Data collection
│   ├── data_collector_v2.py           Systematic (x,y,dir,action) collector + ReplayBuffer
│   ├── data_collector_fourrooms.py    FourRooms variant with position tracking
│   │
│   ├── # World model training
│   ├── trainer.py                     Phase 1: VICReg JEPA on Empty env
│   ├── trainer_fourrooms.py           Phase 2: Fine-tune on FourRooms
│   │
│   ├── # Controller training
│   ├── controller_dagger.py           DAgger on Empty (91%)
│   ├── controller_stage4_empty.py     World model rollout on Empty (90%)
│   ├── controller_stage4_4_walls.py   DAgger + curriculum on FourRooms
│   ├── controller_highlevel.py        Hierarchical high-level controller
│   ├── controller_rl.py               RL-based controller (experimental)
│   ├── controller_joint.py            Joint training experiment
│   │
│   ├── # Representation experiments
│   ├── finetune_room_aware.py         Triplet contrastive loss (attempt 1)
│   ├── finetune_room_aware_v2.py      Hierarchical triplet loss (attempt 2)
│   ├── finetune_geodesic.py           Geodesic distance regression (attempt 3)
│   │
│   ├── # Evaluation
│   ├── evaluate_fourrooms.py          500-episode FourRooms evaluation
│   ├── evaluate_dagger.py             Empty env evaluation
│   │
│   ├── # Diagnostics
│   ├── test_room_encoding.py          Room separation metric (1.15x finding)
│   ├── test_doorway_encoding.py       Doorway distinctiveness probe
│   ├── diagnose_hierarchical.py       Hierarchical bottleneck analysis
│   ├── diagnose_latent_planning.py    Latent-space doorway routing test
│   ├── confirm_confound.py            Layout confound confirmation
│   │
│   └── # Visualization
│       ├── visualize_agent.py         Agent trajectory visualization
│       ├── visualize_navigation.py    Navigation path rendering
│       └── visualize_policy.py        Policy heatmap
│
├── checkpoints/                       Trained model weights
├── data/                              Replay buffers (not tracked in git)
└── README.md
```

---

## Setup

```bash
# Create environment
conda create -n worldmodel python=3.10
conda activate worldmodel

# Install dependencies
pip install torch torchvision
pip install gymnasium minigrid
pip install numpy pillow
```

**Hardware:** Trained on NVIDIA GPU (CUDA). CPU inference works but is slow for evaluation.

---

## Checkpoints

| File | Description |
|---|---|
| `jepa_phase1_final.pt` | World model trained on Empty env |
| `jepa_fourrooms_final.pt` | World model fine-tuned on FourRooms |
| `jepa_geodesic_final.pt` | Geodesic fine-tuned encoder (experimental) |
| `controller_dagger_dagger_final.pt` | DAgger controller, Empty (91%) |
| `controller_stage4_empty_stage4_empty_final.pt` | World model rollout controller (90%) |
| `controller_fourrooms_fourrooms_final.pt` | FourRooms flat controller (32%) |
| `highlevel_hierarchical_final.pt` | Hierarchical high-level controller |

---

## References

- **PPUU** — Henaff et al. (2019). *Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic.*
- **VICReg** — Bardes et al. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.*
- **DAgger** — Ross et al. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.*
- **JEPA** — LeCun (2022). *A Path Towards Autonomous Machine Intelligence.*
- **Truck Backer-Upper** — Nguyen & Widrow (1989). *The truck backer-upper: An example of self-learning in neural networks.*
- **SNN** — Savinov et al. (2018). *Semi-parametric Topological Memory for Navigation.*
- **HIRO** — Nachum et al. (2018). *Data-Efficient Hierarchical Reinforcement Learning.*

---

<!-- ## Interview Narrative

> "I implemented a self-supervised JEPA world model and trained navigation controllers using DAgger imitation learning. The key result: a controller trained purely on world model imagination (90%) matched one trained with ground-truth expert interaction (91%), validating that learned dynamics can serve as a differentiable simulator. For cross-room navigation in FourRooms, I identified that the encoder organized z-space by visual position rather than navigational connectivity — a 1.15x room separation ratio where we'd want much higher. I designed and ran three principled representation-fixing experiments (contrastive, hierarchical triplet, geodesic regression), diagnosed a layout-confound failure in the geodesic approach via controlled ablation, and identified the weighted geodesic loss on a fixed layout as the clean next step."

--- -->

*Built as part of NYU Courant MS CS research. Environment: MiniGrid (Chevalier-Boisvert et al.). Architecture inspired by PPUU and I-JEPA.*