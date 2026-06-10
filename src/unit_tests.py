import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import minigrid
from pathlib import Path
from PIL import Image
import os

from encoder_v2 import Encoder, Predictor

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


class WorldModelTester:
    """
    Unit tests for the trained JEPA world model.

    Tests whether the predictor correctly learned
    the transition dynamics of the environment:

    1. Transition accuracy:
       Given (obs_t, action), does predicted z_t+1
       match actual z_t+1 better than random?

    2. Action consistency:
       Does forward always produce a different z than turning?
       Does wall-hitting produce z ≈ z_current?

    3. Directional consistency:
       Turn left 4 times → should return to original z
       (360 degree rotation = identity)

    4. Position prediction accuracy:
       Use linear probe on predicted z — does it predict
       the correct next position?
    """
    def __init__(self, checkpoint_path, latent_dim=256, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder   = Encoder(latent_dim).to(self.device)
        self.predictor = Predictor(latent_dim, n_actions=3).to(self.device)

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        self.encoder.load_state_dict(ckpt["online_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

        self.encoder.eval()
        self.predictor.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

        self.env = gym.make(
            "MiniGrid-FourRooms-v0",
            render_mode="rgb_array"
        )
        print(f"WorldModelTester loaded. Device: {self.device}")

    def encode_frame(self, frame):
        """Raw frame → latent vector."""
        img = Image.fromarray(frame).resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        t   = torch.tensor(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(t)

    def predict_next(self, z, action):
        """Predict next latent state given current z and action."""
        action_t = torch.tensor([action], device=self.device)
        with torch.no_grad():
            z_pred = self.predictor(z, action_t)
            return F.normalize(z_pred, dim=-1)

    # ──────────────────────────────────────────────────────────────────
    # TEST 1 — Transition Accuracy
    # ──────────────────────────────────────────────────────────────────
    def test_transition_accuracy(self, n_samples=500):
        """
        Core test: for N real transitions (obs_t, action, obs_t+1),
        does predicted z_t+1 match actual z_t+1?

        Metric: cosine similarity between predicted and actual z_t+1
        Baseline: cosine similarity between random z pairs (should be ~0)
        Good model: similarity >> baseline
        """
        print("\n" + "="*50)
        print("TEST 1: Transition Accuracy")
        print("="*50)
        print(f"Collecting {n_samples} real transitions...")

        similarities_predicted = []
        similarities_random    = []
        similarities_by_action = {0: [], 1: [], 2: []}

        obs, _ = self.env.reset()

        for i in range(n_samples):
            # Get current frame and encode
            frame_t = self.env.render()
            z_t     = self.encode_frame(frame_t)

            # Take a real action
            action = np.random.randint(0, 3)
            self.env.step(action)

            # Get actual next frame and encode
            frame_t1 = self.env.render()
            z_t1_actual = self.encode_frame(frame_t1)

            # Predict next frame
            z_t1_predicted = self.predict_next(z_t, action)

            # Measure similarity: predicted vs actual
            sim = F.cosine_similarity(
                z_t1_predicted, z_t1_actual, dim=-1
            ).item()
            similarities_predicted.append(sim)
            similarities_by_action[action].append(sim)

            # Baseline: random z vs actual (should be ~0)
            z_random = F.normalize(
                torch.randn_like(z_t1_actual), dim=-1
            )
            sim_random = F.cosine_similarity(
                z_random, z_t1_actual, dim=-1
            ).item()
            similarities_random.append(sim_random)

            # Reset occasionally
            if i % 100 == 99:
                self.env.reset()

        avg_predicted = np.mean(similarities_predicted)
        avg_random    = np.mean(similarities_random)
        improvement   = avg_predicted - avg_random

        print(f"\nResults:")
        print(f"  Predicted vs Actual similarity: {avg_predicted:.4f}")
        print(f"  Random baseline similarity:     {avg_random:.4f}")
        print(f"  Improvement over random:        {improvement:.4f}")
        print(f"\nBy action:")
        action_names = ["Turn Left", "Turn Right", "Move Forward"]
        for a, name in enumerate(action_names):
            sims = similarities_by_action[a]
            if sims:
                print(f"  {name}: {np.mean(sims):.4f}")

        passed = avg_predicted > avg_random + 0.1
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"World model predicts better than random "
              f"by {improvement:.4f}")

        return dict(
            avg_predicted         = avg_predicted,
            avg_random            = avg_random,
            improvement           = improvement,
            similarities_predicted= similarities_predicted,
            similarities_by_action= similarities_by_action,
            passed                = passed
        )

    # ──────────────────────────────────────────────────────────────────
    # TEST 2 — Wall Detection
    # ──────────────────────────────────────────────────────────────────
    def test_wall_detection(self, n_samples=200):
        """
        Does the world model correctly predict that
        hitting a wall produces no state change?

        z_t+1_predicted ≈ z_t when agent faces wall + moves forward
        z_t+1_predicted ≠ z_t when agent moves into open space
        """
        print("\n" + "="*50)
        print("TEST 2: Wall Detection")
        print("="*50)

        wall_changes    = []   # state change when hitting wall
        open_changes    = []   # state change when moving freely

        self.env.reset()

        for _ in range(n_samples):
            frame_t  = self.env.render()
            z_t      = self.encode_frame(frame_t)
            pos_before = tuple(self.env.unwrapped.agent_pos)

            # Try forward action
            self.env.step(2)
            pos_after  = tuple(self.env.unwrapped.agent_pos)
            frame_t1   = self.env.render()
            z_t1_actual = self.encode_frame(frame_t1)

            # Predict what world model thought would happen
            z_t1_pred = self.predict_next(z_t, action=2)

            # How much did the predicted state change from current?
            predicted_change = (
                1 - F.cosine_similarity(z_t1_pred, z_t, dim=-1).item()
            )

            hit_wall = (pos_before == pos_after)

            if hit_wall:
                wall_changes.append(predicted_change)
            else:
                open_changes.append(predicted_change)

            if np.random.random() < 0.1:
                self.env.reset()

        avg_wall_change = np.mean(wall_changes) if wall_changes else 0
        avg_open_change = np.mean(open_changes) if open_changes else 0

        print(f"\nResults:")
        print(f"  Wall hit samples:  {len(wall_changes)}")
        print(f"  Open move samples: {len(open_changes)}")
        print(f"\n  Predicted state change when hitting wall:  "
              f"{avg_wall_change:.4f}")
        print(f"  Predicted state change when moving freely: "
              f"{avg_open_change:.4f}")

        passed = avg_open_change > avg_wall_change
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"World model predicts less change "
              f"for wall hits than open moves")

        return dict(
            avg_wall_change = avg_wall_change,
            avg_open_change = avg_open_change,
            wall_changes    = wall_changes,
            open_changes    = open_changes,
            passed          = passed
        )

    # ──────────────────────────────────────────────────────────────────
    # TEST 3 — Rotation Consistency
    # ──────────────────────────────────────────────────────────────────
    def test_rotation_consistency(self, n_trials=50):
        """
        Turn left 4 times = 360 degrees = back to original orientation.

        The predicted z after 4 left turns should be similar
        to the original z. This tests whether the world model
        learned consistent rotational dynamics.

        action=0 is turn left
        4 x turn left = full rotation
        """
        print("\n" + "="*50)
        print("TEST 3: Rotation Consistency (4x left = 360°)")
        print("="*50)

        similarities = []
        self.env.reset()

        for trial in range(n_trials):
            # Get current observation
            frame = self.env.render()
            z_original = self.encode_frame(frame)

            # Imagine 4 left turns in latent space
            z_current = z_original
            for _ in range(4):
                z_current = self.predict_next(z_current, action=0)

            # How similar is z after 4 turns to original z?
            sim = F.cosine_similarity(
                z_current, z_original, dim=-1
            ).item()
            similarities.append(sim)

            # Also do 4 actual turns to compare
            for _ in range(4):
                self.env.step(0)

            if trial % 10 == 9:
                self.env.reset()

        avg_sim = np.mean(similarities)
        print(f"\nResults ({n_trials} trials):")
        print(f"  Avg similarity after 4 left turns: {avg_sim:.4f}")
        print(f"  (1.0 = perfect, 0.0 = random)")

        passed = avg_sim > 0.5
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"4 left turns returns close to original state "
              f"(similarity={avg_sim:.4f})")

        return dict(
            similarities = similarities,
            avg_sim      = avg_sim,
            passed       = passed
        )

    # ──────────────────────────────────────────────────────────────────
    # TEST 4 — Action Distinguishability
    # ──────────────────────────────────────────────────────────────────
    def test_action_distinguishability(self, n_samples=300):
        """
        Given the same starting state, do the 3 actions
        produce meaningfully different predicted next states?

        If the world model learned action semantics correctly:
        - predict(z, left)    ≠ predict(z, right)
        - predict(z, left)    ≠ predict(z, forward)
        - predict(z, right)   ≠ predict(z, forward)

        Measured by cosine distance between predictions.
        Low distance = actions produce similar predictions (bad)
        High distance = actions are distinguishable (good)
        """
        print("\n" + "="*50)
        print("TEST 4: Action Distinguishability")
        print("="*50)

        left_vs_right   = []
        left_vs_forward = []
        right_vs_forward= []

        self.env.reset()

        for _ in range(n_samples):
            frame = self.env.render()
            z     = self.encode_frame(frame)

            z_left    = self.predict_next(z, action=0)
            z_right   = self.predict_next(z, action=1)
            z_forward = self.predict_next(z, action=2)

            # Distance between predictions for different actions
            # High distance = world model knows they're different
            d_lr = 1 - F.cosine_similarity(
                z_left, z_right, dim=-1
            ).item()
            d_lf = 1 - F.cosine_similarity(
                z_left, z_forward, dim=-1
            ).item()
            d_rf = 1 - F.cosine_similarity(
                z_right, z_forward, dim=-1
            ).item()

            left_vs_right.append(d_lr)
            left_vs_forward.append(d_lf)
            right_vs_forward.append(d_rf)

            # Take random action to vary the state
            self.env.step(np.random.randint(0, 3))
            if np.random.random() < 0.05:
                self.env.reset()

        avg_lr = np.mean(left_vs_right)
        avg_lf = np.mean(left_vs_forward)
        avg_rf = np.mean(right_vs_forward)

        print(f"\nResults (cosine distance between action predictions):")
        print(f"  Left vs Right:       {avg_lr:.4f}")
        print(f"  Left vs Forward:     {avg_lf:.4f}")
        print(f"  Right vs Forward:    {avg_rf:.4f}")
        print(f"  (higher = more distinguishable)")

        passed = min(avg_lr, avg_lf, avg_rf) > 0.01
        print(f"\n{'PASSED' if passed else 'FAILED'}: "
              f"All action pairs produce distinguishable predictions")

        return dict(
            avg_left_vs_right    = avg_lr,
            avg_left_vs_forward  = avg_lf,
            avg_right_vs_forward = avg_rf,
            passed               = passed
        )

    # ──────────────────────────────────────────────────────────────────
    # VISUALIZE ALL RESULTS
    # ──────────────────────────────────────────────────────────────────
    def visualize_results(self, t1, t2, t3, t4):
        """Plot all test results in one figure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "World Model Unit Tests",
            fontsize=14, fontweight="bold"
        )

        # Test 1: Transition accuracy distribution
        ax = axes[0, 0]
        ax.hist(
            t1["similarities_predicted"],
            bins=30, alpha=0.7,
            color="steelblue", label="Predicted vs Actual"
        )
        ax.axvline(
            t1["avg_predicted"], color="blue",
            linestyle="--", label=f"Mean: {t1['avg_predicted']:.3f}"
        )
        ax.axvline(
            t1["avg_random"], color="red",
            linestyle="--", label=f"Random: {t1['avg_random']:.3f}"
        )
        ax.set_title(
            f"Test 1: Transition Accuracy\n"
            f"{'PASSED' if t1['passed'] else 'FAILED'}"
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # Test 2: Wall vs open state change
        ax = axes[0, 1]
        ax.hist(
            t2["wall_changes"], bins=20, alpha=0.7,
            color="red", label=f"Wall hit (n={len(t2['wall_changes'])})"
        )
        ax.hist(
            t2["open_changes"], bins=20, alpha=0.7,
            color="green",
            label=f"Open move (n={len(t2['open_changes'])})"
        )
        ax.set_title(
            f"Test 2: Wall Detection\n"
            f"{'PASSED' if t2['passed'] else 'FAILED'}"
        )
        ax.set_xlabel("Predicted State Change")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # Test 3: Rotation consistency
        ax = axes[1, 0]
        ax.hist(
            t3["similarities"], bins=20,
            color="purple", alpha=0.7
        )
        ax.axvline(
            t3["avg_sim"], color="black",
            linestyle="--",
            label=f"Mean: {t3['avg_sim']:.3f}"
        )
        ax.set_title(
            f"Test 3: Rotation Consistency (4x left = 360°)\n"
            f"{'PASSED' if t3['passed'] else 'FAILED'}"
        )
        ax.set_xlabel("Similarity to Original State")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # Test 4: Action distinguishability
        ax = axes[1, 1]
        action_pairs = ["Left vs Right", "Left vs Forward",
                        "Right vs Forward"]
        distances    = [
            t4["avg_left_vs_right"],
            t4["avg_left_vs_forward"],
            t4["avg_right_vs_forward"]
        ]
        bars = ax.bar(
            action_pairs, distances,
            color=["#e74c3c", "#3498db", "#2ecc71"]
        )
        for bar, d in zip(bars, distances):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{d:.4f}", ha="center", fontsize=9
            )
        ax.set_title(
            f"Test 4: Action Distinguishability\n"
            f"{'PASSED' if t4['passed'] else 'FAILED'}"
        )
        ax.set_ylabel("Cosine Distance Between Predictions")
        ax.set_ylim(0, max(distances) * 1.3)

        plt.tight_layout()
        Path("notebooks").mkdir(exist_ok=True)
        plt.savefig(
            "notebooks/unit_tests.png",
            dpi=150, bbox_inches="tight"
        )
        print("\nSaved: notebooks/unit_tests.png")
        plt.show()

    def run_all_tests(self):
        print("\nRunning World Model Unit Tests")
        print("="*50)

        t1 = self.test_transition_accuracy(n_samples=500)
        t2 = self.test_wall_detection(n_samples=200)
        t3 = self.test_rotation_consistency(n_trials=50)
        t4 = self.test_action_distinguishability(n_samples=300)

        self.visualize_results(t1, t2, t3, t4)

        # Final summary
        tests  = [t1, t2, t3, t4]
        names  = [
            "Transition Accuracy",
            "Wall Detection",
            "Rotation Consistency",
            "Action Distinguishability"
        ]
        passed = sum(t["passed"] for t in tests)

        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for name, test in zip(names, tests):
            status = "PASSED" if test["passed"] else "FAILED"
            print(f"  {status}  {name}")
        print(f"\n{passed}/{len(tests)} tests passed")
        self.env.close()


if __name__ == "__main__":
    tester = WorldModelTester("checkpoints/jepa_5000.pt")
    tester.run_all_tests()