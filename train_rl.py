#!/usr/bin/env python3
"""
PatchCascade SOC — RL Training Script
=======================================

Train reinforcement learning agents on PatchCascade SOC using
Stable-Baselines3. Supports multiple algorithms, all 5 task levels,
and produces training curves + checkpoints.

Usage:
    # Train PPO on medium (default)
    python train_rl.py

    # Train on all levels with curriculum learning
    python train_rl.py --curriculum

    # Train specific algorithm on specific level
    python train_rl.py --algo ppo --task medium --steps 50000

    # Quick test run
    python train_rl.py --task easy --steps 5000

Requirements:
    pip install stable-baselines3 gymnasium matplotlib

Author: PatchCascade SOC Team
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with:")
    print("    pip install 'stable-baselines3[extra]' gymnasium matplotlib")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gym_wrapper import PatchCascadeGymEnv


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================


class RewardTrackingCallback(BaseCallback):
    """
    Custom callback to track episode rewards, lengths, and success rates
    during training. Saves results for plotting.
    """

    def __init__(self, check_freq: int = 100, log_dir: str = "results", verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self.timesteps_log: list[int] = []
        self.mean_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Check for completed episodes in monitor info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    # Track success (raw reward > 0 means net positive outcome)
                    raw_reward = info.get("episode_reward", ep_reward)
                    self.episode_successes.append(raw_reward > 0)

        # Periodic logging
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            success_rate = np.mean(self.episode_successes[-100:]) if self.episode_successes else 0.0

            self.timesteps_log.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)

            if self.verbose > 0:
                print(
                    f"  [{self.num_timesteps:>7d} steps] "
                    f"mean_reward={mean_reward:>8.2f} | "
                    f"mean_ep_len={mean_length:>5.1f} | "
                    f"success_rate={success_rate:.1%} | "
                    f"episodes={len(self.episode_rewards)}"
                )

        return True

    def save_results(self, filename: str = "training_log.json"):
        """Save training log to JSON file."""
        data = {
            "timesteps": self.timesteps_log,
            "mean_rewards": self.mean_rewards,
            "all_rewards": self.episode_rewards,
            "all_lengths": self.episode_lengths,
            "total_episodes": len(self.episode_rewards),
        }
        filepath = self.log_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"  📊 Training log saved to {filepath}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def make_env(task_level: str, seed: int = 42) -> PatchCascadeGymEnv:
    """Create a monitored PatchCascade environment."""
    env = PatchCascadeGymEnv(task_level=task_level, seed=seed)
    return env


ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
}


def train_single(
    algo_name: str = "ppo",
    task_level: str = "medium",
    total_timesteps: int = 50_000,
    seed: int = 42,
    output_dir: str = "results",
    n_envs: int = 4,
) -> dict:
    """
    Train a single agent on a specific task level.

    Args:
        algo_name: Algorithm ("ppo", "a2c")
        task_level: Task level ("easy", "medium", "hard", etc.)
        total_timesteps: Total training steps
        seed: Random seed
        output_dir: Directory for results & checkpoints
        n_envs: Number of parallel environments

    Returns:
        Dict with training results and model path.
    """
    if not HAS_SB3:
        return {"error": "stable-baselines3 not installed"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training {algo_name.upper()} on {task_level}")
    print(f"  Steps: {total_timesteps:,} | Envs: {n_envs} | Seed: {seed}")
    print(f"{'='*60}\n")

    # Create vectorized environments
    def make_env_fn(rank: int):
        def _init():
            env = PatchCascadeGymEnv(
                task_level=task_level,
                seed=seed + rank,
                normalize_obs=True,
                reward_scale=0.01,
            )
            env = Monitor(env)
            return env
        return _init

    vec_env = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])

    # Create evaluation environment
    eval_env = Monitor(PatchCascadeGymEnv(
        task_level=task_level,
        seed=seed + 1000,
        normalize_obs=True,
        reward_scale=0.01,
    ))

    # Select algorithm
    AlgoClass = ALGO_MAP.get(algo_name.lower())
    if AlgoClass is None:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from: {list(ALGO_MAP.keys())}")

    # Hyperparameters tuned for PatchCascade
    if algo_name == "ppo":
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )
    elif algo_name == "a2c":
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
            learning_rate=7e-4,
            n_steps=16,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )
    else:
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=seed,
        )

    # Callbacks
    reward_callback = RewardTrackingCallback(
        check_freq=500, log_dir=str(output_path), verbose=1
    )

    model_prefix = f"{algo_name}_{task_level}"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 5, 1000),
        save_path=str(output_path / "checkpoints"),
        name_prefix=model_prefix,
    )

    # Train!
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_callback, checkpoint_callback],
        progress_bar=True,
    )
    training_time = time.time() - t0

    # Save final model
    model_path = output_path / f"{model_prefix}_final"
    model.save(str(model_path))
    print(f"\n  💾 Model saved to {model_path}")

    # Save training log
    reward_callback.save_results(f"{model_prefix}_log.json")

    # Evaluate final model
    print(f"\n  📊 Evaluating final model...")
    eval_rewards = []
    eval_successes = []

    for ep in range(20):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        eval_rewards.append(total_reward)
        eval_successes.append(info.get("num_vulns", 1) == 0)

    mean_eval_reward = np.mean(eval_rewards)
    eval_success_rate = np.mean(eval_successes)

    results = {
        "algorithm": algo_name,
        "task_level": task_level,
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "total_episodes": len(reward_callback.episode_rewards),
        "eval_mean_reward": float(mean_eval_reward),
        "eval_success_rate": float(eval_success_rate),
        "model_path": str(model_path),
        "training_log": {
            "timesteps": reward_callback.timesteps_log,
            "mean_rewards": [float(r) for r in reward_callback.mean_rewards],
        },
    }

    # Save results summary
    results_path = output_path / f"{model_prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ Training complete!")
    print(f"     Eval Mean Reward: {mean_eval_reward:.4f}")
    print(f"     Eval Success Rate: {eval_success_rate:.1%}")
    print(f"     Training Time: {training_time:.1f}s")
    print(f"     Total Episodes: {results['total_episodes']}")

    vec_env.close()
    eval_env.close()

    return results


def train_curriculum(
    algo_name: str = "ppo",
    steps_per_level: int = 20_000,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    """
    Curriculum learning: train progressively on easy → medium → hard.

    The same model is trained across increasing difficulty levels,
    demonstrating that the 5-level curriculum design facilitates
    progressive skill acquisition.

    Args:
        algo_name: Algorithm to use.
        steps_per_level: Training steps per curriculum stage.
        seed: Random seed.
        output_dir: Output directory.

    Returns:
        Dict with curriculum training results.
    """
    if not HAS_SB3:
        return {"error": "stable-baselines3 not installed"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Curriculum: progressive difficulty
    curriculum = ["easy", "medium", "hard"]

    print(f"\n{'='*60}")
    print(f"  🎓 CURRICULUM LEARNING: {' → '.join(curriculum)}")
    print(f"  Algorithm: {algo_name.upper()} | Steps/level: {steps_per_level:,}")
    print(f"{'='*60}\n")

    model = None
    all_results = []

    for stage_idx, task_level in enumerate(curriculum):
        print(f"\n  📚 Stage {stage_idx + 1}/{len(curriculum)}: {task_level.upper()}")

        # Create environment for this level
        def make_env_fn(rank: int):
            def _init():
                env = PatchCascadeGymEnv(
                    task_level=task_level,
                    seed=seed + rank + stage_idx * 100,
                    normalize_obs=True,
                    reward_scale=0.01,
                )
                env = Monitor(env)
                return env
            return _init

        vec_env = DummyVecEnv([make_env_fn(i) for i in range(4)])

        AlgoClass = ALGO_MAP[algo_name.lower()]

        if model is None:
            # First stage: create new model
            model = AlgoClass(
                "MlpPolicy",
                vec_env,
                verbose=0,
                seed=seed,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                ),
            )
        else:
            # Subsequent stages: transfer the model to new env
            model.set_env(vec_env)

        # Train on this level
        callback = RewardTrackingCallback(
            check_freq=500, log_dir=str(output_path), verbose=1
        )

        model.learn(
            total_timesteps=steps_per_level,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )

        # Save stage checkpoint
        stage_path = output_path / f"curriculum_{algo_name}_stage{stage_idx}_{task_level}"
        model.save(str(stage_path))

        # Evaluate on this level
        eval_env = Monitor(PatchCascadeGymEnv(
            task_level=task_level, seed=seed + 2000, normalize_obs=True,
            reward_scale=0.01
        ))

        eval_rewards = []
        for _ in range(10):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            eval_rewards.append(total_reward)

        eval_env.close()

        stage_result = {
            "stage": stage_idx,
            "task_level": task_level,
            "eval_mean_reward": float(np.mean(eval_rewards)),
            "eval_std_reward": float(np.std(eval_rewards)),
            "episodes_trained": len(callback.episode_rewards),
        }
        all_results.append(stage_result)

        print(f"     Stage {stage_idx + 1} complete: mean_reward={stage_result['eval_mean_reward']:.4f}")

        vec_env.close()

    # Save final curriculum model
    final_path = output_path / f"curriculum_{algo_name}_final"
    model.save(str(final_path))

    curriculum_results = {
        "type": "curriculum_learning",
        "algorithm": algo_name,
        "stages": all_results,
        "model_path": str(final_path),
    }

    results_path = output_path / f"curriculum_{algo_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(curriculum_results, f, indent=2)

    print(f"\n  🎓 Curriculum learning complete!")
    print(f"     Final model: {final_path}")

    return curriculum_results


# =============================================================================
# PLOTTING
# =============================================================================


def plot_training_curves(results_dir: str = "results"):
    """
    Generate publication-quality training curve plots from saved logs.

    Produces:
    - training_curves.png: Reward vs timesteps for each task level
    - curriculum_comparison.png: Curriculum vs direct training comparison
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not installed. Skipping plot generation.")
        return

    results_path = Path(results_dir)

    # Collect all training logs
    log_files = list(results_path.glob("*_log.json"))
    if not log_files:
        print("  No training logs found. Run training first.")
        return

    # ── Plot 1: Training Curves per Task Level ────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "easy": "#00ff88",
        "medium": "#ffaa00",
        "hard": "#ff3355",
        "incident_response": "#aa55ff",
        "zero_day": "#00bbff",
    }

    for log_file in sorted(log_files):
        with open(log_file) as f:
            data = json.load(f)

        # Extract task level from filename
        name = log_file.stem.replace("_log", "")
        parts = name.split("_")
        algo = parts[0]
        task = "_".join(parts[1:])

        if "timesteps" in data and "mean_rewards" in data:
            color = colors.get(task, "#ffffff")
            label = f"{algo.upper()} on {task}"
            ax.plot(
                data["timesteps"],
                data["mean_rewards"],
                label=label,
                color=color,
                linewidth=2,
                alpha=0.9,
            )

    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward (100-episode window)", fontsize=12)
    ax.set_title("PatchCascade SOC — RL Training Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    plot_path = results_path / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.close()
    print(f"  📈 Training curves saved to {plot_path}")

    # ── Plot 2: Results Summary Bar Chart ─────────────────────────
    result_files = list(results_path.glob("*_results.json"))
    if result_files:
        fig, ax = plt.subplots(figsize=(10, 5))

        labels = []
        rewards = []
        bar_colors = []

        for rf in sorted(result_files):
            if "curriculum" in rf.stem:
                continue
            with open(rf) as f:
                data = json.load(f)
            label = f"{data.get('algorithm', '?').upper()}\n{data.get('task_level', '?')}"
            labels.append(label)
            rewards.append(data.get("eval_mean_reward", 0))
            task = data.get("task_level", "")
            bar_colors.append(colors.get(task, "#888"))

        if labels:
            bars = ax.bar(labels, rewards, color=bar_colors, edgecolor="#333", linewidth=1.5)
            ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
            ax.set_title("PatchCascade SOC — Agent Evaluation Results", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_facecolor("#0a0a1a")
            fig.patch.set_facecolor("#0a0a1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#333")

            # Add value labels on bars
            for bar, val in zip(bars, rewards):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    color="white", fontweight="bold", fontsize=10,
                )

            plt.tight_layout()
            bar_path = results_path / "evaluation_results.png"
            plt.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
            plt.close()
            print(f"  📊 Evaluation results saved to {bar_path}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PatchCascade SOC — RL Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_rl.py                           # Train PPO on medium (default)
  python train_rl.py --task easy --steps 10000  # Quick test on easy
  python train_rl.py --curriculum               # Curriculum learning
  python train_rl.py --all                      # Train on all 5 levels
  python train_rl.py --plot                     # Generate training curves
        """,
    )

    parser.add_argument("--algo", default="ppo", choices=["ppo", "a2c"],
                        help="RL algorithm to use (default: ppo)")
    parser.add_argument("--task", default="medium",
                        choices=["easy", "medium", "hard", "incident_response", "zero_day"],
                        help="Task level (default: medium)")
    parser.add_argument("--steps", type=int, default=50_000,
                        help="Total training timesteps (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")

    # Special modes
    parser.add_argument("--curriculum", action="store_true",
                        help="Run curriculum learning (easy → medium → hard)")
    parser.add_argument("--all", action="store_true",
                        help="Train on all 5 task levels")
    parser.add_argument("--plot", action="store_true",
                        help="Generate training curve plots from saved logs")

    args = parser.parse_args()

    if not HAS_SB3 and not args.plot:
        print("\n❌ stable-baselines3 is required for training.")
        print("   Install with: pip install 'stable-baselines3[extra]' gymnasium matplotlib")
        sys.exit(1)

    # Plot mode
    if args.plot:
        plot_training_curves(args.output)
        return

    # Curriculum mode
    if args.curriculum:
        results = train_curriculum(
            algo_name=args.algo,
            steps_per_level=args.steps,
            seed=args.seed,
            output_dir=args.output,
        )
        plot_training_curves(args.output)
        return

    # All levels mode
    if args.all:
        all_results = []
        for level in ["easy", "medium", "hard", "incident_response", "zero_day"]:
            result = train_single(
                algo_name=args.algo,
                task_level=level,
                total_timesteps=args.steps,
                seed=args.seed,
                output_dir=args.output,
                n_envs=args.n_envs,
            )
            all_results.append(result)

        # Print summary table
        print(f"\n{'='*70}")
        print(f"  📊 TRAINING SUMMARY — {args.algo.upper()}")
        print(f"{'='*70}")
        print(f"  {'Task':<20} {'Eval Reward':>12} {'Success Rate':>14} {'Episodes':>10}")
        print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*10}")
        for r in all_results:
            if "error" not in r:
                print(f"  {r['task_level']:<20} {r['eval_mean_reward']:>12.4f} "
                      f"{r['eval_success_rate']:>13.1%} {r['total_episodes']:>10}")
        print(f"{'='*70}\n")

        plot_training_curves(args.output)
        return

    # Single training run
    result = train_single(
        algo_name=args.algo,
        task_level=args.task,
        total_timesteps=args.steps,
        seed=args.seed,
        output_dir=args.output,
        n_envs=args.n_envs,
    )

    plot_training_curves(args.output)


if __name__ == "__main__":
    main()
