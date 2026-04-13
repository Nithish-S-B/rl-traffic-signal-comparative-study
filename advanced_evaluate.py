"""
Advanced Evaluation Script with detailed step-by-step analytics.

Provides fine-grained time-series analysis of traffic dynamics for
both DQN and Fixed-Time controllers.

Usage:
    python advanced_evaluate.py
    python advanced_evaluate.py --sim
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
from evaluate import FixedTimeController, FIXED_GREEN_TIME

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_model.pth")
SUMO_CFG = os.path.join(os.path.dirname(__file__), "sumo", "simulation.sumocfg")


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Traffic Evaluation")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    return parser.parse_args()


def run_detailed_episode(env: TrafficEnv, controller, use_greedy: bool = False):
    """Run episode and collect step-level data."""
    state = env.reset()
    done = False

    if hasattr(controller, "reset"):
        controller.reset()

    step_rewards = []
    step_queues = []
    step_waiting = []
    step_phases = []
    step_actions = []

    while not done:
        if use_greedy and hasattr(controller, "select_action_greedy"):
            action = controller.select_action_greedy(state)
        else:
            action = controller.select_action(state)

        state, reward, done, info = env.step(action)

        step_rewards.append(reward)
        step_queues.append(sum(info["queue_lengths"].values()))
        step_waiting.append(sum(info["waiting_times"].values()))
        step_phases.append(info["current_phase"])
        step_actions.append(action)

    return {
        "rewards": step_rewards,
        "queues": step_queues,
        "waiting": step_waiting,
        "phases": step_phases,
        "actions": step_actions,
        "throughput_final": env.throughput_history[-1] if env.throughput_history else 0,
    }


def plot_timeseries_comparison(rl_data: dict, fixed_data: dict, save_dir: str):
    """Plot step-level time series for both controllers."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.35)
    fig.suptitle("Step-Level Traffic Dynamics: DQN vs Fixed-Time",
                 fontsize=16, fontweight="bold")

    colors = {"DQN": "#2E75B6", "Fixed": "#ED7D31"}
    steps = range(len(rl_data["rewards"]))

    def smooth(data, w=30):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode="valid")

    # --- Reward per step ---
    for col, (label, data) in enumerate([("DQN (RL)", rl_data), ("Fixed-Time", fixed_data)]):
        ax = fig.add_subplot(gs[0, col])
        color = colors["DQN"] if col == 0 else colors["Fixed"]
        ax.plot(steps, data["rewards"], alpha=0.3, color=color, linewidth=0.6)
        if len(data["rewards"]) >= 30:
            sm = smooth(data["rewards"])
            ax.plot(range(30, len(data["rewards"]) + 1), sm, color=color,
                    linewidth=2.0, label="Smoothed")
        ax.set_title(f"{label}: Reward per Step", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    # --- Queue length per step ---
    for col, (label, data) in enumerate([("DQN (RL)", rl_data), ("Fixed-Time", fixed_data)]):
        ax = fig.add_subplot(gs[1, col])
        color = colors["DQN"] if col == 0 else colors["Fixed"]
        ax.fill_between(steps, data["queues"], alpha=0.3, color=color)
        ax.plot(steps, data["queues"], alpha=0.6, color=color, linewidth=0.8)
        if len(data["queues"]) >= 30:
            sm = smooth(data["queues"])
            ax.plot(range(30, len(data["queues"]) + 1), sm, color=color,
                    linewidth=2.0, label="Smoothed")
        ax.set_title(f"{label}: Queue Length per Step", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Queue (vehicles)")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    # --- Waiting time per step ---
    for col, (label, data) in enumerate([("DQN (RL)", rl_data), ("Fixed-Time", fixed_data)]):
        ax = fig.add_subplot(gs[2, col])
        color = colors["DQN"] if col == 0 else colors["Fixed"]
        ax.plot(steps, data["waiting"], alpha=0.5, color=color, linewidth=0.8)
        if len(data["waiting"]) >= 30:
            sm = smooth(data["waiting"])
            ax.plot(range(30, len(data["waiting"]) + 1), sm, color=color,
                    linewidth=2.0, label="Smoothed")
        ax.set_title(f"{label}: Waiting Time per Step", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Wait (s)")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    # --- Phase switching patterns ---
    for col, (label, data) in enumerate([("DQN (RL)", rl_data), ("Fixed-Time", fixed_data)]):
        ax = fig.add_subplot(gs[3, col])
        color = colors["DQN"] if col == 0 else colors["Fixed"]
        ax.step(steps, data["phases"], where="post", color=color,
                linewidth=1.5, alpha=0.85)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["NS Green", "EW Green"])
        ax.set_title(f"{label}: Phase Switching", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Phase")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

        # Count switches
        switches = sum(1 for i in range(1, len(data["phases"]))
                       if data["phases"][i] != data["phases"][i - 1])
        ax.text(0.98, 0.97, f"Switches: {switches}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="gray")

    save_path = os.path.join(save_dir, "timeseries_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_distribution_analysis(rl_data: dict, fixed_data: dict, save_dir: str):
    """Plot distribution analysis (histograms + box plots)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Distribution Analysis: DQN vs Fixed-Time",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("rewards", "Step Reward", "Reward"),
        ("queues", "Queue Length Distribution", "Vehicles"),
        ("waiting", "Waiting Time Distribution", "Seconds"),
    ]
    colors_rl = "#2E75B6"
    colors_fx = "#ED7D31"

    for col, (key, title, xlabel) in enumerate(metrics):
        # Histogram
        ax = axes[0, col]
        ax.hist(rl_data[key], bins=30, alpha=0.6, color=colors_rl,
                label="DQN (RL)", density=True, edgecolor="white")
        ax.hist(fixed_data[key], bins=30, alpha=0.6, color=colors_fx,
                label="Fixed-Time", density=True, edgecolor="white")
        ax.set_title(f"{title}\n(Histogram)", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

        # Box plot
        ax = axes[1, col]
        bp = ax.boxplot(
            [rl_data[key], fixed_data[key]],
            labels=["DQN (RL)", "Fixed-Time"],
            patch_artist=True,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2.5, color="white"),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=2),
        )
        bp["boxes"][0].set_facecolor(colors_rl)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(colors_fx)
        bp["boxes"][1].set_alpha(0.7)
        ax.set_title(f"{title}\n(Box Plot)", fontweight="bold")
        ax.set_ylabel(xlabel)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#F8F9FA")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "distribution_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_statistics(rl_data: dict, fixed_data: dict):
    """Print detailed statistical comparison."""
    print("\n" + "=" * 70)
    print("  DETAILED STATISTICAL ANALYSIS")
    print("=" * 70)

    metrics = [
        ("rewards", "Reward per Step"),
        ("queues", "Queue Length (vehicles)"),
        ("waiting", "Waiting Time (s)"),
    ]

    for key, label in metrics:
        rl = np.array(rl_data[key])
        fx = np.array(fixed_data[key])
        print(f"\n  {label}:")
        print(f"  {'':20} {'DQN':>12} {'Fixed':>12}")
        print(f"  {'─'*44}")
        print(f"  {'Mean':20} {np.mean(rl):>12.3f} {np.mean(fx):>12.3f}")
        print(f"  {'Std Dev':20} {np.std(rl):>12.3f} {np.std(fx):>12.3f}")
        print(f"  {'Median':20} {np.median(rl):>12.3f} {np.median(fx):>12.3f}")
        print(f"  {'Min':20} {np.min(rl):>12.3f} {np.min(fx):>12.3f}")
        print(f"  {'Max':20} {np.max(rl):>12.3f} {np.max(fx):>12.3f}")
        print(f"  {'95th Pct':20} {np.percentile(rl, 95):>12.3f} {np.percentile(fx, 95):>12.3f}")

    print(f"\n  {'Throughput (total)':20} {rl_data['throughput_final']:>12} {fixed_data['throughput_final']:>12}")

    # Phase switching
    rl_switches = sum(1 for i in range(1, len(rl_data["phases"]))
                      if rl_data["phases"][i] != rl_data["phases"][i - 1])
    fx_switches = sum(1 for i in range(1, len(fixed_data["phases"]))
                      if fixed_data["phases"][i] != fixed_data["phases"][i - 1])
    print(f"\n  {'Phase Switches':20} {rl_switches:>12} {fx_switches:>12}")
    print("=" * 70)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Advanced Traffic Signal Evaluation")
    print("=" * 65)

    env = TrafficEnv(sumo_cfg_path=SUMO_CFG, sim_mode=args.sim)

    # DQN agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    if os.path.exists(args.model):
        agent.load(args.model)
        agent.epsilon = 0.0
    else:
        print(f"[WARNING] No model at {args.model}. Using untrained agent.")

    # Fixed controller
    fixed_ctrl = FixedTimeController()

    print("\n[Advanced] Running DQN detailed episode...")
    rl_data = run_detailed_episode(env, agent, use_greedy=True)

    print("[Advanced] Running Fixed-Time detailed episode...")
    fixed_data = run_detailed_episode(env, fixed_ctrl)

    env.close()

    # Plots and stats
    print("\n[Advanced] Generating time-series comparison...")
    plot_timeseries_comparison(rl_data, fixed_data, RESULTS_DIR)

    print("[Advanced] Generating distribution analysis...")
    plot_distribution_analysis(rl_data, fixed_data, RESULTS_DIR)

    print_statistics(rl_data, fixed_data)
    print(f"\n  All plots saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
