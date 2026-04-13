"""
compare.py — Multi-Agent Comparison: Q-Learning vs DQN vs Double DQN
=====================================================================
Loads saved metrics OR runs fresh evaluation episodes, then produces:
  1. Side-by-side training curves (all 3 agents on same plot)
  2. Evaluation bar charts (mean ± std)
  3. A printed summary table with the winner analysis

Usage:
    # Plot from already-saved training metrics:
    python compare.py --from-saved --sim

    # Run fresh evaluation episodes (needs trained models):
    python compare.py --sim --episodes 10

    # Specify which agents to include:
    python compare.py --agents q dqn ddqn --sim --from-saved
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
from env.traffic_env  import TrafficEnv
from agent.dqn_agent  import DQNAgent
from agent.ddqn_agent import DDQNAgent
from agent.q_agent    import QLearningAgent

RESULTS  = os.path.join(os.path.dirname(__file__), "results")
SUMO_CFG = os.path.join(os.path.dirname(__file__), "sumo", "simulation.sumocfg")

AGENT_LABELS = {"q": "Q-Learning", "dqn": "DQN", "ddqn": "Double DQN"}
AGENT_COLORS = {"q": "#ED7D31",    "dqn": "#2E75B6", "ddqn": "#70AD47"}
AGENT_MARKS  = {"q": "o",         "dqn": "s",       "ddqn": "D"}


# ── CLI ──────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--agents",      nargs="+", choices=["q", "dqn", "ddqn"],
                   default=["q", "dqn", "ddqn"])
    p.add_argument("--episodes",    type=int, default=10,
                   help="Evaluation episodes per agent (default: 10)")
    p.add_argument("--sim",         action="store_true")
    p.add_argument("--from-saved",  action="store_true",
                   help="Load training metrics from .npy files (skip live eval)")
    return p.parse_args()


# ── Agent factory ────────────────────────────────────────────────────────
def load_agent(key, state_size, action_size):
    ext  = "pkl" if key == "q" else "pth"
    path = os.path.join(RESULTS, f"{key}_model.{ext}")
    if key == "q":
        agent = QLearningAgent(state_size, action_size)
    elif key == "dqn":
        agent = DQNAgent(state_size, action_size)
    else:
        agent = DDQNAgent(state_size, action_size)
    agent.load(path)
    agent.eps = 0.0   # greedy evaluation
    return agent


# ── One evaluation episode ───────────────────────────────────────────────
def eval_episode(env, agent_act):
    state = env.reset()
    total_r = 0.0
    done = False
    while not done:
        action = agent_act(state)
        state, r, done, _ = env.step(action)
        total_r += r
    m = env.get_metrics()
    return total_r, m


# ── Load from saved .npy files ───────────────────────────────────────────
def load_saved_metrics(agents):
    data = {}
    for key in agents:
        metrics = {}
        for name in ["rewards", "queues", "waiting", "switches"]:
            path = os.path.join(RESULTS, f"{key}_{name}.npy")
            if os.path.exists(path):
                metrics[name] = list(np.load(path))
            else:
                print(f"  [warn] {path} not found — run train.py first")
                metrics[name] = []
        data[key] = metrics
    return data


# ── Run live evaluation ───────────────────────────────────────────────────
def run_eval(agents, n_episodes, sim):
    env = TrafficEnv(SUMO_CFG, sim_mode=sim)
    data = {}

    for key in agents:
        label = AGENT_LABELS[key]
        print(f"\n  Evaluating {label} ({n_episodes} episodes)...")
        agent = load_agent(key, env.state_size, env.action_size)

        rewards, queues, waiting, switches = [], [], [], []
        for ep in range(1, n_episodes + 1):
            r, m = eval_episode(env, lambda s: agent.act(s, greedy=True))
            rewards.append(r)
            queues.append(m["avg_queue"])
            waiting.append(m["avg_waiting"])
            switches.append(m["total_switches"])
            print(f"    Ep {ep}: reward={r:>9.1f}  queue={m['avg_queue']:.2f}"
                  f"  wait={m['avg_waiting']:.1f}s  switches={m['total_switches']}")

        data[key] = {
            "rewards": rewards, "queues": queues,
            "waiting": waiting, "switches": switches,
        }

    env.close()
    return data


# ── Smoothing helper ─────────────────────────────────────────────────────
def _smooth(x, w=20):
    if len(x) < w:
        return np.array(x), list(range(1, len(x) + 1))
    return np.convolve(x, np.ones(w) / w, "valid"), list(range(w, len(x) + 1))


# ════════════════════════════════════════════════════════════════════════
# PLOT 1: Training curves — all agents on same axes
# ════════════════════════════════════════════════════════════════════════
def plot_training_curves(data: dict):
    agents = [k for k in ["q", "dqn", "ddqn"] if k in data]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Comparative RL Study: Q-Learning vs DQN vs Double DQN\n"
        "Adaptive Traffic Signal Control",
        fontsize=15, fontweight="bold"
    )

    metric_cfg = [
        (axes[0, 0], "rewards",  "Episode Reward",       "Reward",    True),
        (axes[0, 1], "queues",   "Avg Queue Length",     "Vehicles",  False),
        (axes[1, 0], "waiting",  "Avg Waiting Time",     "Seconds",   False),
        (axes[1, 1], "switches", "Phase Switches / Ep",  "Switches",  None),
    ]

    for ax, metric, title, ylabel, higher_better in metric_cfg:
        for key in agents:
            arr = data[key][metric]
            if not arr:
                continue
            color = AGENT_COLORS[key]
            label = AGENT_LABELS[key]
            eps   = range(1, len(arr) + 1)

            # Raw trace (faint)
            ax.plot(eps, arr, alpha=0.18, color=color, lw=0.8)

            # Smoothed trace
            sm, sm_eps = _smooth(arr)
            ax.plot(sm_eps, sm, color=color, lw=2.4,
                    marker=AGENT_MARKS[key], markevery=max(1, len(sm)//8),
                    markersize=5, label=label)

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F9FAFB")

        if higher_better is True:
            ax.text(0.02, 0.05, "↑ higher = better",
                    transform=ax.transAxes, fontsize=8, color="#666")
        elif higher_better is False:
            ax.text(0.02, 0.05, "↓ lower = better",
                    transform=ax.transAxes, fontsize=8, color="#666")

    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════
# PLOT 2: Bar charts (mean ± std) per metric
# ════════════════════════════════════════════════════════════════════════
def plot_bar_comparison(data: dict):
    agents = [k for k in ["q", "dqn", "ddqn"] if k in data and data[k]["rewards"]]
    labels = [AGENT_LABELS[k] for k in agents]
    colors = [AGENT_COLORS[k] for k in agents]

    metric_cfg = [
        ("rewards",  "Total Reward",       "Reward",   True),
        ("queues",   "Avg Queue Length",   "Vehicles", False),
        ("waiting",  "Avg Waiting Time",   "Seconds",  False),
        ("switches", "Phase Switches/Ep",  "Count",    None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle(
        "Agent Performance Summary (Mean ± Std)",
        fontsize=14, fontweight="bold"
    )

    for ax, (metric, title, ylabel, higher_better) in zip(axes, metric_cfg):
        means = [np.mean(data[k][metric]) for k in agents]
        stds  = [np.std(data[k][metric])  for k in agents]

        bars = ax.bar(labels, means, yerr=stds, capsize=9,
                      color=colors, alpha=0.85, edgecolor="white",
                      linewidth=1.5, error_kw={"elinewidth": 2})

        # Annotate each bar with its value
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.07,
                f"{m:.1f}",
                ha="center", va="bottom", fontweight="bold", fontsize=11
            )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#F9FAFB")
        ax.tick_params(axis="x", labelsize=9)

        if higher_better is True:
            ax.text(0.5, 0.01, "↑ higher = better", transform=ax.transAxes,
                    fontsize=8, color="#666", ha="center")
        elif higher_better is False:
            ax.text(0.5, 0.01, "↓ lower = better", transform=ax.transAxes,
                    fontsize=8, color="#666", ha="center")

    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_bar_charts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════
# PLOT 3: Episode-by-episode scatter for eval metrics
# ════════════════════════════════════════════════════════════════════════
def plot_eval_scatter(data: dict):
    agents = [k for k in ["q", "dqn", "ddqn"] if k in data and data[k]["rewards"]]
    # Only meaningful if data is from eval (not thousands of training eps)
    max_len = max(len(data[k]["rewards"]) for k in agents)
    if max_len > 200:
        print("  (Skipping scatter — data is training-length; use --from-saved=False for eval scatter)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Evaluation Episodes: All Agents", fontsize=14, fontweight="bold")

    metric_cfg = [
        (axes[0, 0], "rewards",  "Episode Reward",      "Reward"),
        (axes[0, 1], "queues",   "Avg Queue Length",    "Vehicles"),
        (axes[1, 0], "waiting",  "Avg Waiting Time",    "Seconds"),
        (axes[1, 1], "switches", "Phase Switches / Ep", "Count"),
    ]

    for ax, (metric, title, ylabel) in [(ax, cfg) for ax, cfg in zip(axes.flat, metric_cfg)]:
        for key in agents:
            arr = data[key][metric]
            if not arr:
                continue
            eps = range(1, len(arr) + 1)
            ax.plot(eps, arr, marker=AGENT_MARKS[key], color=AGENT_COLORS[key],
                    lw=1.8, ms=7, label=AGENT_LABELS[key])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F9FAFB")

    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_eval_episodes.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════
# PLOT 4: Convergence analysis — rolling mean & std band
# ════════════════════════════════════════════════════════════════════════
def plot_convergence(data: dict):
    agents = [k for k in ["q", "dqn", "ddqn"] if k in data and data[k]["rewards"]]
    if not any(len(data[k]["rewards"]) > 50 for k in agents):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Convergence Analysis: Rolling Mean ± 1σ (window=30)",
                 fontsize=14, fontweight="bold")

    for ax, (metric, title, ylabel, higher_better) in [
        (axes[0], ("rewards", "Episode Reward",    "Reward",   True)),
        (axes[1], ("queues",  "Avg Queue Length",  "Vehicles", False)),
    ]:
        W = 30
        for key in agents:
            arr = np.array(data[key][metric])
            if len(arr) < W:
                continue
            rolling_mean = np.convolve(arr, np.ones(W) / W, "valid")
            rolling_std  = np.array([arr[i:i+W].std() for i in range(len(arr)-W+1)])
            xs = range(W, len(arr) + 1)

            ax.plot(xs, rolling_mean, color=AGENT_COLORS[key],
                    lw=2.5, label=AGENT_LABELS[key])
            ax.fill_between(xs,
                            rolling_mean - rolling_std,
                            rolling_mean + rolling_std,
                            color=AGENT_COLORS[key], alpha=0.15)

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F9FAFB")
        suffix = "↑ better" if higher_better else "↓ better"
        ax.text(0.02, 0.04, suffix, transform=ax.transAxes,
                fontsize=9, color="#555")

    plt.tight_layout()
    path = os.path.join(RESULTS, "comparison_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ════════════════════════════════════════════════════════════════════════
# Summary table + winner analysis
# ════════════════════════════════════════════════════════════════════════
def print_summary(data: dict):
    agents = [k for k in ["q", "dqn", "ddqn"] if k in data and data[k]["rewards"]]

    print("\n" + "=" * 75)
    print("  COMPARATIVE STUDY — RESULTS SUMMARY")
    print("=" * 75)

    metrics_info = [
        ("rewards",  "Total Reward",      True),
        ("queues",   "Avg Queue Length",  False),
        ("waiting",  "Avg Waiting Time",  False),
        ("switches", "Phase Switches/Ep", None),
    ]

    # Header
    col = 22
    header = f"  {'Metric':<{col}}"
    for k in agents:
        header += f" {AGENT_LABELS[k]:>14}"
    print(header)
    print("  " + "-" * (col + 15 * len(agents)))

    scores = {k: 0 for k in agents}

    for metric, label, higher_better in metrics_info:
        means = {k: np.mean(data[k][metric]) for k in agents}
        stds  = {k: np.std(data[k][metric])  for k in agents}

        row = f"  {label:<{col}}"
        for k in agents:
            row += f"  {means[k]:>8.1f}±{stds[k]:<4.1f}"
        print(row)

        # Score best performer
        if higher_better is True:
            winner = max(agents, key=lambda k: means[k])
            scores[winner] += 1
        elif higher_better is False:
            winner = min(agents, key=lambda k: means[k])
            scores[winner] += 1

    print("  " + "-" * (col + 15 * len(agents)))
    score_row = f"  {'Score (wins)':>{col}}"
    for k in agents:
        score_row += f"  {scores[k]:>13}"
    print(score_row)
    print("=" * 75)

    # Overall winner
    overall_winner = max(agents, key=lambda k: scores[k])
    print(f"\n  OVERALL WINNER: {AGENT_LABELS[overall_winner]}")
    print("=" * 75)

    # Narrative analysis
    print("\n  ANALYSIS")
    print("  " + "-" * 60)

    rew = {k: np.mean(data[k]["rewards"])  for k in agents}
    q   = {k: np.mean(data[k]["queues"])   for k in agents}
    w   = {k: np.mean(data[k]["waiting"])  for k in agents}

    if "ddqn" in agents and "dqn" in agents:
        ddqn_vs_dqn_reward = ((rew["ddqn"] - rew["dqn"]) / abs(rew["dqn"]) * 100
                               if rew["dqn"] != 0 else 0)
        print(f"\n  Q-Learning (Tabular):")
        print(f"    + Simplest to implement; interpretable Q-table")
        print(f"    + Fast per-step updates (no replay buffer overhead)")
        print(f"    - Coarse discretisation loses nuance in queue lengths")
        print(f"    - Does not generalise to unseen queue combinations")
        if "q" in agents:
            print(f"    → Avg reward: {rew['q']:.1f}  |  Avg queue: {q['q']:.2f}")

        print(f"\n  DQN (Deep Q-Network):")
        print(f"    + Function approximation → generalises over continuous states")
        print(f"    + Experience replay breaks temporal correlations")
        print(f"    - Target-network bootstrap can overestimate Q-values")
        print(f"    → Avg reward: {rew['dqn']:.1f}  |  Avg queue: {q['dqn']:.2f}")

        print(f"\n  Double DQN:")
        print(f"    + Decouples action selection (online) from evaluation (target)")
        print(f"    + Eliminates maximisation bias → more accurate Q-values")
        print(f"    + Typically converges faster and to better optima than DQN")
        pct_str = f"{ddqn_vs_dqn_reward:+.1f}%"
        print(f"    → Avg reward: {rew['ddqn']:.1f}  |  Avg queue: {q['ddqn']:.2f}"
              f"  ({pct_str} vs DQN)")

    print(f"\n  RECOMMENDED FOR PRODUCTION: {AGENT_LABELS[overall_winner]}")
    print("  " + "-" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════════
def main():
    args = parse()
    os.makedirs(RESULTS, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Comparative RL Study — Traffic Signal Control")
    print("=" * 65)

    if args.from_saved:
        print("  Loading saved training metrics...")
        data = load_saved_metrics(args.agents)
    else:
        print(f"  Running {args.episodes}-episode evaluation per agent...")
        data = run_eval(args.agents, args.episodes, args.sim)

    # Generate all plots
    print("\n  Generating comparison plots...")
    plot_training_curves(data)
    plot_bar_comparison(data)
    plot_eval_scatter(data)
    plot_convergence(data)

    # Print analysis
    print_summary(data)

    print(f"  All plots saved to: {RESULTS}/")


if __name__ == "__main__":
    main()
