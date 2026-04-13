"""
train.py — Comparative RL Training for Adaptive Traffic Signal Control
=======================================================================
Train any of three agents against the same environment and save
per-episode metrics for later comparison.

Usage:
    python train.py --agent q    --sim --episodes 500
    python train.py --agent dqn  --sim --episodes 500
    python train.py --agent ddqn --sim --episodes 500
    python train.py --agent all  --sim --episodes 500   # train all three

SUMO mode (requires SUMO + SUMO_HOME):
    python train.py --agent ddqn --episodes 300
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.traffic_env  import TrafficEnv
from agent.dqn_agent  import DQNAgent
from agent.ddqn_agent import DDQNAgent
from agent.q_agent    import QLearningAgent

RESULTS  = os.path.join(os.path.dirname(__file__), "results")
SUMO_CFG = os.path.join(os.path.dirname(__file__), "sumo", "simulation.sumocfg")

AGENT_CHOICES = ["q", "dqn", "ddqn", "all"]
AGENT_LABELS  = {"q": "Q-Learning", "dqn": "DQN", "ddqn": "Double DQN"}
AGENT_COLORS  = {"q": "#ED7D31", "dqn": "#2E75B6", "ddqn": "#70AD47"}


def parse():
    p = argparse.ArgumentParser(
        description="Train traffic signal agents (Q-Learning / DQN / DDQN)")
    p.add_argument("--agent",    choices=AGENT_CHOICES, default="dqn",
                   help="Which agent to train (default: dqn)")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--sim",      action="store_true")
    p.add_argument("--gui",      action="store_true")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--verify",   action="store_true")
    return p.parse_args()


def make_agent(key, state_size, action_size):
    if key == "q":
        return QLearningAgent(state_size, action_size)
    if key == "dqn":
        return DQNAgent(state_size, action_size)
    if key == "ddqn":
        return DDQNAgent(state_size, action_size)
    raise ValueError(f"Unknown agent: {key!r}")


def model_path(key):
    ext = "pkl" if key == "q" else "pth"
    return os.path.join(RESULTS, f"{key}_model.{ext}")


def _smooth(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, "valid")


def train_agent(key, args):
    label = AGENT_LABELS[key]
    print("\n" + "="*65)
    print(f"  Training: {label}")
    print(f"  Episodes : {args.episodes}  |  Sim: {args.sim}")
    print("="*65)

    os.makedirs(RESULTS, exist_ok=True)
    env   = TrafficEnv(SUMO_CFG, use_gui=args.gui, sim_mode=args.sim)
    agent = make_agent(key, env.state_size, env.action_size)

    mpath = model_path(key)
    if args.resume and os.path.exists(mpath):
        agent.load(mpath)

    ep_rewards, ep_queues, ep_waiting, ep_switches = [], [], [], []
    best_r = -np.inf
    t0     = time.time()

    for ep in range(1, args.episodes + 1):
        state   = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            action = agent.act(state)
            next_state, r, done, info = env.step(action)

            if key == "q":
                agent.learn_step(state, action, r, next_state, done)
            else:
                agent.remember(state, action, r, next_state, done)
                agent.learn()

            state    = next_state
            total_r += r

        agent.decay_eps()
        m = env.get_metrics()

        ep_rewards.append(total_r)
        ep_queues.append(m["avg_queue"])
        ep_waiting.append(m["avg_waiting"])
        ep_switches.append(m["total_switches"])

        if total_r > best_r:
            best_r = total_r
            ext = "pkl" if key == "q" else "pth"
            best_path = os.path.join(RESULTS, f"{key}_model_best.{ext}")
            agent.save(best_path)

        if ep % 50 == 0:
            st = agent.stats()
            elapsed = time.time() - t0
            print(f"  Ep {ep:>4}/{args.episodes} | "
                  f"Reward {np.mean(ep_rewards[-50:]):>9.1f} | "
                  f"Queue {np.mean(ep_queues[-50:]):>5.2f} | "
                  f"Switches {np.mean(ep_switches[-50:]):>5.1f} | "
                  f"eps={st['eps']:.3f} | {elapsed:.0f}s")

    agent.save(mpath)
    env.close()

    metrics = {
        "rewards":  ep_rewards,
        "queues":   ep_queues,
        "waiting":  ep_waiting,
        "switches": ep_switches,
    }
    for name, arr in metrics.items():
        np.save(os.path.join(RESULTS, f"{key}_{name}.npy"), np.array(arr))

    elapsed = time.time() - t0
    print(f"\n  {label} done in {elapsed:.0f}s  |  Best reward: {best_r:.1f}")
    _plot_single(key, label, metrics)

    if args.verify:
        env2 = TrafficEnv(SUMO_CFG, sim_mode=args.sim)
        agent.eps = 0.0
        _verify_reactive(agent, env2, label)

    return metrics


def _plot_single(key, label, metrics):
    color = AGENT_COLORS[key]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{label} Training — Adaptive Traffic Signal Control",
                 fontsize=14, fontweight="bold")

    pairs = [
        (axes[0, 0], metrics["rewards"],  "Episode Reward",      "Reward"),
        (axes[0, 1], metrics["queues"],   "Avg Queue Length",    "Vehicles"),
        (axes[1, 0], metrics["waiting"],  "Avg Waiting Time",    "Seconds"),
        (axes[1, 1], metrics["switches"], "Phase Switches / Ep", "Switches"),
    ]
    eps = range(1, len(metrics["rewards"]) + 1)
    for ax, data, title, ylabel in pairs:
        ax.plot(eps, data, alpha=0.25, color=color, lw=0.7)
        if len(data) >= 20:
            sm = _smooth(data)
            ax.plot(range(20, len(data) + 1), sm, color=color, lw=2.2,
                    label="Smoothed (20-ep)")
            ax.legend(fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F8F9FA")

    plt.tight_layout()
    path = os.path.join(RESULTS, f"{key}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves -> {path}")


def _verify_reactive(agent, env, label):
    print(f"\n{'='*70}")
    print(f"  REACTIVITY CHECK -- {label} -- 60 steps greedy")
    print(f"{'='*70}")
    print(f"{'Step':>4}  {'Phase':>9}  {'Q_N':>4}{'Q_S':>4}{'Q_E':>4}{'Q_W':>4}  {'Action':>8}  Switch?")
    print("-"*70)
    state = env.reset()
    for step in range(1, 61):
        action = agent.act(state, greedy=True)
        state, reward, done, info = env.step(action)
        ph = "NS-green" if info["phase"] == 0 else "EW-green"
        q  = info["queues"]
        print(f"{step:>4}  {ph:>9}  {q['N']:>4}{q['S']:>4}{q['E']:>4}{q['W']:>4}"
              f"  {'SWITCH' if action else 'keep':>8}  {'<<< YES' if info['switched'] else ''}")
        if done:
            break
    print(f"\n  Total switches: {sum(env.switch_hist)}")
    print("="*70)
    env.close()


def train_all(args):
    all_metrics = {}
    for key in ["q", "dqn", "ddqn"]:
        all_metrics[key] = train_agent(key, args)
    return all_metrics


if __name__ == "__main__":
    args = parse()
    if args.agent == "all":
        train_all(args)
        print("\n\n  All agents trained. Run  python compare.py --sim  to compare.")
    else:
        train_agent(args.agent, args)
        print(f"\n  Run  python compare.py --sim  to compare results.")
