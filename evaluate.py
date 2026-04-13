"""
evaluate.py
===========
Compare DQN (traffic-reactive) vs Fixed-Time controller.

Run:
    python evaluate.py --sim
    python evaluate.py          # with SUMO

The evaluation also prints a SWITCH PATTERN table so you can see
*when* each controller changes the phase, proving the DQN reacts
to queue lengths rather than a clock.
"""

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent

RESULTS  = os.path.join(os.path.dirname(__file__), "results")
MODEL    = os.path.join(RESULTS, "dqn_model.pth")
SUMO_CFG = os.path.join(os.path.dirname(__file__), "sumo", "simulation.sumocfg")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--sim", action="store_true")
    p.add_argument("--model", default=MODEL)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════
class FixedTimeController:
    """
    Baseline: switches phase every `green` steps regardless of queues.
    This is what most real intersections use today.
    """
    def __init__(self, green: int = 30):
        self.green = green
        self._timer = 0

    def reset(self):
        self._timer = 0

    def act(self, state) -> int:
        self._timer += 1
        if self._timer >= self.green:
            self._timer = 0
            return 1   # switch
        return 0       # keep


# ════════════════════════════════════════════════════════════════════════
def run_episode(env, act_fn, reset_fn=None):
    """Returns step-level data for one episode."""
    state = env.reset()
    if reset_fn: reset_fn()

    steps_data = []
    total_r = 0.0
    done = False

    while not done:
        action = act_fn(state)
        state, r, done, info = env.step(action)
        total_r += r
        steps_data.append({
            "step":     env._step,
            "phase":    info["phase"],
            "switched": info["switched"],
            "action":   action,
            "q_N":      info["queues"]["N"],
            "q_S":      info["queues"]["S"],
            "q_E":      info["queues"]["E"],
            "q_W":      info["queues"]["W"],
            "reward":   r,
        })

    return total_r, env.get_metrics(), steps_data


# ════════════════════════════════════════════════════════════════════════
def print_switch_table(steps_data, label, max_rows=80):
    """
    Print the first `max_rows` steps showing queue lengths and when the
    phase switches — this visually proves or disproves traffic-reactivity.
    """
    print(f"\n  {'─'*70}")
    print(f"  {label}  —  Switch Pattern (first {max_rows} steps)")
    print(f"  {'─'*70}")
    print(f"  {'Step':>4}  {'Phase':>9}  {'Q_N':>4}{'Q_S':>4}{'Q_E':>4}{'Q_W':>4}"
          f"  {'Action':>7}  {'Switch?':>10}")
    print(f"  {'─'*70}")

    for d in steps_data[:max_rows]:
        phase_s  = "NS-green" if d["phase"] == 0 else "EW-green"
        act_s    = "SWITCH"   if d["action"] == 1 else "keep"
        sw_s     = "<<< YES" if d["switched"] else ""
        print(f"  {d['step']:>4}  {phase_s:>9}  "
              f"{d['q_N']:>4}{d['q_S']:>4}{d['q_E']:>4}{d['q_W']:>4}"
              f"  {act_s:>7}  {sw_s}")

    switch_steps = [d["step"] for d in steps_data if d["switched"]]
    print(f"\n  Switches at steps: {switch_steps[:20]}")
    if len(switch_steps) > 1:
        gaps = [switch_steps[i+1]-switch_steps[i]
                for i in range(min(10, len(switch_steps)-1))]
        print(f"  Gap between switches: {gaps}")
        if len(set(gaps)) == 1:
            print(f"  ⚠  All gaps are equal ({gaps[0]}) → FIXED TIMER behaviour")
        else:
            print(f"  ✓  Gaps are variable ({min(gaps)}–{max(gaps)}) → TRAFFIC-REACTIVE")
    print(f"  {'─'*70}\n")


# ════════════════════════════════════════════════════════════════════════
def evaluate(args):
    os.makedirs(RESULTS, exist_ok=True)

    env   = TrafficEnv(SUMO_CFG, sim_mode=args.sim)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(args.model)
    agent.eps = 0.0    # greedy evaluation — no exploration noise
    fixed = FixedTimeController(green=30)

    rl_rewards,  rl_queues,  rl_waiting,  rl_tp  = [], [], [], []
    fx_rewards,  fx_queues,  fx_waiting,  fx_tp  = [], [], [], []
    rl_steps_ep1 = fx_steps_ep1 = None

    print("\n" + "═"*65)
    print(f"  DQN (traffic-reactive) — {args.episodes} eval episodes")
    print("═"*65)

    for ep in range(1, args.episodes+1):
        r, m, steps = run_episode(env, lambda s: agent.act(s, greedy=True))
        rl_rewards.append(r); rl_queues.append(m["avg_queue"])
        rl_waiting.append(m["avg_waiting"]); rl_tp.append(m["total_throughput"])
        if ep == 1: rl_steps_ep1 = steps
        print(f"  Ep {ep}: reward={r:>9.1f}  queue={m['avg_queue']:.2f}"
              f"  wait={m['avg_waiting']:.1f}s  switches={m['total_switches']}")

    print("\n" + "═"*65)
    print(f"  Fixed-Time (30s/phase) — {args.episodes} eval episodes")
    print("═"*65)

    for ep in range(1, args.episodes+1):
        r, m, steps = run_episode(env, fixed.act, fixed.reset)
        fx_rewards.append(r); fx_queues.append(m["avg_queue"])
        fx_waiting.append(m["avg_waiting"]); fx_tp.append(m["total_throughput"])
        if ep == 1: fx_steps_ep1 = steps
        print(f"  Ep {ep}: reward={r:>9.1f}  queue={m['avg_queue']:.2f}"
              f"  wait={m['avg_waiting']:.1f}s  switches={m['total_switches']}")

    env.close()

    # ── switch pattern tables (PROOF OF REACTIVITY) ──────────────────
    print_switch_table(rl_steps_ep1, "DQN Agent")
    print_switch_table(fx_steps_ep1, "Fixed-Time Controller")

    # ── summary table ─────────────────────────────────────────────────
    print("═"*65)
    print("  SUMMARY")
    print("═"*65)
    print(f"  {'Metric':<22} {'DQN':>10} {'Fixed':>10} {'RL vs Fixed':>14}")
    print(f"  {'─'*58}")

    for lbl, rl_v, fx_v, hi in [
        ("Total Reward",    np.mean(rl_rewards), np.mean(fx_rewards), True),
        ("Avg Queue",       np.mean(rl_queues),  np.mean(fx_queues),  False),
        ("Avg Wait (s)",    np.mean(rl_waiting), np.mean(fx_waiting), False),
        ("Throughput",      np.mean(rl_tp),      np.mean(fx_tp),      True),
    ]:
        pct = ((rl_v-fx_v)/abs(fx_v)*100) if hi else ((fx_v-rl_v)/abs(fx_v)*100)
        sym = "✓" if pct > 0 else "✗"
        print(f"  {lbl:<22} {rl_v:>10.1f} {fx_v:>10.1f} {sym}{pct:>+10.1f}%")
    print("═"*65)

    # ── plots ─────────────────────────────────────────────────────────
    _plot_comparison(
        {"rewards":rl_rewards,"queues":rl_queues,
         "waiting":rl_waiting,"tp":rl_tp},
        {"rewards":fx_rewards,"queues":fx_queues,
         "waiting":fx_waiting,"tp":fx_tp},
        rl_steps_ep1, fx_steps_ep1,
    )


# ════════════════════════════════════════════════════════════════════════
def _plot_comparison(rl, fx, rl_steps, fx_steps):
    eps = range(1, len(rl["rewards"])+1)
    C   = {"DQN": "#2E75B6", "Fixed": "#ED7D31"}

    # --- 4 metric comparison plots ---
    for key, title, ylabel, hi, fname in [
        ("rewards","Total Reward",       "Reward",     True,  "cmp_reward.png"),
        ("queues", "Avg Queue Length",   "Vehicles",   False, "cmp_queue.png"),
        ("waiting","Avg Waiting Time",   "Seconds",    False, "cmp_waiting.png"),
        ("tp",     "Throughput",         "Vehicles",   True,  "cmp_throughput.png"),
    ]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"DQN vs Fixed-Time: {title}", fontsize=13, fontweight="bold")

        ax1.plot(eps, rl[key], "o-", color=C["DQN"],   lw=2, ms=6, label="DQN")
        ax1.plot(eps, fx[key], "s--",color=C["Fixed"], lw=2, ms=6, label="Fixed")
        ax1.set_xlabel("Episode"); ax1.set_ylabel(ylabel)
        ax1.set_title("Per Episode"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.text(0.02,0.97,"Higher=better↑" if hi else "Lower=better↓",
                 transform=ax1.transAxes, fontsize=8, va="top", color="gray")

        means = [np.mean(rl[key]), np.mean(fx[key])]
        stds  = [np.std(rl[key]),  np.std(fx[key])]
        bars  = ax2.bar(["DQN","Fixed"], means, yerr=stds, capsize=8,
                        color=[C["DQN"],C["Fixed"]], alpha=0.85, edgecolor="white")
        for b, m in zip(bars, means):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+max(stds)*0.05,
                     f"{m:.1f}", ha="center", fontweight="bold", fontsize=11)
        pct = ((means[0]-means[1])/abs(means[1])*100) if hi \
              else ((means[1]-means[0])/abs(means[1])*100)
        ax2.set_title(f"Mean ± SD  (DQN {pct:+.1f}%)", fontweight="bold")
        ax2.set_ylabel(ylabel); ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{RESULTS}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {RESULTS}/{fname}")

    # --- Phase switch timeline plot (PROOF OF REACTIVITY) ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle("Phase Switch Pattern: DQN (traffic-reactive) vs Fixed-Time",
                 fontsize=13, fontweight="bold")

    for ax, steps, label, color in [
        (axes[0], rl_steps, "DQN Agent",          C["DQN"]),
        (axes[1], fx_steps, "Fixed-Time (30s/ph)", C["Fixed"]),
    ]:
        xs     = [d["step"] for d in steps]
        phases = [d["phase"] for d in steps]
        q_ns   = [d["q_N"]+d["q_S"] for d in steps]
        q_ew   = [d["q_E"]+d["q_W"] for d in steps]

        ax.step(xs, phases, where="post", color=color, lw=2, label="Phase (0=NS,1=EW)")
        ax2t = ax.twinx()
        ax2t.plot(xs, q_ns, color="#27AE60", lw=1.2, alpha=0.7, label="NS queue")
        ax2t.plot(xs, q_ew, color="#E74C3C", lw=1.2, alpha=0.7, label="EW queue")
        ax2t.set_ylabel("Queue length", color="gray")

        # Mark every switch
        sw = [d["step"] for d in steps if d["switched"]]
        for s in sw:
            ax.axvline(s, color=color, alpha=0.25, lw=0.8)

        ax.set_yticks([0,1]); ax.set_yticklabels(["NS-green","EW-green"])
        ax.set_ylabel("Phase")
        ax.set_title(f"{label}  (total switches: {len(sw)})", fontweight="bold")
        ax.grid(True, alpha=0.2)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2t.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2, loc="upper right", fontsize=8)

    axes[1].set_xlabel("Simulation Step")
    plt.tight_layout()
    path = f"{RESULTS}/phase_switch_pattern.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Summary dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DQN vs Fixed-Time: Full Evaluation Dashboard",
                 fontsize=15, fontweight="bold")
    for ax,(key,title,ylabel,hi) in zip(axes.flat,[
        ("rewards","Total Reward",     "Reward",  True),
        ("queues", "Avg Queue",        "Vehicles",False),
        ("waiting","Avg Wait Time",    "Seconds", False),
        ("tp",     "Throughput",       "Vehicles",True),
    ]):
        ax.plot(eps, rl[key], "o-",  color=C["DQN"],   lw=2, ms=5, label="DQN")
        ax.plot(eps, fx[key], "s--", color=C["Fixed"],  lw=2, ms=5, label="Fixed")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.text(0.02,0.03,"↑ better" if hi else "↓ better",
                transform=ax.transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS}/dashboard.png")


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    evaluate(parse())
