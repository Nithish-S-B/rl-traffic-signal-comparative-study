"""
q_agent.py — Tabular Q-Learning Agent for Traffic Signal Control
================================================================
Discretizes the continuous state [queue_N, queue_S, queue_E, queue_W]
into buckets and maintains a Q-table mapping (state_bucket, action) → value.

Action space: 2  (0 = keep phase, 1 = switch phase)
State space : 4 queue lengths, each bucketed into N_BINS bins
"""

import os
import pickle
import random
import numpy as np


class QLearningAgent:
    # ── Hyperparameters ─────────────────────────────────────────────────
    ALPHA       = 0.1        # learning rate
    GAMMA       = 0.99       # discount factor
    EPS_START   = 1.0        # initial exploration
    EPS_END     = 0.02       # minimum exploration
    EPS_DECAY   = 0.997      # per-episode decay (matches DQN for fair comparison)

    # Discretisation: queue lengths bucketed into N_BINS intervals
    N_BINS      = 6          # 0–4, 5–9, 10–14, 15–19, 20–24, 25+
    MAX_QUEUE   = 30         # clamp before bucketing
    BIN_EDGES   = [0, 5, 10, 15, 20, 25, 30]  # N_BINS+1 edges

    def __init__(self, state_size: int = 4, action_size: int = 2):
        self.state_size  = state_size
        self.action_size = action_size
        self.eps         = self.EPS_START
        self.total_steps = 0

        # Q-table: dict keyed by (bucket_0, …, bucket_{state_size-1})
        # Values: list of Q-values per action
        self.q_table: dict = {}
        self.losses: list  = []   # tracks |TD error| for consistency with DQN API

        print(f"[QLearningAgent] Tabular Q-learning | "
              f"bins={self.N_BINS} | α={self.ALPHA} | γ={self.GAMMA}")

    # ── State discretisation ─────────────────────────────────────────────
    def _discretize(self, state: np.ndarray) -> tuple:
        """Map continuous queue lengths → discrete bucket indices."""
        buckets = []
        for q in state:
            q = float(np.clip(q, 0, self.MAX_QUEUE))
            b = int(np.digitize(q, self.BIN_EDGES[1:])) # 0-indexed bucket
            buckets.append(min(b, self.N_BINS - 1))
        return tuple(buckets)

    def _get_q(self, key: tuple) -> np.ndarray:
        """Return Q-values for a state key, initialising to zeros if new."""
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size, dtype=np.float64)
        return self.q_table[key]

    # ── Action selection (ε-greedy) ──────────────────────────────────────
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.action_size - 1)
        key = self._discretize(state)
        return int(np.argmax(self._get_q(key)))

    # ── Q-table update (online, single-step TD) ──────────────────────────
    def learn_step(self, s: np.ndarray, a: int, r: float,
                   ns: np.ndarray, done: bool) -> float:
        """
        Standard Q-learning update:
            Q(s,a) ← Q(s,a) + α * [r + γ·max_a' Q(s',a') − Q(s,a)]
        """
        s_key  = self._discretize(s)
        ns_key = self._discretize(ns)

        q_sa   = self._get_q(s_key)[a]
        q_ns   = 0.0 if done else np.max(self._get_q(ns_key))
        target = r + self.GAMMA * q_ns
        td_err = target - q_sa

        self._get_q(s_key)[a] += self.ALPHA * td_err
        self.total_steps += 1

        abs_err = abs(float(td_err))
        self.losses.append(abs_err)
        return abs_err

    # ── Compatibility shims (mirrors DQN/DDQN API) ───────────────────────
    def remember(self, s, a, r, ns, done):
        """Q-Learning is online; we learn immediately in train.py."""
        pass

    def learn(self):
        """Not used for tabular Q-Learning; learning happens in learn_step."""
        return None

    def decay_eps(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "eps":     self.eps,
            "total_steps": self.total_steps,
            "losses":  self.losses,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [saved] {path}  states={len(self.q_table)}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"  [warn] {path} not found")
            return
        with open(path, "rb") as f:
            ck = pickle.load(f)
        self.q_table     = ck.get("q_table", {})
        self.eps         = ck.get("eps", self.EPS_END)
        self.total_steps = ck.get("total_steps", 0)
        self.losses      = ck.get("losses", [])
        print(f"  [loaded] {path}  states={len(self.q_table)}  "
              f"eps={self.eps:.4f}  steps={self.total_steps}")

    def stats(self) -> dict:
        return {
            "eps":         self.eps,
            "total_steps": self.total_steps,
            "train_steps": self.total_steps,
            "memory":      len(self.q_table),   # number of unique states visited
            "avg_loss":    float(np.mean(self.losses[-100:])) if self.losses else 0.0,
        }
