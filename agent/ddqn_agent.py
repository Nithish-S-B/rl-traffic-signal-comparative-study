"""
ddqn_agent.py — Double Deep Q-Network Agent for Traffic Signal Control
=======================================================================
Key difference from DQN:
  • Action selection  → online network  (picks best action)
  • Value estimation  → target network  (evaluates that action)

This decoupling eliminates the maximisation bias that causes standard
DQN to over-estimate Q-values, leading to more stable training and
better final performance.

Architecture: 4 → 64 → 64 → 2  (same as DQN for fair comparison)
"""

import os
import sys
import random
import pickle
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH = True
except ImportError:
    TORCH = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_model import DQNModel


# ════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, cap: int = 20_000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, done):
        self.buf.append((
            np.array(s,  dtype=np.float32),
            int(a), float(r),
            np.array(ns, dtype=np.float32),
            float(done),
        ))

    def sample(self, n: int):
        b = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*b)
        return (np.stack(s), np.array(a, np.int64),
                np.array(r, np.float32), np.stack(ns),
                np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)


# ════════════════════════════════════════════════════════════════════════
class DDQNAgent:
    """
    Double DQN Agent.

    The only algorithmic change versus DQN is in target computation:

    DQN    : target = r + γ · max_a'  Q_target(s', a')
    DDQN   : a*     = argmax_a'  Q_online(s', a')       ← online picks action
             target = r + γ · Q_target(s', a*)          ← target evaluates it
    """

    GAMMA       = 0.99
    LR          = 0.001
    BATCH       = 64
    EPS_START   = 1.0
    EPS_END     = 0.02
    EPS_DECAY   = 0.997      # per episode (matches DQN / Q-learning)
    TARGET_SYNC = 100        # every N training steps
    MIN_MEM     = 500

    def __init__(self, state_size: int = 4, action_size: int = 2):
        self.state_size  = state_size
        self.action_size = action_size
        self.eps         = self.EPS_START
        self.total_steps = 0
        self.train_steps = 0
        self.losses: list = []

        self.online = DQNModel(state_size, action_size)
        self.target = DQNModel(state_size, action_size)
        self._sync()
        self.memory = ReplayBuffer()

        if TORCH:
            self.optim  = optim.Adam(self.online.parameters(), lr=self.LR)
            self.lossfn = nn.MSELoss()
            print("[DDQNAgent] PyTorch backend — Double DQN")
        else:
            print("[DDQNAgent] NumPy backend — Double DQN")

    # ── Action (ε-greedy via online network) ────────────────────────────
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.action_size - 1)
        return self.online.predict(np.array(state, dtype=np.float32))

    # ── Experience storage ────────────────────────────────────────────
    def remember(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)
        self.total_steps += 1

    # ── Learning (Double DQN update) ──────────────────────────────────
    def learn(self):
        if len(self.memory) < self.MIN_MEM:
            return None

        s, a, r, ns, d = self.memory.sample(self.BATCH)

        if TORCH:
            loss = self._torch_step(s, a, r, ns, d)
        else:
            loss = self._numpy_step(s, a, r, ns, d)

        self.losses.append(loss)
        self.train_steps += 1
        if self.train_steps % self.TARGET_SYNC == 0:
            self._sync()
        return loss

    def _torch_step(self, s, a, r, ns, d) -> float:
        s_t  = torch.FloatTensor(s)
        ns_t = torch.FloatTensor(ns)
        a_t  = torch.LongTensor(a).unsqueeze(1)
        r_t  = torch.FloatTensor(r)
        d_t  = torch.FloatTensor(d)

        # Current Q-values for chosen actions
        curr_q = self.online(s_t).gather(1, a_t).squeeze(1)

        with torch.no_grad():
            # DDQN: online selects best action in next state
            best_a  = self.online(ns_t).argmax(1, keepdim=True)   # ← KEY CHANGE
            # Target evaluates that action
            next_q  = self.target(ns_t).gather(1, best_a).squeeze(1)
            target  = r_t + self.GAMMA * next_q * (1.0 - d_t)

        loss = self.lossfn(curr_q, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()
        return float(loss.item())

    def _numpy_step(self, s, a, r, ns, d) -> float:
        # DDQN: online selects action, target evaluates it
        q_online_ns = self.online.forward(ns)       # (B, 2)
        best_a      = np.argmax(q_online_ns, axis=1) # (B,)  ← online picks action
        q_target_ns = self.target.forward(ns)        # (B, 2)
        q_best      = q_target_ns[np.arange(len(ns)), best_a]  # target evaluates

        td_targets  = r + self.GAMMA * q_best * (1.0 - d)
        return self.online.update_batch(s, a, td_targets)

    def decay_eps(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)

    def _sync(self):
        self.target.load_state_dict(self.online.state_dict())

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ck = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "eps":    self.eps,
            "total_steps": self.total_steps,
            "losses": self.losses,
        }
        if TORCH:
            torch.save(ck, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(ck, f)
        print(f"  [saved] {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"  [warn] {path} not found")
            return
        if TORCH:
            ck = torch.load(path, map_location="cpu")
        else:
            with open(path, "rb") as f:
                ck = pickle.load(f)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.eps         = ck.get("eps", self.EPS_END)
        self.total_steps = ck.get("total_steps", 0)
        self.losses      = ck.get("losses", [])
        print(f"  [loaded] {path}  eps={self.eps:.4f}  steps={self.total_steps}")

    def stats(self) -> dict:
        return {
            "eps":         self.eps,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "memory":      len(self.memory),
            "avg_loss":    float(np.mean(self.losses[-100:])) if self.losses else 0.0,
        }
