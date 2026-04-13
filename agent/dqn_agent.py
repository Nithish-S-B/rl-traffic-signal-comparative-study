"""
dqn_agent.py — Traffic-reactive DQN Agent
==========================================
The agent reads [queue_N, queue_S, queue_E, queue_W] every step and
produces Q-values for {keep, switch}.  Training uses full backpropagation
so the Q-network genuinely learns to associate heavy queues on one side
with "switch" actions.

After training you can verify reactivity:
    python evaluate.py --sim          # prints switch-pattern tables
"""

import os, sys, random, pickle
import numpy as np
from collections import deque

try:
    import torch, torch.nn as nn, torch.optim as optim
    TORCH = True
except ImportError:
    TORCH = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_model import DQNModel


# ════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, cap=20_000):
        self.buf = deque(maxlen=cap)
    def push(self, s, a, r, ns, done):
        self.buf.append((np.array(s,  dtype=np.float32),
                         int(a), float(r),
                         np.array(ns, dtype=np.float32),
                         float(done)))
    def sample(self, n):
        b = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*b)
        return (np.stack(s), np.array(a, np.int64),
                np.array(r, np.float32), np.stack(ns),
                np.array(d, np.float32))
    def __len__(self): return len(self.buf)


# ════════════════════════════════════════════════════════════════════════
class DQNAgent:
    GAMMA       = 0.99
    LR          = 0.001
    BATCH       = 64
    EPS_START   = 1.0
    EPS_END     = 0.02
    EPS_DECAY   = 0.997      # per episode
    TARGET_SYNC = 100        # every N training steps
    MIN_MEM     = 500

    def __init__(self, state_size=4, action_size=2):
        self.state_size  = state_size
        self.action_size = action_size
        self.eps         = self.EPS_START
        self.total_steps = 0
        self.train_steps = 0
        self.losses      = []

        self.online = DQNModel(state_size, action_size)
        self.target = DQNModel(state_size, action_size)
        self._sync()
        self.memory = ReplayBuffer()

        if TORCH:
            self.optim  = optim.Adam(self.online.parameters(), lr=self.LR)
            self.lossfn = nn.MSELoss()
            print("[DQNAgent] PyTorch backend")
        else:
            print("[DQNAgent] NumPy full-backprop backend")

    # ── action ──────────────────────────────────────────────────────────
    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.action_size - 1)
        return self.online.predict(np.array(state, dtype=np.float32))

    # ── memory ──────────────────────────────────────────────────────────
    def remember(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)
        self.total_steps += 1

    # ── learning ────────────────────────────────────────────────────────
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

    def _numpy_step(self, s, a, r, ns, d):
        # Compute TD targets using target network
        q_next = self.target.forward(ns)          # (B, 2)
        td_targets = r + self.GAMMA * q_next.max(axis=1) * (1 - d)
        # Full-backprop update on online network
        return self.online.update_batch(s, a, td_targets)

    def _torch_step(self, s, a, r, ns, d):
        s  = torch.FloatTensor(s)
        ns = torch.FloatTensor(ns)
        a  = torch.LongTensor(a).unsqueeze(1)
        r  = torch.FloatTensor(r)
        d  = torch.FloatTensor(d)
        curr_q  = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q  = self.target(ns).max(1)[0]
            target  = r + self.GAMMA * next_q * (1 - d)
        loss = self.lossfn(curr_q, target)
        self.optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()
        return float(loss.item())

    def decay_eps(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)

    def _sync(self):
        self.target.load_state_dict(self.online.state_dict())

    # ── save / load ──────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ck = {"online": self.online.state_dict(),
              "target": self.target.state_dict(),
              "eps": self.eps, "total_steps": self.total_steps,
              "losses": self.losses}
        if TORCH:
            torch.save(ck, path)
        else:
            with open(path, "wb") as f: pickle.dump(ck, f)
        print(f"  [saved] {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"  [warn] {path} not found"); return
        if TORCH:
            ck = torch.load(path, map_location="cpu")
        else:
            with open(path, "rb") as f: ck = pickle.load(f)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.eps         = ck.get("eps", self.EPS_END)
        self.total_steps = ck.get("total_steps", 0)
        self.losses      = ck.get("losses", [])
        print(f"  [loaded] {path}  eps={self.eps:.4f}  steps={self.total_steps}")

    def stats(self):
        return {"eps": self.eps, "total_steps": self.total_steps,
                "train_steps": self.train_steps,
                "memory": len(self.memory),
                "avg_loss": float(np.mean(self.losses[-100:])) if self.losses else 0.}
