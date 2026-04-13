"""
dqn_model.py — 4 → 64 → 64 → 2  Deep Q-Network
Uses PyTorch when available; falls back to a fully-correct numpy
implementation with proper backpropagation through all layers.
"""
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False


# ════════════════════════════════════════════════════════════════════════
# PyTorch model
# ════════════════════════════════════════════════════════════════════════
if TORCH:
    class DQNModel(nn.Module):
        def __init__(self, s=4, a=2, h=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(s, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, a),
            )
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x): return self.net(x)

        def predict(self, s: np.ndarray) -> int:
            with torch.no_grad():
                return int(self.forward(torch.FloatTensor(s).unsqueeze(0)).argmax())


# ════════════════════════════════════════════════════════════════════════
# NumPy model with FULL backpropagation through all three layers
# ════════════════════════════════════════════════════════════════════════
else:
    class DQNModel:                              # type: ignore[no-redef]
        """
        Fully-differentiable 4→64→64→2 network.

        Training is done inside this class via `update_batch` which
        runs one step of mini-batch SGD with proper chain-rule gradients
        through all layers.
        """

        def __init__(self, s=4, a=2, h=64, lr=0.001):
            self.s, self.a, self.h, self.lr = s, a, h, lr
            # Xavier initialisation
            def xavier(fan_in, fan_out):
                lim = np.sqrt(6.0 / (fan_in + fan_out))
                return np.random.uniform(-lim, lim, (fan_in, fan_out))

            self.W1 = xavier(s, h);  self.b1 = np.zeros(h)
            self.W2 = xavier(h, h);  self.b2 = np.zeros(h)
            self.W3 = xavier(h, a);  self.b3 = np.zeros(a)

        # ── forward ──────────────────────────────────────────────────
        def forward(self, x: np.ndarray) -> np.ndarray:
            """x: (B, 4) → (B, 2)"""
            self._x  = x
            self._h1 = np.maximum(0, x  @ self.W1 + self.b1)   # (B, h)
            self._h2 = np.maximum(0, self._h1 @ self.W2 + self.b2)  # (B, h)
            return self._h2 @ self.W3 + self.b3                 # (B, 2)

        def predict(self, s: np.ndarray) -> int:
            return int(np.argmax(self.forward(s.reshape(1, -1))))

        # ── mini-batch update  (full backprop) ───────────────────────
        def update_batch(self, states, actions, td_targets,
                         gamma=0.99) -> float:
            """
            Compute TD loss and update ALL weights via backprop.

            Parameters
            ----------
            states     : (B, 4)
            actions    : (B,)  int – chosen action indices
            td_targets : (B,)  float – r + γ * max Q(s', ·)
            """
            B = len(states)

            # ── forward ──
            h1   = np.maximum(0, states @ self.W1 + self.b1)   # (B, h)
            h2   = np.maximum(0, h1     @ self.W2 + self.b2)   # (B, h)
            q    = h2 @ self.W3 + self.b3                       # (B, 2)

            # ── TD errors for chosen actions ──
            idx         = np.arange(B)
            td_errors   = q[idx, actions] - td_targets           # (B,)
            loss        = float(np.mean(td_errors ** 2))

            # ── output-layer gradient ──
            dQ          = np.zeros_like(q)                        # (B, 2)
            dQ[idx, actions] = 2.0 * td_errors / B

            dW3  = h2.T @ dQ                                      # (h, 2)
            db3  = dQ.sum(axis=0)                                 # (2,)

            # ── backprop through h2  (ReLU) ──
            dh2  = dQ @ self.W3.T                                 # (B, h)
            dh2 *= (h2 > 0).astype(float)                        # ReLU mask

            dW2  = h1.T @ dh2                                     # (h, h)
            db2  = dh2.sum(axis=0)                                # (h,)

            # ── backprop through h1  (ReLU) ──
            dh1  = dh2 @ self.W2.T                                # (B, h)
            dh1 *= (h1 > 0).astype(float)

            dW1  = states.T @ dh1                                 # (4, h)
            db1  = dh1.sum(axis=0)                                # (h,)

            # ── gradient clipping (global norm) ──
            grads = [dW1, db1, dW2, db2, dW3, db3]
            gnorm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if gnorm > 1.0:
                scale = 1.0 / gnorm
                grads = [g * scale for g in grads]
            dW1, db1, dW2, db2, dW3, db3 = grads

            # ── SGD update ──
            self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3

            return loss

        # ── state dict (for copying to target network) ───────────────
        def state_dict(self):
            return (self.W1.copy(), self.b1.copy(),
                    self.W2.copy(), self.b2.copy(),
                    self.W3.copy(), self.b3.copy())

        def load_state_dict(self, d):
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
                [x.copy() for x in d]
