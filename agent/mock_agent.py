"""
Lightweight mock of the PyTorch-specific components so that
the project can run (and produce real plots) even without
a PyTorch installation. Used only when torch is not importable.

This module is NOT part of the public API; it's a testing shim.
"""

import numpy as np
import random
from collections import deque
import os


class MockTensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def item(self):
        return float(self.data.flat[0])

    def argmax(self):
        return MockTensor(np.argmax(self.data))

    def numpy(self):
        return self.data


class MockDQNModel:
    """Lightweight numpy-based Q-network for testing."""

    def __init__(self, state_size=4, action_size=2, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        scale = 0.1
        self.W1 = np.random.randn(state_size, hidden_size) * scale
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, action_size) * scale
        self.b3 = np.zeros(action_size)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def get_action_values(self, state_tensor):
        if isinstance(state_tensor, MockTensor):
            x = state_tensor.data
        else:
            x = state_tensor
        q = self.forward(x)
        return MockTensor(q)

    def __call__(self, x):
        if isinstance(x, MockTensor):
            x = x.data
        return self.forward(x)

    def state_dict(self):
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3}

    def load_state_dict(self, d):
        self.W1 = d["W1"]; self.b1 = d["b1"]
        self.W2 = d["W2"]; self.b2 = d["b2"]
        self.W3 = d["W3"]; self.b3 = d["b3"]

    def eval(self):
        return self

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def count_parameters(self):
        return sum(p.size for p in self.parameters())

    def __repr__(self):
        return f"MockDQNModel(4->64->64->2, params={self.count_parameters()})"


class MockReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class MockDQNAgent:
    """Numpy-based DQN agent for testing without PyTorch."""

    GAMMA = 0.99
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 50
    REPLAY_BUFFER_SIZE = 10_000
    MIN_REPLAY_SIZE = 200

    def __init__(self, state_size=4, action_size=2, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = "cpu"
        self.policy_net = MockDQNModel(state_size, action_size)
        self.target_net = MockDQNModel(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_buffer = MockReplayBuffer(self.REPLAY_BUFFER_SIZE)
        self.epsilon = self.EPSILON_START
        self.total_steps = 0
        self.training_steps = 0
        self.losses = []
        print("[MockDQNAgent] Using numpy backend (install PyTorch for full performance)")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.select_action_greedy(state)

    def select_action_greedy(self, state):
        q = self.policy_net.forward(np.array(state, dtype=np.float32).reshape(1, -1))
        return int(np.argmax(q))

    def remember(self, s, a, r, ns, d):
        self.replay_buffer.push(s, a, r, ns, d)
        self.total_steps += 1

    def train(self):
        if len(self.replay_buffer) < self.MIN_REPLAY_SIZE:
            return None

        s, a, r, ns, d = self.replay_buffer.sample(self.BATCH_SIZE)

        # Current Q-values
        curr_q = np.array([self.policy_net.forward(s[i:i+1])[0, a[i]]
                           for i in range(self.BATCH_SIZE)])

        # Target Q-values
        next_q = np.array([self.target_net.forward(ns[i:i+1]).max()
                           for i in range(self.BATCH_SIZE)])
        target_q = r + self.GAMMA * next_q * (1 - d)

        # Simple gradient update (SGD-like)
        loss = float(np.mean((curr_q - target_q) ** 2))
        self._update_weights(s, a, target_q)

        self.training_steps += 1
        self.losses.append(loss)

        if self.training_steps % self.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    def _update_weights(self, states, actions, targets):
        """Mini gradient update for the mock network."""
        lr = self.LEARNING_RATE
        for i in range(len(states)):
            x = states[i:i+1]
            a = actions[i]
            tgt = targets[i]

            # Forward
            h1 = np.maximum(0, x @ self.policy_net.W1 + self.policy_net.b1)
            h2 = np.maximum(0, h1 @ self.policy_net.W2 + self.policy_net.b2)
            q = h2 @ self.policy_net.W3 + self.policy_net.b3
            err = q[0, a] - tgt

            # Backward (output layer only for speed)
            dW3 = h2.T * err
            db3 = np.zeros(self.action_size)
            db3[a] = err

            self.policy_net.W3[:, a] -= lr * dW3.flatten()
            self.policy_net.b3[a] -= lr * err

    def decay_epsilon(self):
        self.epsilon = max(self.EPSILON_END, self.epsilon * self.EPSILON_DECAY)

    def save(self, path):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "losses": self.losses,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[MockDQNAgent] Model saved to: {path}")

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.policy_net.load_state_dict(data["policy_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.epsilon = data.get("epsilon", self.EPSILON_END)
        self.total_steps = data.get("total_steps", 0)
        self.training_steps = data.get("training_steps", 0)
        self.losses = data.get("losses", [])
        print(f"[MockDQNAgent] Model loaded from: {path}")

    def get_stats(self):
        return {
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "replay_buffer_size": len(self.replay_buffer),
            "avg_loss_recent": np.mean(self.losses[-100:]) if self.losses else 0.0,
        }

    def __repr__(self):
        return f"MockDQNAgent(4->64->64->2, ε={self.epsilon:.4f})"
