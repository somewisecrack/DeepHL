from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np
from .models import Transition


class ReplayStore:
    def __init__(self, path: Path, capacity: int = 200_000):
        self.path = path
        self.capacity = capacity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: list[Transition] = []
        self.load()

    def load(self):
        if not self.path.exists(): return
        for line in self.path.read_text().splitlines()[-self.capacity:]:
            try: self.data.append(Transition(**json.loads(line)))
            except Exception: pass

    def add(self, t: Transition):
        self.data.append(t)
        if len(self.data) > self.capacity: self.data = self.data[-self.capacity:]
        with self.path.open("a") as f: f.write(json.dumps(t.__dict__) + "\n")

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.data, min(n, len(self.data)))

    def compact(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t.__dict__) + "\n" for t in self.data[-self.capacity:]))
        tmp.replace(self.path)


class DuelingDoubleDQN:
    """Numpy fallback: Double-Q linear model with the same API as a future PyTorch Dueling DQN."""
    def __init__(self, input_dim: int, actions: int = 5, ckpt: Path | None = None):
        self.input_dim, self.actions = input_dim, actions
        self.rng = np.random.default_rng(7)
        self.w_a = self.rng.normal(0, 0.001, (actions, input_dim + 1))
        self.w_b = self.rng.normal(0, 0.001, (actions, input_dim + 1))
        self.epsilon = 0.25
        self.updates = 0
        self.gamma = 0.997
        self.lr = 0.00035
        self.ckpt = ckpt
        if ckpt and ckpt.exists(): self.load()

    def q_values(self, state: list[float]) -> np.ndarray:
        x = np.r_[1.0, np.asarray(state, dtype=float)]
        return ((self.w_a @ x) + (self.w_b @ x)) / 2

    def select(self, state: list[float], mask: list[bool]) -> int:
        valid = [i for i, ok in enumerate(mask) if ok]
        if not valid: return 0
        if random.random() < self.epsilon: return random.choice(valid)
        q = self.q_values(state)
        return max(valid, key=lambda i: q[i])

    def train(self, batch: list[Transition]):
        for t in batch:
            x = np.r_[1.0, np.asarray(t.state, dtype=float)]
            nx = np.r_[1.0, np.asarray(t.next_state, dtype=float)]
            use_a = random.random() < 0.5
            online = self.w_a if use_a else self.w_b
            target_net = self.w_b if use_a else self.w_a
            valid = [i for i, ok in enumerate(t.next_mask) if ok]
            if t.done or not valid:
                bootstrap = 0.0
            else:
                nq = online @ nx
                na = max(valid, key=lambda i: nq[i])
                bootstrap = float(target_net[na] @ nx)
            target = t.reward + self.gamma * bootstrap
            pred = float(online[t.action] @ x)
            err = float(np.clip(target - pred, -10, 10))
            online[t.action] += self.lr * err * x
            self.updates += 1
        self.epsilon = max(0.03, self.epsilon * 0.9995)

    def save(self):
        if not self.ckpt: return
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt.with_suffix(".tmp")
        np.savez(tmp, w_a=self.w_a, w_b=self.w_b, epsilon=self.epsilon, updates=self.updates)
        tmp_npz = Path(str(tmp) + ".npz")
        tmp_npz.replace(self.ckpt)

    def load(self):
        z = np.load(self.ckpt)
        self.w_a = z["w_a"]
        self.w_b = z["w_b"]
        self.epsilon = float(z["epsilon"])
        self.updates = int(z["updates"])
