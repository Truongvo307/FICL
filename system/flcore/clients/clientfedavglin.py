"""Client of SmokeFedAvgLinear (multi-round smoke test): local SGD on a linear softmax head."""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client


class clientFedAvgLinear(Client):
    def on_begin_task(self, t: int) -> None:
        self.Z, self.y = self.load_task_data()          # current task only (no replay)
        self._upload, self._report = None, {}

    def receive(self, t: int, r, payload: dict) -> None:
        self.classes = np.asarray(payload["classes"], dtype=np.int64)
        self.W = np.array(payload["W"], dtype=np.float64)
        self.b = np.array(payload["b"], dtype=np.float64)

    def train_round(self, t: int, r: int) -> None:
        n = len(self.y)
        if n == 0:
            self._upload, self._report = None, {}
            return
        col = np.full(self.num_classes, -1, dtype=np.int64)
        col[self.classes] = np.arange(len(self.classes))
        target = col[self.y]
        if np.any(target < 0):
            raise RuntimeError(f"client {self.id}: a training label is not among the seen classes")
        rng = self.rng(t, r)
        lr, wd, bs = float(self.learning_rate), float(self.hp["weight_decay"]), int(self.batch_size)
        W, b = self.W, self.b
        loss_sum = 0.0
        for _epoch in range(int(self.local_epochs)):
            perm = rng.permutation(n)
            loss_sum = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                Zb, tb, m = self.Z[idx], target[idx], len(idx)
                logits = Zb @ W + b
                logits -= logits.max(axis=1, keepdims=True)
                expl = np.exp(logits)
                denom = expl.sum(axis=1, keepdims=True)
                loss_sum += float((np.log(denom[:, 0]) - logits[np.arange(m), tb]).sum())
                grad = expl / denom
                grad[np.arange(m), tb] -= 1.0
                W -= lr * (Zb.T @ grad / m + wd * W)
                b -= lr * grad.mean(axis=0)
        self._upload = {"W": W, "b": b, "n": int(n)}
        self._report = {"train_loss": loss_sum / n, "n": int(n)}       # mean loss of the last local epoch

    def upload_payload(self, t: int, r: int):
        return self._upload

    def round_report(self) -> dict:
        return self._report
