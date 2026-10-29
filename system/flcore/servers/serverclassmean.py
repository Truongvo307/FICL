"""SmokeClassMean: nearest class mean, single-round (closed-form). A SMOKE TEST of the loop.

Not an experimental arm: results carry the SMOKE_TEST flag and must not enter result tables
without the researcher's approval.

Once per task every client uploads the per-class feature sums and counts of its current-task data;
the server adds them up and predicts the nearest class mean among the classes seen so far. Sums
are additive, so the federated means equal the centralized ones for any client split and the
counts of earlier classes never change: a loop-level check of task gating and exact aggregation.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientclassmean import clientClassMean
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND


class SmokeClassMean(Server):
    REGIME = SINGLE_ROUND
    CLIENT_CLASS = clientClassMean
    RUN_FLAGS = ("SMOKE_TEST",)
    HPARAMS = {}

    def init_algorithm(self) -> None:
        self.sums = np.zeros((self.num_classes, self.store.dim), dtype=np.float64)
        self.counts = np.zeros(self.num_classes, dtype=np.int64)

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        task_classes = set(self.scenario.tasks[t].tolist())
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            if not set(cls.tolist()) <= task_classes or len(set(cls.tolist())) != len(cls):
                raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")
            self.sums[cls] += p["sums"]
            self.counts[cls] += p["counts"]

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if self.counts[c] == 0]
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")

    def class_means(self, classes: np.ndarray) -> np.ndarray:
        classes = np.asarray(classes, dtype=np.int64)
        return self.sums[classes] / self.counts[classes, None]

    def final_broadcast_payload(self, t: int):
        seen = self.scenario.seen_classes(t)
        return {"classes": seen, "means": self.class_means(seen)}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        mu = self.class_means(seen)
        d2 = (Z * Z).sum(axis=1, keepdims=True) - 2.0 * Z @ mu.T + (mu * mu).sum(axis=1)
        return seen[np.argmin(d2, axis=1)]
