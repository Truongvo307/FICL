"""Client of SmokeClassMean (single-round smoke test): per-class feature sums of the current task."""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client


class clientClassMean(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None

    def train_round(self, t: int, r: int) -> None:
        """One pass over the current task's training rows: class sums (float64) and counts."""
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        classes = self.task_classes()
        sums = np.zeros((len(classes), self.store.dim), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for Z, y in self.store.iter_chunks(rows):
            for j, c in enumerate(classes):
                mask = y == c
                if mask.any():
                    sums[j] += Z[mask].sum(axis=0)
                    counts[j] += int(mask.sum())
        keep = counts > 0
        self._payload = {"classes": classes[keep].astype(np.int64), "sums": sums[keep], "counts": counts[keep]}

    def upload_payload(self, t: int, r: int):
        return self._payload
