"""Client of FedCGS (Guan et al., AAAI 2025): global feature statistics of the current task, one upload.

Once per task a client passes over its current-task training features and uploads (ClientStats of the
paper's Algorithm 1; Eqs. 3-5):

    classes   the classes it holds in this task (classes with no sample are left out)
    sums      A_i^j = sum of the features of class j                     [m, d] float
    counts    N_i^j = number of samples of class j                       [m] int
    gram      B_i = sum over its samples of z z^T (uncentred)            [d, d] float, sent as Symmetric

All in float64, as the official code (it casts the features to double before summing). The official
script also forms the client's total feature sum; it is the sum of the class sums, so it is not sent
(the paper's communication count, d^2 + C d + C for ResNet-18 on CIFAR-10, leaves it out too).
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client
from utils.comm_utils import Symmetric


def client_statistics(Z: np.ndarray, y: np.ndarray, classes: np.ndarray, gram: np.ndarray | None = None,
                      sums: np.ndarray | None = None, counts: np.ndarray | None = None):
    """Add one chunk (features Z [b, d], labels y [b]) to (sums [len(classes), d], counts, gram [d, d]).

    Starts from zeros when the accumulators are None; returns the three accumulators (float64, int64)."""
    classes = np.asarray(classes, dtype=np.int64)
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    if sums is None:
        sums = np.zeros((len(classes), d), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        gram = np.zeros((d, d), dtype=np.float64)
    pos = np.searchsorted(classes, y)
    if np.any(pos >= len(classes)) or not np.array_equal(classes[np.minimum(pos, len(classes) - 1)], y):
        raise RuntimeError("a training label is not among the current task's classes")
    onehot = np.zeros((len(y), len(classes)), dtype=np.float64)
    onehot[np.arange(len(y)), pos] = 1.0
    sums += onehot.T @ Z                                 # class sums, one matrix product per chunk
    counts += np.bincount(pos, minlength=len(classes)).astype(np.int64)
    gram += Z.T @ Z
    return sums, counts, gram


class clientFedCGS(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None

    def train_round(self, t: int, r: int) -> None:
        """One pass over the current task's training rows (no training: FedCGS is closed-form)."""
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None                       # no data this task: nothing to send
            return
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        sums = counts = gram = None
        for Z, y in self.store.iter_chunks(rows):
            sums, counts, gram = client_statistics(Z, y, classes, gram, sums, counts)
        keep = counts > 0
        self._payload = {"classes": classes[keep], "sums": sums[keep], "counts": counts[keep],
                         "gram": Symmetric(gram)}

    def upload_payload(self, t: int, r: int):
        return self._payload
