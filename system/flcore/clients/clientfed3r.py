"""Client of Fed3R (Fani et al., ICML 2024): ridge-regression statistics of the current task, one upload.

Per task, a client passes once over its current-task training features z and uploads (official
src/fed3r.py, fed3r_step):

    A_k = sum over its samples of [z; 1][z; 1]^T            [d+1, d+1], sent as Symmetric
    b_k = sum over its samples of [z; 1] y^T  (one-hot y)    [d+1, m], the columns of the m classes it holds

The constant 1 appended to every feature is the bias of the official code. Columns of classes the client
does not hold are zero in the official b_k; they are not sent (the class ids are). With Fed3R-RF
(hyperparameter rf_dim > 0) z is first mapped to rf_dim random Fourier features of an RBF kernel
(Rahimi and Recht 2007), drawn from a seed the server sends once (see serverfed3r.py).
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client
from flcore.closedform_utils import class_positions, one_hot
from utils.comm_utils import Symmetric


def random_fourier_map(seed: int, in_dim: int, rf_dim: int, sigma: float):
    """(omega [in_dim, rf_dim], phase [rf_dim]) of z -> sqrt(2 / D) cos(z omega + phase), the random
    Fourier features of the RBF kernel exp(-||z - z'||^2 / (2 sigma^2)): omega ~ N(0, sigma^-2), phase ~ U[0, 2 pi)."""
    rng = np.random.default_rng(int(seed))
    omega = rng.standard_normal((int(in_dim), int(rf_dim))) / float(sigma)
    phase = rng.uniform(0.0, 2.0 * np.pi, int(rf_dim))
    return omega, phase


def apply_rf(Z: np.ndarray, rf) -> np.ndarray:
    if rf is None:
        return np.asarray(Z, dtype=np.float64)
    omega, phase = rf
    return np.sqrt(2.0 / omega.shape[1]) * np.cos(np.asarray(Z, dtype=np.float64) @ omega + phase)


def with_bias(F: np.ndarray) -> np.ndarray:
    return np.hstack([F, np.ones((F.shape[0], 1), dtype=np.float64)])


class clientFed3R(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None
        if not hasattr(self, "_rf"):
            self._rf = None

    def receive(self, t: int, r, payload: dict) -> None:
        if payload and "rf_seed" in payload:            # Fed3R-RF: the shared random features, once
            self._rf = random_fourier_map(payload["rf_seed"], self.store.dim, self.hp["rf_dim"], self.hp["rf_sigma"])

    def train_round(self, t: int, r: int) -> None:
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        if self.hp["rf_dim"] > 0 and self._rf is None:
            raise RuntimeError(f"client {self.id}: Fed3R-RF needs the random features seed before training")
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        D = (self.hp["rf_dim"] or self.store.dim) + 1
        A = np.zeros((D, D), dtype=np.float64)
        b = np.zeros((D, len(classes)), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for Z, y in self.store.iter_chunks(rows):
            F = with_bias(apply_rf(Z, self._rf))
            pos = class_positions(y, classes)
            A += F.T @ F
            b += F.T @ one_hot(pos, len(classes))
            counts += np.bincount(pos, minlength=len(classes))
        keep = counts > 0
        self._payload = {"classes": classes[keep], "A": Symmetric(A), "b": b[:, keep]}

    def upload_payload(self, t: int, r: int):
        return self._payload

