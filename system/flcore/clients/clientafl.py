"""Client of AFL (He et al., CVPR 2025): a local closed-form (ridge) classifier of the current task, one upload.

Per task a client passes once over its current-task training features X (official CIFAR-100/afl.py,
local_update; paper Eqs. 11-12):

    R_k = (X^T X + gamma I)^-1                  [d, d], sent as Symmetric (its upper triangle)
    W_k = R_k X^T Y   (one-hot Y)               [d, m], the columns of the m classes it holds

The official client also returns C_k = R_k^-1; the server computes it from R_k (the same operation,
moved to the receiver, so nothing else travels). Because only the upper triangle of R_k is sent, the
client mirrors it before computing W_k, so client and server hold the same matrix.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client
from flcore.closedform_utils import class_positions, one_hot, symmetrize_upper
from utils.comm_utils import Symmetric


def afl_local(XtX: np.ndarray, XtY: np.ndarray, gamma: float):
    """(W_k, R_k) of local_update, in float64."""
    R = symmetrize_upper(np.linalg.inv(XtX + float(gamma) * np.eye(XtX.shape[0])))
    return R @ XtY, R


class clientAFL(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None

    def train_round(self, t: int, r: int) -> None:
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        d = int(self.store.dim)
        XtX = np.zeros((d, d), dtype=np.float64)
        XtY = np.zeros((d, len(classes)), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for Z, y in self.store.iter_chunks(rows):
            Z = np.asarray(Z, dtype=np.float64)
            pos = class_positions(y, classes)
            XtX += Z.T @ Z
            XtY += Z.T @ one_hot(pos, len(classes))
            counts += np.bincount(pos, minlength=len(classes))
        keep = counts > 0
        W, R = afl_local(XtX, XtY[:, keep], self.hp["rg"])
        self._payload = {"classes": classes[keep], "W": W, "R": Symmetric(R), "counts": counts[keep]}

    def upload_payload(self, t: int, r: int):
        return self._payload
