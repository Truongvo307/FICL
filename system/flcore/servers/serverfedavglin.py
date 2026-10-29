"""SmokeFedAvgLinear: FedAvg on a linear softmax head over cached features, multi-round. A SMOKE TEST.

Not an experimental arm: results carry the SMOKE_TEST flag and must not enter result tables
without the researcher's approval.

Every round the server sends the head columns of the classes seen so far to the selected clients;
each runs local_epochs epochs of minibatch SGD with cross-entropy over the seen classes on its
current-task data only (no replay, so earlier classes are forgotten: expected); the server averages
the returned heads weighted by sample counts. It exercises rounds, client selection, curve points,
model downloads and byte counting of the multi-round path.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientfedavglin import clientFedAvgLinear
from flcore.servers.serverbase import Server
from utils.config_utils import MULTI_ROUND, HParam


class SmokeFedAvgLinear(Server):
    REGIME = MULTI_ROUND
    CLIENT_CLASS = clientFedAvgLinear
    RUN_FLAGS = ("SMOKE_TEST",)
    HPARAMS = {"weight_decay": HParam("float", 0.0, "L2 penalty added to the local SGD gradient", low=0.0)}

    def init_algorithm(self) -> None:
        self.W = np.zeros((self.store.dim, self.num_classes), dtype=np.float64)
        self.b = np.zeros(self.num_classes, dtype=np.float64)

    def _model(self, t: int) -> dict:
        seen = self.scenario.seen_classes(t)
        return {"classes": seen, "W": self.W[:, seen], "b": self.b[seen]}

    def download_payload(self, t: int, r: int, client):
        return self._model(t)

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        total = sum(int(p["n"]) for _cid, p in uploads)
        if total == 0:
            return                                  # nobody had data: keep the model
        seen = self.scenario.seen_classes(t)
        W = np.zeros((self.store.dim, len(seen)))
        b = np.zeros(len(seen))
        for _cid, p in uploads:
            w = int(p["n"]) / total
            W += w * p["W"]
            b += w * p["b"]
        self.W[:, seen], self.b[seen] = W, b

    def final_broadcast_payload(self, t: int):
        return self._model(t)

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        return seen[np.argmax(Z @ self.W[:, seen] + self.b[seen], axis=1)]
