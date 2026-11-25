"""Fed3R: Federated Recursive Ridge Regression (Fani, Camoriano, Caputo and Ciccone, "Accelerating
Heterogeneous Federated Learning with Closed-form Classifiers", ICML 2024), on cached features of a frozen
backbone, statistics added over tasks. A REFERENCE baseline.

The method (paper Algorithm 1, Eq. 4; official code github.com/Erosinho13/Fed3R, src/fed3r.py, commit
c2a8e5d: fed3r_step, calc_ridge_params, normalize_W):

    clients   A_k = sum [z;1][z;1]^T,  b_k = sum [z;1] y^T     (clientfed3r.py; z -> RF(z) for Fed3R-RF)
    server    A = lambda I + sum_k A_k,  b = sum_k b_k          (the official code starts A at lambda I,
                                                                  over the d + 1 entries, bias included)
              [W; bias] = A^-1 b                                (torch.linalg.solve)
              W_c <- W_c / ||W_c||,  bias_c <- bias_c / ||W_c|| (normalize_W: the norm of column c of W
                                                                  without the bias row; 0 -> 1e-6)
    predict   argmax_c  z . W_c + bias_c

Class-incremental use: Fed3R is recursive by design (the statistics are sums over samples, and the
paper notes "exact and efficient incremental RR updates"). Each task is one exchange: clients send the
statistics of their current task's data, the server adds them to those of earlier tasks and solves over
the classes seen so far. After task t the classifier is exactly Fed3R on all training data of tasks 0..t,
for any client split (tests check this).

Hyperparameters (all defaults are the official values):
    ridge_lambda   lambda; 0.01 (configs/fed3r_*.yml, and the paper: "we set lambda = 0.01")
    rf_dim         0 = Fed3R; D > 0 = Fed3R-RF with D random features (paper: 5000 and 10000)
    rf_sigma       RBF bandwidth of Fed3R-RF; 1000 (paper, "tuned once in the centralized Landmarks setting")
Fed3R-RF is not in the released code. Its feature map here is the standard random Fourier features of
Rahimi and Recht (2007), which the paper cites: sqrt(2/D) cos(z omega + phase), omega ~ N(0, sigma^-2 I),
phase ~ U[0, 2 pi); the server draws one seed and sends it to every client once (8 bytes), so all clients
use the same map. The bias is appended after the map, as for Fed3R. See docs/BASELINES.md.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientfed3r import apply_rf, clientFed3R, random_fourier_map
from flcore.closedform_utils import symmetrize_upper, wire_seed
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam


def fed3r_solve(A: np.ndarray, b: np.ndarray):
    """calc_ridge_params + normalize_W of the official code: W [D-1, m], bias [m] from A [D, D], b [D, m]."""
    sol = np.linalg.solve(A, b)
    W, bias = sol[:-1, :], sol[-1, :]
    norm = np.linalg.norm(W, axis=0)
    norm = np.where(norm == 0.0, 1e-6, norm)
    return W / norm, bias / norm


class Fed3R(Server):
    """Fed3R (ICML 2024): ridge regression from summed client statistics (features with a bias), class
    columns normalized; Fed3R-RF with rf_dim > 0 (REFERENCE baseline)."""

    REGIME = SINGLE_ROUND
    CLIENT_CLASS = clientFed3R
    RUN_FLAGS = ("REFERENCE",)
    HPARAMS = {
        "ridge_lambda": HParam("float", 0.01, "lambda of the ridge regression, added once at the server "
                                              "(official default 0.01)", low=0.0),
        "rf_dim": HParam("int", 0, "0 = Fed3R; D > 0 = Fed3R-RF with D random Fourier features (paper: 5000, "
                                   "10000; memory grows as D^2)", low=0),
        "rf_sigma": HParam("float", 1000.0, "RBF bandwidth of Fed3R-RF (paper: 1000)", low=1e-12),
    }

    def init_algorithm(self) -> None:
        dim = self.hp["rf_dim"] or int(self.store.dim)
        self.A = float(self.hp["ridge_lambda"]) * np.eye(dim + 1)
        self.b = np.zeros((dim + 1, self.num_classes), dtype=np.float64)
        self.rf, self.rf_seed, self._seed_sent = None, None, set()
        if self.hp["rf_dim"] > 0:
            self.rf_seed = wire_seed(self.rng("fed3r_rf"))
            self.rf = random_fourier_map(self.rf_seed, self.store.dim, self.hp["rf_dim"], self.hp["rf_sigma"])
        self.head_classes, self.W, self.bias = None, None, None

    def extra_run_info(self) -> dict:
        return {"fed3r": {"variant": "Fed3R-RF" if self.rf is not None else "Fed3R",
                          "feature_dim": int(self.A.shape[0] - 1), "rf_seed": self.rf_seed}}

    def download_payload(self, t: int, r: int, client):
        if self.rf is not None and client.id not in self._seed_sent:
            self._seed_sent.add(client.id)
            return {"rf_seed": int(self.rf_seed)}
        return None

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        task_classes = {int(c) for c in self.scenario.tasks[t]}
        D = self.A.shape[0]
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            if not set(cls.tolist()) <= task_classes or len(set(cls.tolist())) != len(cls):
                raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")
            A_k = p["A"].matrix
            if A_k.shape != (D, D) or p["b"].shape != (D, len(cls)):
                raise RuntimeError(f"client {cid}: statistics of the wrong shape")
            self.A += symmetrize_upper(A_k)
            self.b[:, cls] += p["b"]

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if self.b[-1, c] == 0]   # last row of b = class counts
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")
        seen = np.asarray(self.scenario.seen_classes(t), dtype=np.int64)
        self.W, self.bias = fed3r_solve(self.A, self.b[:, seen])
        self.head_classes = seen
        self.logger.info(f"    Fed3R head after task {t}: {len(seen)} classes, {int(self.b[-1].sum())} samples, "
                         f"feature dim {self.A.shape[0] - 1} (+ bias), lambda {self.hp['ridge_lambda']:g}")

    def final_broadcast_payload(self, t: int):
        return {"classes": self.head_classes, "W": self.W, "b": self.bias}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        if self.head_classes is None or not np.array_equal(np.sort(self.head_classes), np.sort(seen)):
            raise RuntimeError("the Fed3R head does not cover exactly the seen classes")
        logits = apply_rf(Z, self.rf) @ self.W + self.bias
        return self.head_classes[np.argmax(logits, axis=1)]
