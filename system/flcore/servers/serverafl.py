"""AFL: Analytic Federated Learning (He, Tong, Fang, Sun, Zeng, Li, Chen and Zhuang, "AFL: A Single-Round
Analytic Approach for Federated Learning with Pre-trained Models", CVPR 2025), on cached features of a frozen
backbone, aggregated over tasks. A REFERENCE baseline.

The method (paper Theorems 1-2, Eqs. 9-15, Algorithm 1; official code github.com/RyougiSh1k1/AFL,
CIFAR-100/afl.py, commit 2a5f7ea: local_update, aggregation, clean_regularization):

    clients   R_k = (X_k^T X_k + gamma I)^-1,  W_k = R_k X_k^T Y_k      (clientafl.py)
    server    C_k = R_k^-1. Absolute Aggregation (AA) law, one upload at a time:
                W <- (I - R C_k + R C_k (C + C_k)^-1 C_k) W + (I - R_k C + R_k C (C + C_k)^-1 C) W_k
                C <- C + C_k,  R <- C^-1
              (the first upload initialises W, C, R; the official aggregation() loop, whose signs differ
               from the printed Eq. 7-10; both give W = (sum C_k)^-1 sum C_k W_k)
              Regularization Intermediary (RI, clean_reg): with K uploads so far,
                W_clean = W + K gamma (C - K gamma I)^-1 W                      (Eq. 15, clean_regularization)
              which equals the least-squares solution (sum X^T X)^-1 sum X^T Y of all data
    predict   argmax_c  z . W_c   (no bias, LinearAnalytic)

Class-incremental use: the AA law adds one upload at a time, so each task's uploads are aggregated into the
running (W, C, R) of the earlier tasks, in client order; K counts every upload so far, and RI is applied
to a copy for evaluation after each task (the running W keeps its regularization, as the official
aggregate-then-clean order). After task t the classifier is AFL over all uploads of tasks 0..t, and with
RI it equals least squares on all training data seen so far, for any client split (tests check this).
The repository's --continual option is a different thing: a one-shot run whose clients hold several class
groups, evaluated once.

If sum X^T X is singular (fewer samples than feature dimensions, e.g. an early task of a fine-grained
dataset), RI has no defined result (the official torch.inverse returns meaningless numbers); the run stops
and says so. clean_reg = false then keeps the ridge with K gamma, as the official code without --clean_reg.

Hyperparameters (official run.sh and the paper's main experiments):
    rg          gamma; 1
    clean_reg   apply RI after aggregation; true
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientafl import clientAFL
from flcore.closedform_utils import symmetrize_upper
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam


class SingularGramError(RuntimeError):
    pass


def aa_step(W, C, R, W_k, C_k, R_k):
    """One step of the official aggregation() loop: (W, C, R) aggregated with one more upload."""
    eye = np.eye(C.shape[0])
    S_inv = np.linalg.inv(C + C_k)
    W_new = (eye - R @ C_k + R @ C_k @ S_inv @ C_k) @ W + (eye - R_k @ C + R_k @ C @ S_inv @ C) @ W_k
    C_new = C + C_k
    return W_new, C_new, np.linalg.inv(C_new)


def clean_regularization(W, C, K: int, gamma: float):
    """clean_regularization(): W + K gamma (C - K gamma I)^-1 W."""
    G = C - K * float(gamma) * np.eye(C.shape[0])
    rank = int(np.linalg.matrix_rank(G, hermitian=True))
    if rank < G.shape[0]:
        raise SingularGramError(
            f"AFL's regularization removal needs the aggregated X^T X to be invertible, but it has rank {rank} < "
            f"d = {G.shape[0]} (fewer samples than feature dimensions?). The official code would invert it anyway "
            f"and return meaningless numbers. Run with --clean_reg false (keeps the ridge K gamma) and record it")
    return W + K * float(gamma) * np.linalg.inv(G) @ W


class AFL(Server):
    """AFL (CVPR 2025): local closed-form classifiers merged by the Absolute Aggregation law, regularization
    removed after aggregation (RI); uploads of later tasks aggregated into the same model (REFERENCE baseline)."""

    REGIME = SINGLE_ROUND
    CLIENT_CLASS = clientAFL
    RUN_FLAGS = ("REFERENCE",)
    HPARAMS = {
        "rg": HParam("float", 1.0, "gamma, the local regularization (official run.sh and paper: 1)", low=1e-300),
        "clean_reg": HParam("bool", True, "remove the regularization after aggregation (RI process; official "
                                          "run.sh uses --clean_reg)"),
    }

    def init_algorithm(self) -> None:
        d = int(self.store.dim)
        self.Wt, self.Ct, self.Rt = None, None, None
        self.uploads = 0
        self.counts = np.zeros(self.num_classes, dtype=np.int64)
        self.d = d
        self.head_classes, self.W = None, None

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        task_classes = {int(c) for c in self.scenario.tasks[t]}
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            if not set(cls.tolist()) <= task_classes or len(set(cls.tolist())) != len(cls):
                raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")
            R_k = symmetrize_upper(p["R"].matrix)
            if R_k.shape != (self.d, self.d) or p["W"].shape != (self.d, len(cls)):
                raise RuntimeError(f"client {cid}: statistics of the wrong shape")
            W_k = np.zeros((self.d, self.num_classes), dtype=np.float64)
            W_k[:, cls] = p["W"]
            C_k = np.linalg.inv(R_k)
            if self.Wt is None:
                self.Wt, self.Ct, self.Rt = W_k, C_k, R_k
            else:
                self.Wt, self.Ct, self.Rt = aa_step(self.Wt, self.Ct, self.Rt, W_k, C_k, R_k)
            self.uploads += 1
            self.counts[cls] += p["counts"]

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if self.counts[c] == 0]
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")
        seen = np.asarray(self.scenario.seen_classes(t), dtype=np.int64)
        W = self.Wt
        if self.hp["clean_reg"]:
            W = clean_regularization(self.Wt, self.Ct, self.uploads, self.hp["rg"])
        self.W = W[:, seen]
        self.head_classes = seen
        self.logger.info(f"    AFL head after task {t}: {len(seen)} classes, {self.uploads} uploads aggregated, "
                         f"gamma {self.hp['rg']:g}, RI {'on' if self.hp['clean_reg'] else 'off'}")

    def final_broadcast_payload(self, t: int):
        return {"classes": self.head_classes, "W": self.W}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        if self.head_classes is None or not np.array_equal(np.sort(self.head_classes), np.sort(seen)):
            raise RuntimeError("the AFL head does not cover exactly the seen classes")
        return self.head_classes[np.argmax(np.asarray(Z, dtype=np.float64) @ self.W, axis=1)]
