"""FedCGS (Guan, Zhou and Gu, "Capture Global Feature Statistics for One-Shot Federated Learning", AAAI 2025),
adapted to class-incremental FL on cached features of a frozen backbone. A REFERENCE baseline.

The method (paper Algorithm 1, Eqs. 3-11 with the corrected bias of Appendix Eq. 14; official code
github.com/Yuqin-G/FedCGS, main_fedcgs.py and utils_fl.eval, commit e02cc1f):

    clients   class sums A_i^j, class counts N_i^j and the uncentred second moment B_i = sum z z^T
              (clientfedcgs.py), one upload, no training
    server    N^j = sum_i N_i^j,  A^j = sum_i A_i^j,  B = sum_i B_i,  N = sum_j N^j,  A = sum_j A^j
              mu   = A / N                                     global mean (Eq. 8)
              mu^j = A^j / N^j                                 class means (Eq. 6)
              S    = (B - mu A^T - A mu^T + N mu mu^T) / (N - 1)   global covariance (Eq. 7): the TOTAL
                                                                   covariance around the global mean,
                                                                   not a pooled within-class one
              W_j  = S^-1 mu^j  (official: W = means @ inv(S))
              b_j  = -1/2 mu^j^T S^-1 mu^j + log(N^j / N)      Gaussian discriminant with a shared covariance
    predict   argmax_j  z . W_j + b_j

Class-incremental adaptation (the one change of the protocol; reports/FEDCGS.md):
    FedCGS is one-shot: one upload per client, all classes present. Here each task is one exchange
    (single-round regime). Clients upload the statistics of their current task's data only; the server
    adds them to what it received in earlier tasks. Every statistic is a sum over samples and the tasks'
    samples are disjoint, so after task t the head is exactly the official FedCGS fitted on all training
    data seen so far (any client split, any task split; tests check this). Classes of earlier tasks keep
    their sums and counts, the global mean, covariance and priors cover the seen classes, and prediction
    is among the seen classes only. Nothing of a later task is used, and no test data.

Hyperparameter:
    gamma   added to the diagonal of S before inversion. The official script inverts S + gamma I with
            gamma_list = [0], and the paper has no such term, so the default 0 is the official method.
            With few samples per feature dimension (N - 1 < d, e.g. an early task of a fine-grained dataset)
            S is singular; the official np.linalg.inv would then return meaningless numbers, so this
            implementation stops and says so instead. gamma > 0 is then a protocol choice to record (never
            chosen on test data).
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientfedcgs import clientFedCGS
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam


class SingularCovarianceError(RuntimeError):
    pass


def fedcgs_head(class_sums: np.ndarray, class_counts: np.ndarray, gram: np.ndarray, gamma: float = 0.0):
    """The FedCGS classifier from aggregated statistics of m classes.

    class_sums [m, d] (A^j), class_counts [m] (N^j > 0), gram [d, d] (B, summed over the same samples).
    Returns (W [m, d], b [m], info). W and b follow main_fedcgs.py step by step, in float64.

    info also holds the same classifier written around the global mean mu, which predict() uses:
        logit_j(z) = (mu_j - mu)^T S^-1 (z - mu) - 1/2 (mu_j - mu)^T S^-1 (mu_j - mu) + log(N_j / N)
    It differs from z . W_j + b_j by 1/2 mu^T S^-1 mu + mu^T S^-1 (z - mu), the same for every class, so the
    argmax is the same. The numbers are far smaller when the features share a large offset and S is
    ill-conditioned, so rounding cannot reorder classes (keys "mean", "W_centered", "b_centered")."""
    A_j = np.asarray(class_sums, dtype=np.float64)
    N_j = np.asarray(class_counts, dtype=np.float64)
    B = np.asarray(gram, dtype=np.float64)
    m, d = A_j.shape
    if B.shape != (d, d) or N_j.shape != (m,):
        raise ValueError(f"statistics do not fit: sums {A_j.shape}, counts {N_j.shape}, gram {B.shape}")
    if np.any(N_j <= 0):
        raise ValueError("every class of the head needs at least one training sample")
    N = float(N_j.sum())
    if N < 2:
        raise SingularCovarianceError("the covariance needs at least 2 training samples")
    A = A_j.sum(axis=0)                                  # the official Ax: the sum of all features
    mu = A / N
    S = (B - np.outer(mu, A) - np.outer(A, mu) + np.outer(mu, mu) * N) / (N - 1)
    S_hat = S + float(gamma) * np.eye(d)
    rank = int(np.linalg.matrix_rank(S_hat, hermitian=True))
    if rank < d:
        raise SingularCovarianceError(
            f"the FedCGS covariance (+ gamma I, gamma = {gamma:g}) has rank {rank} < d = {d} from N = {int(N)} "
            f"samples, so its inverse is not defined (the official code would invert it anyway and return "
            f"meaningless numbers). Set the hyperparameter gamma > 0 (--gamma); it is a protocol choice: record "
            f"it in reports/CHANGELOG.md and never choose it on test data")
    S_inv = np.linalg.inv(S_hat)
    means = A_j / N_j[:, None]
    W = means @ S_inv
    b = -0.5 * np.einsum("ij,jk,ik->i", means, S_inv, means) + np.log(N_j / N)
    delta = means - mu
    Wc = delta @ S_inv
    bc = -0.5 * np.einsum("ij,jk,ik->i", delta, S_inv, delta) + np.log(N_j / N)
    eig = np.linalg.eigvalsh(S_hat)
    info = {"N": int(N), "d": int(d), "rank": rank, "condition_number": float(eig[-1] / eig[0]),
            "mean": mu, "W_centered": Wc, "b_centered": bc}
    return W, b, info


class FedCGS(Server):
    """FedCGS: one-shot global feature statistics (class means, global covariance, priors) and a Gaussian
    discriminant head; statistics added over tasks (REFERENCE baseline, AAAI 2025)."""

    REGIME = SINGLE_ROUND
    CLIENT_CLASS = clientFedCGS
    RUN_FLAGS = ("REFERENCE",)
    HPARAMS = {
        "gamma": HParam("float", 0.0, "added to the diagonal of the global covariance before inversion; 0 is the "
                                      "official FedCGS (gamma_list = [0]); > 0 only when the covariance is "
                                      "singular, as a recorded protocol choice", low=0.0),
    }

    def init_algorithm(self) -> None:
        d = int(self.store.dim)
        self.sums = np.zeros((self.num_classes, d), dtype=np.float64)
        self.counts = np.zeros(self.num_classes, dtype=np.int64)
        self.gram = np.zeros((d, d), dtype=np.float64)
        self.head_classes, self.W, self.b, self.head_info = None, None, None, None

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        task_classes = {int(c) for c in self.scenario.tasks[t]}
        d = self.gram.shape[0]
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            if not set(cls.tolist()) <= task_classes or len(set(cls.tolist())) != len(cls):
                raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")
            gram = p["gram"].matrix
            if gram.shape != (d, d) or p["sums"].shape != (len(cls), d):
                raise RuntimeError(f"client {cid}: statistics of the wrong shape")
            self.sums[cls] += p["sums"]
            self.counts[cls] += np.asarray(p["counts"], dtype=np.int64)
            self.gram += gram

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if self.counts[c] == 0]
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")
        seen = np.asarray(self.scenario.seen_classes(t), dtype=np.int64)
        self.W, self.b, info = fedcgs_head(self.sums[seen], self.counts[seen], self.gram, self.hp["gamma"])
        self.head_classes, self.head_info = seen, info
        self.mean, self.Wc, self.bc = info["mean"], info["W_centered"], info["b_centered"]
        self.logger.info(f"    FedCGS head after task {t}: {len(seen)} classes from N = {info['N']} samples, "
                         f"d = {info['d']}, covariance rank {info['rank']}, condition number "
                         f"{info['condition_number']:.3g}")

    def final_broadcast_payload(self, t: int):
        return {"classes": self.head_classes, "mean": self.mean, "W": self.Wc, "b": self.bc}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        if self.head_classes is None or not np.array_equal(np.sort(self.head_classes), np.sort(seen)):
            raise RuntimeError("the FedCGS head does not cover exactly the seen classes")
        logits = (np.asarray(Z, dtype=np.float64) - self.mean) @ self.Wc.T + self.bc   # see fedcgs_head
        return self.head_classes[np.argmax(logits, axis=1)]
