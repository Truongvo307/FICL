"""FedCOF: Federated learning with COvariances for Free (Goswami, Magistri, Wang, Twardowski, Bagdanov and
van de Weijer, "Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning",
NeurIPS 2025), on cached features of a frozen backbone, statistics added over tasks. A REFERENCE baseline.

The method (paper Eqs. 1, 5-8; official code github.com/dipamgoswami/FedCOF, federate_classes.py,
FedCOFServer.aggregate and get_classifier_weights, commit 960014f):

    clients   class means mu_{k,c} and counts n_{k,c}                      (clientfedcof.py)
    server    N_c = sum_k n_{k,c},  mu_c = sum_k n_{k,c} mu_{k,c} / N_c       (Eq. 1)
              Sigma_c = 1/(K_c - 1) sum_k n_{k,c} (mu_{k,c} - mu_c)(mu_{k,c} - mu_c)^T + gamma I
                                                                            (Eqs. 5-6; K_c = number of means
                                                                             of class c; the code divides by
                                                                             K_c when K_c = 1)
              G = sum_c (N_c - 1) Sigma_c + N mu_g mu_g^T,  mu_g = sum_c N_c mu_c / N
                                                                            (Eq. 8: within-class covariances
                                                                             only, between-class scatter removed)
              W_c = pinv(G) (N_c mu_c),  then W_c <- W_c / ||W_c||       (torch.linalg.pinv, F.normalize)
    predict   argmax_c  z . W_c                                             (no bias, as the official head)

Class-incremental use: each task is one exchange; the server keeps every (mean, count) it received, per
class. Tasks hold disjoint classes, so after task t the classifier is exactly FedCOF over all means sent
for the classes seen so far (any client split gives the same mean set structure; tests check the
estimator against a direct transcription). N and mu_g are over the seen classes.

Hyperparameters:
    gamma_shrink       gamma; 1.0 for ViT-B/16 and SqueezeNet, 0.1 for MobileNetV2 (paper Appendix L,
                       bash_for_testmethods/*.sh)
    means_per_client   M means per class per client; 1 is the released code (paper Appendix E for M > 1)
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientfedcof import clientFedCOF
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam


def fedcof_head(means_by_class: list, counts_by_class: list, gamma: float):
    """FedCOFServer.aggregate + get_classifier_weights for m classes, in float64.

    means_by_class[j] [K_j, d], counts_by_class[j] [K_j]. Returns (W [m, d] with unit rows, info)."""
    d = means_by_class[0].shape[1]
    G = np.zeros((d, d), dtype=np.float64)
    B = np.zeros((len(means_by_class), d), dtype=np.float64)
    N, g_sum = 0.0, np.zeros(d, dtype=np.float64)
    for j, (Mc, nc) in enumerate(zip(means_by_class, counts_by_class, strict=True)):
        Mc = np.asarray(Mc, dtype=np.float64)
        nc = np.asarray(nc, dtype=np.float64)
        Nc = float(nc.sum())
        mu = (nc[:, None] * Mc).sum(axis=0) / Nc
        diff = (Mc - mu) * np.sqrt(nc)[:, None]
        K = len(nc)
        cov = diff.T @ diff / (K - 1 if K > 1 else K) + float(gamma) * np.eye(d)
        G += (Nc - 1.0) * cov
        B[j] = mu * Nc                                   # class_mean_sums
        N += Nc
        g_sum += mu * Nc
    g = g_sum / N
    G += N * np.outer(g, g)
    W = (np.linalg.pinv(G) @ B.T).T
    norm = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)   # F.normalize(eps=1e-12)
    return W / norm, {"N": int(N), "d": int(d), "means": int(sum(len(n) for n in counts_by_class))}


class FedCOF(Server):
    """FedCOF (NeurIPS 2025): class covariances estimated from client class means, within-class ridge
    classifier with unit-norm class weights (REFERENCE baseline)."""

    REGIME = SINGLE_ROUND
    CLIENT_CLASS = clientFedCOF
    RUN_FLAGS = ("REFERENCE",)
    HPARAMS = {
        "gamma_shrink": HParam("float", 1.0, "shrinkage gamma added to every estimated class covariance (paper: 1 "
                                             "for ViT-B/16 and SqueezeNet, 0.1 for MobileNetV2)", low=0.0),
        "means_per_client": HParam("int", 1, "class means each client sends per class (1 = released code; > 1 = "
                                             "paper Appendix E, disjoint random subsets of >= 2 samples)", low=1),
    }

    def init_algorithm(self) -> None:
        self.class_means = {c: [] for c in range(self.num_classes)}
        self.class_counts = {c: [] for c in range(self.num_classes)}
        self.head_classes, self.W = None, None

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        task_classes = {int(c) for c in self.scenario.tasks[t]}
        d = int(self.store.dim)
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            if not set(cls.tolist()) <= task_classes:
                raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")
            if p["means"].shape != (len(cls), d) or np.any(np.asarray(p["counts"]) <= 0):
                raise RuntimeError(f"client {cid}: means of the wrong shape or empty counts")
            for c, mu, n in zip(cls, p["means"], p["counts"], strict=True):
                self.class_means[int(c)].append(np.asarray(mu, dtype=np.float64))
                self.class_counts[int(c)].append(int(n))

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if not self.class_counts[int(c)]]
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")
        seen = np.asarray(self.scenario.seen_classes(t), dtype=np.int64)
        self.W, info = fedcof_head([np.stack(self.class_means[int(c)]) for c in seen],
                                   [np.asarray(self.class_counts[int(c)]) for c in seen], self.hp["gamma_shrink"])
        self.head_classes = seen
        self.logger.info(f"    FedCOF head after task {t}: {len(seen)} classes, N = {info['N']}, {info['means']} "
                         f"class means, gamma {self.hp['gamma_shrink']:g}")

    def final_broadcast_payload(self, t: int):
        return {"classes": self.head_classes, "W": self.W}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        if self.head_classes is None or not np.array_equal(np.sort(self.head_classes), np.sort(seen)):
            raise RuntimeError("the FedCOF head does not cover exactly the seen classes")
        return self.head_classes[np.argmax(np.asarray(Z, dtype=np.float64) @ self.W.T, axis=1)]
