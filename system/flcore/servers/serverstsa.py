"""STSA and STSA-E (Guan, Zhu, Zhou, Liu, Wang, Luo and Gu, "Enhancing Federated Class-Incremental Learning via
Spatial-Temporal Statistics Aggregation", WWW 2026, arXiv:2506.01327), on cached features of a frozen backbone.
REFERENCE baselines.

Official code github.com/Yuqin-G/STSA, methods/STSA.py (commit aaaa006): setup_RP, replace_fc1 (STSA),
replace_fc_ours3 + compute_multiple_sum_by_class (STSA-E), convs/linears.py CosineLinear (prediction).

    projection  h = ReLU(z W_rand), W_rand [d, M] ~ N(0, 1), the same for every client; none when M = d
    STSA        clients: G_k = sum h h^T, C_k = sum h y^T (current task)                      (paper Eq. 1)
                server:  G_{1:t} = G_{1:t-1} + sum_k G_k,  C_{1:t} = C_{1:t-1} + sum_k C_k    (Eqs. 2, 4)
    STSA-E      clients: class sums s_{g,i} and counts n_{g,i} of K_D random groups ("dummy clients")
                server, with K = number of groups of the task, n_i = sum_g n_{g,i}, C_i = sum_g s_{g,i}:
                  G_t = sum_i [ A_i sum_g s_{g,i} s_{g,i}^T / n_{g,i}  -  B_i C_i C_i^T ]          (Eq. 9)
                  A_i = (n_i - 1) / (K - 1),  B_i = (n_i - K) / ((K - 1) n_i),
                  each set to 0 where negative (the official torch.where); then as STSA
    classifier  W = (G_{1:t} + gamma I)^-1 C_{1:t}  (torch.linalg.solve)                       (Eqs. 8, 10)
    predict     CosineLinear: argmax_c  <h / ||h||, W_c / ||W_c||>  (sigma = 1 scales all classes alike)

Missing here, and reported in docs/BASELINES.md: the FIRST STAGE ADAPTATION. In the paper and the code the
backbone (ViT-B/16 IN-21K with AdaptFormer-style adapters, r = 64, s = 0.1) is first trained with FedAvg on
task 1 (10 rounds on CIFAR-100, 20 on ImageNet-R, 2 local epochs, lr 0.2), then frozen. On cached features
of a frozen backbone there is nothing to train, so these classes run the statistics aggregation and the
classifier exactly, on the features of the frozen pretrained backbone. run.json records this as
"first_stage_adaptation": "not implemented".

Hyperparameters (defaults: the paper's ViT-B/16 values):
    rp_dim    M; 1250 (paper, ViT; the code's command-line default is 10000, its README uses 1250);
              M = feature dimension means no projection
    ridge     gamma; 1e6 (paper, ViT; main.py args['ridge'] = 1e6). The paper uses 1e4 for ResNet-18
    subsets   K_D of STSA-E; 10 (paper, ViT; main.py sets args["dummy"] = 5). The paper uses 50 for ResNet-18
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientstsa import clientSTSA, clientSTSAE, project, random_projection
from flcore.closedform_utils import symmetrize_upper, wire_seed
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam

STSA_HPARAMS = {
    "rp_dim": HParam("int", 1250, "M, the random projection dimension (paper, ViT-B/16: 1250); equal to the feature "
                                  "dimension = no projection", low=1),
    "ridge": HParam("float", 1e6, "gamma of the ridge solution (paper: 1e6 for ViT-B/16, 1e4 for ResNet-18)",
                    low=0.0),
}


def stsa_solve(G: np.ndarray, C: np.ndarray, ridge: float) -> np.ndarray:
    """W [m, M] = ((G + ridge I)^-1 C)^T, the official torch.linalg.solve(...).T."""
    return np.linalg.solve(G + float(ridge) * np.eye(G.shape[0]), C).T


def stsae_task_gram(group_sums: np.ndarray, group_counts: np.ndarray, group_classes: np.ndarray,
                    num_groups: int, num_classes: int):
    """G_t and C_t of STSA-E (replace_fc_ours3) from one task's group statistics.

    group_sums [r, M] (s_{g,i}), group_counts [r] (n_{g,i} > 0), group_classes [r] (i), num_groups K.
    Returns (G_t [M, M], C_t [M, num_classes], n [num_classes])."""
    K = int(num_groups)
    if K < 2:
        raise RuntimeError(f"STSA-E needs at least 2 groups per task (got K = {K}): (K - 1) divides")
    M = group_sums.shape[1]
    n = np.bincount(group_classes, weights=group_counts, minlength=num_classes).astype(np.float64)
    C_t = np.zeros((M, num_classes), dtype=np.float64)
    np.add.at(C_t.T, group_classes, group_sums)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (n - 1.0) / (K - 1.0)
        B = (n - K) / ((K - 1.0) * n)
    A = np.where(A < 0, 0.0, A)
    B = np.where(~np.isfinite(B) | (B < 0), 0.0, B)
    coef = A[group_classes] / group_counts
    G_t = (group_sums * coef[:, None]).T @ group_sums
    G_t -= (C_t * B) @ C_t.T
    return G_t, C_t, n


class _STSABase(Server):
    REGIME = SINGLE_ROUND
    RUN_FLAGS = ("REFERENCE",)

    def init_algorithm(self) -> None:
        d, M = int(self.store.dim), int(self.hp["rp_dim"])
        self.rp_seed = None if M == d else wire_seed(self.rng("stsa_rp"))
        self.W_rand = None if M == d else random_projection(self.rp_seed, d, M)
        self._seed_sent = set()
        self.G = np.zeros((M, M), dtype=np.float64)
        self.C = np.zeros((M, self.num_classes), dtype=np.float64)
        self.counts = np.zeros(self.num_classes, dtype=np.float64)
        self.head_classes, self.W = None, None

    def extra_run_info(self) -> dict:
        return {"stsa": {"rp_dim": int(self.hp["rp_dim"]), "projection": self.W_rand is not None,
                         "rp_seed": self.rp_seed, "first_stage_adaptation": "not implemented (cached features of a "
                                                                            "frozen backbone; docs/BASELINES.md)"}}

    def download_payload(self, t: int, r: int, client):
        if self.W_rand is not None and client.id not in self._seed_sent:
            self._seed_sent.add(client.id)
            return {"rp_seed": int(self.rp_seed)}
        return None

    def _check_classes(self, t: int, cid: int, cls: np.ndarray) -> None:
        if not set(cls.tolist()) <= {int(c) for c in self.scenario.tasks[t]}:
            raise RuntimeError(f"client {cid} uploaded statistics of classes outside task {t}")

    def on_end_task(self, t: int) -> None:
        missing = [int(c) for c in self.scenario.tasks[t] if self.counts[c] == 0]
        if missing:
            raise RuntimeError(f"task {t}: no training samples arrived for classes {missing}")
        seen = np.asarray(self.scenario.seen_classes(t), dtype=np.int64)
        self.W = stsa_solve(self.G, self.C[:, seen], self.hp["ridge"])
        self.head_classes = seen
        self.logger.info(f"    {type(self).__name__} head after task {t}: {len(seen)} classes, "
                         f"{int(self.counts.sum())} samples, M = {self.G.shape[0]}, ridge {self.hp['ridge']:g}")

    def final_broadcast_payload(self, t: int):
        return {"classes": self.head_classes, "W": self.W}

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        if self.head_classes is None or not np.array_equal(np.sort(self.head_classes), np.sort(seen)):
            raise RuntimeError("the STSA head does not cover exactly the seen classes")
        H = project(Z, self.W_rand)
        Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        Wn = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-12)
        return self.head_classes[np.argmax(Hn @ Wn.T, axis=1)]


class STSA(_STSABase):
    """STSA (WWW 2026): ridge classifier on randomly projected features from spatially and temporally summed
    second-order statistics; cosine prediction. First-stage adaptation not included (REFERENCE baseline)."""

    CLIENT_CLASS = clientSTSA
    HPARAMS = dict(STSA_HPARAMS)

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        M = self.G.shape[0]
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            self._check_classes(t, cid, cls)
            if p["G"].matrix.shape != (M, M) or p["C"].shape != (M, len(cls)):
                raise RuntimeError(f"client {cid}: statistics of the wrong shape")
            self.G += symmetrize_upper(p["G"].matrix)
            self.C[:, cls] += p["C"]
            self.counts[cls] += p["counts"]


class STSA_E(_STSABase):
    """STSA-E (WWW 2026): as STSA, but clients send only class sums and counts of K_D random groups, and the
    server estimates the second-order statistics from them (REFERENCE baseline)."""

    CLIENT_CLASS = clientSTSAE
    HPARAMS = {**STSA_HPARAMS,
               "subsets": HParam("int", 10, "K_D, groups ('dummy clients') per client and task (paper, ViT-B/16: 10; "
                                            "the released main.py uses 5; ResNet-18: 50)", low=1)}

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        if not uploads:
            return
        M = self.G.shape[0]
        K = 0
        sums, counts, classes = [], [], []
        for cid, p in uploads:
            cls = np.asarray(p["classes"], dtype=np.int64)
            self._check_classes(t, cid, cls)
            if p["sums"].shape != (len(cls), M):
                raise RuntimeError(f"client {cid}: statistics of the wrong shape")
            K += int(p["num_groups"])
            sums.append(p["sums"])
            counts.append(np.asarray(p["counts"], dtype=np.float64))
            classes.append(cls)
        G_t, C_t, n = stsae_task_gram(np.concatenate(sums), np.concatenate(counts), np.concatenate(classes), K,
                                      self.num_classes)
        self.G += G_t
        self.C += C_t
        self.counts += n
