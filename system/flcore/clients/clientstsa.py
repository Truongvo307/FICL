"""Clients of STSA and STSA-E (Guan et al., "Enhancing Federated Class-Incremental Learning via Spatial-Temporal
Statistics Aggregation", WWW 2026), on cached features of a frozen backbone.

Both map a feature z to h = ReLU(z W_rand), W_rand [d, M] ~ N(0, 1) drawn from a seed the server sends once
("the server can distribute a common seed to all clients"); with M equal to the feature dimension no
projection is applied (official methods/STSA.py: `if not (args["M"] == 768 or args["M"] == 512)`).

STSA (clientSTSA; official replace_fc1), per task, current-task data only:
    G_k = sum h h^T                    [M, M], sent as Symmetric
    C_k = sum h y^T  (one-hot)         [M, m], columns of the m classes the client holds

STSA-E (clientSTSAE; official replace_fc_ours3 and compute_multiple_sum_by_class), per task: the client's
current-task samples are permuted at random and cut into K_D consecutive groups ("dummy clients") of
n // K_D samples, the last group taking the remainder; for every group it sends the class sums and class
counts. No second-order statistic is sent.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client
from flcore.closedform_utils import add_class_sums, class_positions, one_hot
from utils.comm_utils import Symmetric


def random_projection(seed: int, in_dim: int, M: int):
    """W_rand [in_dim, M] ~ N(0, 1), or None when M equals the feature dimension (no projection)."""
    if int(M) == int(in_dim):
        return None
    return np.random.default_rng(int(seed)).standard_normal((int(in_dim), int(M)))


def project(Z: np.ndarray, W_rand) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    return Z if W_rand is None else np.maximum(Z @ W_rand, 0.0)


def dummy_groups(n: int, K_D: int, rng: np.random.Generator) -> list:
    """compute_multiple_sum_by_class: a random permutation of n samples cut into K_D groups of n // K_D,
    the last one taking the rest (groups are empty when n < K_D, except the last)."""
    perm = rng.permutation(int(n))
    step = int(n) // int(K_D)
    return [perm[i * step:((i + 1) * step if i < K_D - 1 else int(n))] for i in range(int(K_D))]


class _ProjectedClient(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None
        if not hasattr(self, "_W_rand"):
            self._W_rand, self._have_seed = None, False

    def receive(self, t: int, r, payload: dict) -> None:
        if payload and "rp_seed" in payload:
            self._W_rand = random_projection(payload["rp_seed"], self.store.dim, self.hp["rp_dim"])
            self._have_seed = True

    def _check_projection(self) -> None:
        if int(self.hp["rp_dim"]) != int(self.store.dim) and not self._have_seed:
            raise RuntimeError(f"client {self.id}: the random projection seed has not arrived")

    def upload_payload(self, t: int, r: int):
        return self._payload


class clientSTSA(_ProjectedClient):
    def train_round(self, t: int, r: int) -> None:
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        self._check_projection()
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        M = int(self.hp["rp_dim"])
        G = np.zeros((M, M), dtype=np.float64)
        C = np.zeros((M, len(classes)), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for Z, y in self.store.iter_chunks(rows):
            H = project(Z, self._W_rand)
            pos = class_positions(y, classes)
            G += H.T @ H
            C += H.T @ one_hot(pos, len(classes))
            counts += np.bincount(pos, minlength=len(classes))
        keep = counts > 0
        self._payload = {"classes": classes[keep], "G": Symmetric(G), "C": C[:, keep], "counts": counts[keep]}


class clientSTSAE(_ProjectedClient):
    def train_round(self, t: int, r: int) -> None:
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        self._check_projection()
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        Z, y = self.load_task_data()
        H = project(Z, self._W_rand)
        K_D = int(self.hp["subsets"])
        out_group, out_cls, out_sums, out_counts = [], [], [], []
        for g, idx in enumerate(dummy_groups(len(y), K_D, self.rng(t, r))):
            if len(idx) == 0:
                continue
            sums = np.zeros((len(classes), H.shape[1]), dtype=np.float64)
            counts = np.zeros(len(classes), dtype=np.int64)
            add_class_sums(H[idx], class_positions(y[idx], classes), sums, counts)
            keep = counts > 0
            out_group.append(np.full(int(keep.sum()), g, dtype=np.int64))
            out_cls.append(classes[keep])
            out_sums.append(sums[keep])
            out_counts.append(counts[keep])
        self._payload = {"num_groups": K_D, "group": np.concatenate(out_group), "classes": np.concatenate(out_cls),
                         "sums": np.concatenate(out_sums), "counts": np.concatenate(out_counts)}
