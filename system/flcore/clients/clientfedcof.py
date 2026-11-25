"""Client of FedCOF (Goswami et al., NeurIPS 2025): class means and counts of the current task, one upload.

Per task a client sends, for every class it holds in its current task (official federate_classes.py,
FedCOFClient.compute_class_means):

    means    mu_{k,c} = mean of its features of class c       [r, d]
    counts   n_{k,c}                                           [r] int
    classes  the class of every row                            [r] int

With means_per_client = M > 1 (paper Sec. 4.2 and Appendix E, not in the released code) a client splits
the samples of each class into disjoint random subsets ("sampling without replacement") of at least 2
samples each and sends one mean per subset: min(M, n // 2) subsets of near-equal size (a class with
fewer than 4 samples, or with 1, sends a single mean). The split is drawn from the client's seeded
generator for this task. M = 1 is the released code.
"""
from __future__ import annotations

import numpy as np

from flcore.clients.clientbase import Client
from flcore.closedform_utils import add_class_sums, class_positions


def split_for_means(n: int, M: int, rng: np.random.Generator) -> list:
    """Index sets of the subsets whose means a client sends for one class of n samples (M means at most)."""
    m = max(1, min(int(M), int(n) // 2))
    if m == 1:
        return [np.arange(n)]
    return [np.sort(part) for part in np.array_split(rng.permutation(n), m)]


class clientFedCOF(Client):
    def on_begin_task(self, t: int) -> None:
        self._payload = None

    def train_round(self, t: int, r: int) -> None:
        rows = self.train_rows()
        if len(rows) == 0:
            self._payload = None
            return
        classes = np.sort(np.asarray(self.task_classes(), dtype=np.int64))
        M = int(self.hp["means_per_client"])
        if M == 1:                                  # the released code: one mean per class
            sums = np.zeros((len(classes), self.store.dim), dtype=np.float64)
            counts = np.zeros(len(classes), dtype=np.int64)
            for Z, y in self.store.iter_chunks(rows):
                add_class_sums(np.asarray(Z, dtype=np.float64), class_positions(y, classes), sums, counts)
            keep = counts > 0
            self._payload = {"classes": classes[keep], "means": sums[keep] / counts[keep, None],
                             "counts": counts[keep]}
            return
        Z, y = self.load_task_data()
        Z = np.asarray(Z, dtype=np.float64)
        rng = self.rng(t, r)
        out_cls, out_means, out_counts = [], [], []
        for c in classes:
            Zc = Z[y == c]
            if len(Zc) == 0:
                continue
            for part in split_for_means(len(Zc), M, rng):
                out_cls.append(int(c))
                out_means.append(Zc[part].mean(axis=0))
                out_counts.append(len(part))
        self._payload = {"classes": np.asarray(out_cls, dtype=np.int64), "means": np.stack(out_means),
                         "counts": np.asarray(out_counts, dtype=np.int64)}

    def upload_payload(self, t: int, r: int):
        return self._payload
