"""Small numerical helpers shared by the closed-form baselines on cached features (Fed3R, FedCOF, STSA,
STSA-E, AFL). Each baseline keeps its own method in its own client and server files; these helpers only
do bookkeeping that is identical in all of them (class sums, one-hot products, symmetric matrices,
seeded random matrices). Everything is float64, as the loop's other closed-form code.
"""
from __future__ import annotations

import numpy as np


def class_positions(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Index of every label in `classes` (sorted ascending); refuses labels outside it."""
    classes = np.asarray(classes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    pos = np.searchsorted(classes, y)
    if len(y) and (np.any(pos >= len(classes)) or not np.array_equal(classes[np.minimum(pos, len(classes) - 1)], y)):
        raise RuntimeError("a training label is not among the current task's classes")
    return pos


def one_hot(pos: np.ndarray, m: int) -> np.ndarray:
    """[n, m] float64 one-hot rows from positions 0..m-1."""
    out = np.zeros((len(pos), int(m)), dtype=np.float64)
    out[np.arange(len(pos)), pos] = 1.0
    return out


def add_class_sums(Z: np.ndarray, pos: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> None:
    """sums[j] += sum of the rows of Z with pos == j; counts[j] += their number (in place)."""
    sums += one_hot(pos, sums.shape[0]).T @ Z
    counts += np.bincount(pos, minlength=sums.shape[0]).astype(np.int64)


def symmetrize_upper(M: np.ndarray) -> np.ndarray:
    """The symmetric matrix whose upper triangle (with the diagonal) is that of M: what a receiver
    rebuilds from a Symmetric(M) message under the wire rule of comm_utils."""
    U = np.triu(M)
    return U + np.triu(M, 1).T


def wire_seed(rng: np.random.Generator) -> int:
    """A non-negative int64 seed drawn from a generator, small enough to travel as one integer."""
    return int(rng.integers(0, 2 ** 62))
