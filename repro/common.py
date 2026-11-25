"""Shared pieces of the official-repository reproduction checks (numpy only).

A *dump* is what an official run leaves for the comparison with ours: the exact features the official code
used (as float32, in the official row order), its client split and task split, and the official classifier
and predictions. Dumps are written in the environment of the official repository (torch) and read in the
FICL environment (numpy), so this module must not import torch.

Dump file: one .npz written by save_dump(), with a JSON "meta" entry and these arrays:

    train_x [N, d] float32      features of every training sample the official run used
    train_y [N] int64           its label in the official label space (0..C-1)
    train_task [N] int64        task of the sample (0 for single-task methods)
    train_client [N] int64      client that holds it
    test_x [M, d] float32, test_y [M] int64
    evalrows__<t> int64         indices into test_x: the test samples the official code evaluated after task t,
                                in the official order
    pred__<key>__<t> int64      official prediction for each of those samples (key = official method key)
    W__<key>__<t> float64       official classifier as [classes, dim]; Wcls__<key>__<t> its class ids
    b__<key>__<t> float64       official bias (Fed3R only)
    W_rand float32 [d, M]       STSA's random projection (STSA dumps only)
    group__<t> int64 [N]        STSA-E: the official dummy group of each training row in task t (-1 elsewhere)

meta (JSON):
    format, created, source {repo, commit, entry, argv, notes}, setting {dataset, backbone, num_classes,
    num_clients, partition_label, alpha, tasks [[class ids] per task], seed},
    official {key: {acc: [per task, %], hparams, precision, notes}},
    ours [{algorithm, hparams, compare_to, label}], paper [keys of paper_numbers.json]
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

DUMP_FORMAT = "ficl-official-dump/1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Pre-declared criteria (docs/OFFICIAL_REPRO.md, section 3). Fixed before any official run was made.
MATCH_MIN_AGREEMENT = 0.995      # share of test predictions identical to the official ones, every task
MATCH_MAX_ACC_DIFF = 0.2         # |accuracy ours - accuracy official| in points, every task
PAPER_FLAG_DIFF = 1.0            # official run vs paper: differences above this many points are flagged

REQUIRED = ("train_x", "train_y", "train_task", "train_client", "test_x", "test_y")


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_dump(path: str, meta: dict, arrays: dict) -> str:
    """Check and write a dump. Returns its sha256."""
    meta = dict(meta)
    meta.setdefault("format", DUMP_FORMAT)
    meta.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    check_dump(meta, arrays)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    path = str(path)
    tmp = path + ".partial.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta, indent=1)), **arrays)
    os.replace(tmp, path)
    return sha256_file(path)


def load_dump(path: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    check_dump(meta, arrays)
    return meta, arrays


def check_dump(meta: dict, a: dict) -> None:
    """Raise if a dump is inconsistent (wrong shapes, labels outside the task split, missing predictions)."""
    if meta.get("format") != DUMP_FORMAT:
        raise ValueError(f"unknown dump format {meta.get('format')!r}")
    for k in REQUIRED:
        if k not in a:
            raise ValueError(f"dump lacks {k}")
    n, m = len(a["train_y"]), len(a["test_y"])
    if a["train_x"].ndim != 2 or a["train_x"].shape[0] != n or a["test_x"].shape != (m, a["train_x"].shape[1]):
        raise ValueError(f"feature shapes {a['train_x'].shape} / {a['test_x'].shape} do not match the labels")
    for k in ("train_task", "train_client"):
        if a[k].shape != (n,):
            raise ValueError(f"{k} must have one entry per training row")
    s = meta["setting"]
    tasks = [np.asarray(c, dtype=np.int64) for c in s["tasks"]]
    C = int(s["num_classes"])
    allc = np.concatenate(tasks)
    if len(np.unique(allc)) != len(allc) or allc.min() < 0 or allc.max() >= C:
        raise ValueError("setting.tasks must hold disjoint class ids in 0..num_classes-1")
    for t, cls in enumerate(tasks):
        rows = a["train_task"] == t
        if not np.isin(a["train_y"][rows], cls).all():
            raise ValueError(f"task {t}: training labels outside the task's classes")
    if a["train_task"].min() < 0 or a["train_task"].max() >= len(tasks):
        raise ValueError("train_task outside 0..num_tasks-1")
    if a["train_client"].min() < 0 or a["train_client"].max() >= int(s["num_clients"]):
        raise ValueError("train_client outside 0..num_clients-1")
    if not np.isin(a["test_y"], allc).all():
        raise ValueError("test labels outside the classes of the tasks")
    for key, info in meta.get("official", {}).items():
        acc = info.get("acc", [])
        for t in range(len(acc)):
            if f"pred__{key}__{t}" not in a or f"evalrows__{t}" not in a:
                raise ValueError(f"official {key}: no predictions for task {t}")
            if a[f"pred__{key}__{t}"].shape != a[f"evalrows__{t}"].shape:
                raise ValueError(f"official {key}, task {t}: predictions and evaluated rows differ in length")


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return 100.0 * float(np.mean(np.asarray(pred) == np.asarray(y))) if len(y) else float("nan")


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


def load_paper_numbers() -> dict:
    with open(os.path.join(HERE, "paper_numbers.json")) as f:
        return json.load(f)


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
