"""Tests of the replay machinery (numpy only): run from the repository root or repro/.

    python -m pytest repro/tests -q

Synthetic dumps are built with literal numpy restatements of the official code (tests/official_numpy.py).
run_ours.py must then reproduce the official predictions exactly through FICL's own loop (MATCH, identical
predictions, classifier equal to rounding), and must report MISMATCH when the official side differs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
sys.path.insert(0, REPRO)
sys.path.insert(0, HERE)

import official_numpy as off  # noqa: E402
from common import DUMP_FORMAT, load_dump, save_dump  # noqa: E402

D, C = 12, 6


def features(rng, y, C, d, sep=0.7):
    means = np.random.default_rng(123).standard_normal((C, d)) * sep
    return (means[y] + rng.standard_normal((len(y), d))).astype(np.float32)


def split_dirichlet(rng, y, K, alpha):
    client = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        p = rng.dirichlet(np.full(K, alpha))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            client[part] = k
    for k in range(K):                       # every client gets at least one row
        if not np.any(client == k):
            client[k] = k
    return client


def single_task_dump(path, kind, seed=0, corrupt=False):
    """FedCOF-repo style (FedCOF, FedNCM, Fed3R) or AFL dump on one task."""
    rng = np.random.default_rng(seed)
    K = 5
    ytr = np.repeat(np.arange(C), 30)
    yte = np.repeat(np.arange(C), 10)
    Xtr, Xte = features(rng, ytr, C, D), features(rng, yte, C, D)
    client = split_dirichlet(rng, ytr, K, 0.5)
    parts = [(Xtr[client == k].astype(np.float64), ytr[client == k]) for k in range(K)]
    evalrows = rng.permutation(len(yte))                        # official evaluation order (shuffled loader)
    Z = Xte[evalrows].astype(np.float64)
    arrays = {"train_x": Xtr, "train_y": ytr, "train_task": np.zeros(len(ytr), np.int64), "train_client": client,
              "test_x": Xte, "test_y": yte, "evalrows__0": evalrows}
    official, ours, paper = {}, [], []
    if kind == "fedcof":
        stats = [off.fedcof_repo_client_means(X, y) for X, y in parts]
        gamma = 0.5 if corrupt else 1.0
        W = off.fedcof_repo_fedcof(stats, C, D, gamma)
        Wn = off.fedcof_repo_fedncm(stats, C, D)
        W3, b3 = off.fed3r_official(parts, C, 0.01)
        for key, Wc, b in (("FedCOF", W, None), ("FedNCM", Wn, None), ("Fed3R", W3.T, b3)):
            logits = Z @ Wc.T + (0 if b is None else b)
            arrays[f"pred__{key}__0"] = np.argmax(logits, axis=1)
            arrays[f"W__{key}__0"], arrays[f"Wcls__{key}__0"] = Wc, np.arange(C)
            if b is not None:
                arrays[f"b__{key}__0"] = b
            official[key] = {"acc": [100 * float(np.mean(arrays[f"pred__{key}__0"] == yte[evalrows]))]}
        ours = [{"algorithm": "FedCOF", "label": "FedCOF", "hparams": {"gamma_shrink": 1.0, "means_per_client": 1},
                 "compare_to": "FedCOF"},
                {"algorithm": "Fed3R", "label": "Fed3R", "hparams": {"ridge_lambda": 0.01, "rf_dim": 0},
                 "compare_to": "Fed3R"}]
        paper = ["fedcof_neurips25_t3_vit_fedcof"]
    elif kind == "fed3r":
        W3, b3 = off.fed3r_official(parts, C, 0.01)
        arrays["pred__Fed3R__0"] = np.argmax(Z @ W3 + b3, axis=1)
        arrays["W__Fed3R__0"], arrays["b__Fed3R__0"], arrays["Wcls__Fed3R__0"] = W3.T, b3, np.arange(C)
        official["Fed3R"] = {"acc": [100 * float(np.mean(arrays["pred__Fed3R__0"] == yte[evalrows]))]}
        ours = [{"algorithm": "Fed3R", "label": "Fed3R", "hparams": {"ridge_lambda": 0.01, "rf_dim": 0},
                 "compare_to": "Fed3R"},
                {"algorithm": "Fed3R", "label": "Fed3R-RF5k", "compare_to": None,
                 "hparams": {"ridge_lambda": 0.01, "rf_dim": 64, "rf_sigma": 3.0}}]
        paper = ["fed3r_icml24_t6_fed3r", "fed3r_icml24_t6_fed3r_rf5k"]
    else:
        W = off.afl_official(parts, C, 1.0)
        arrays["pred__AFL__0"] = np.argmax(Z @ W, axis=1)
        arrays["W__AFL__0"], arrays["Wcls__AFL__0"] = W.T, np.arange(C)
        official["AFL"] = {"acc": [100 * float(np.mean(arrays["pred__AFL__0"] == yte[evalrows]))]}
        ours = [{"algorithm": "AFL", "label": "AFL", "hparams": {"rg": 1.0, "clean_reg": True}, "compare_to": "AFL"}]
        paper = ["afl_cvpr25_t2_cifar100"]
    meta = {"format": DUMP_FORMAT, "source": {"repo": f"synthetic-{kind}", "commit": "0" * 40},
            "setting": {"dataset": "synthetic", "backbone": "none", "num_classes": C, "num_clients": K,
                        "partition_label": "dir", "alpha": 0.5, "tasks": [list(range(C))], "seed": seed},
            "official": official, "ours": ours, "paper": paper}
    save_dump(path, meta, arrays)


def stsa_dump(path, variant, corrupt=False, own_groups=False):
    """STSA / STSA-E dump: 3 tasks of 2 classes, 3 clients, M = 24, official W_rand and dummy groups."""
    rng = np.random.default_rng(7)
    K, M, KD, ridge = 3, 24, 3, 1.0
    tasks = [[0, 1], [2, 3], [4, 5]]
    ytr = np.repeat(np.arange(C), 40)
    yte = np.repeat(np.arange(C), 10)
    Xtr, Xte = features(rng, ytr, C, D), features(rng, yte, C, D)
    task = np.array([next(t for t, cl in enumerate(tasks) if c in cl) for c in ytr], dtype=np.int64)
    client = np.empty(len(ytr), dtype=np.int64)
    for t in range(3):
        rows = np.flatnonzero(task == t)
        client[rows] = split_dirichlet(rng, ytr[rows], K, 0.5)
    W_rand = rng.standard_normal((D, M)).astype(np.float32)
    group = {t: np.full(len(ytr), -1, dtype=np.int64) for t in range(3)}
    G, P = np.zeros((M, M)), np.zeros((M, C))
    arrays = {"train_x": Xtr, "train_y": ytr, "train_task": task, "train_client": client, "test_x": Xte,
              "test_y": yte, "W_rand": W_rand}
    accs = []
    for t in range(3):
        P_t, G_t = np.zeros((M, C)), np.zeros((M, M))
        sums, cnts = [], []
        for k in range(K):
            rows = np.flatnonzero((task == t) & (client == k))
            if len(rows) == 0:
                continue
            order = rng.permutation(rows)                         # official loader order (shuffle=True)
            H = off.stsa_project(Xtr[order].astype(np.float64), W_rand.astype(np.float64))
            Y = off.onehot(ytr[order], C)
            P_t += H.T @ Y
            G_t += H.T @ H
            perm = rng.permutation(len(order))
            step = len(order) // KD
            for g in range(KD):
                sl = perm[g * step:((g + 1) * step if g < KD - 1 else len(order))]
                group[t][order[sl]] = g
                s = np.zeros((C, M))
                n = np.zeros(C)
                for i in sl:
                    s[ytr[order[i]]] += H[i]
                    n[ytr[order[i]]] += 1
                sums.append(s)
                cnts.append(n)
        if variant == "STSA_E":
            G_t = off.stsa_e_task_gram(sums, cnts, P_t)
        P += P_t
        G += G_t
        Wo = np.linalg.solve(G + (2.0 if corrupt else ridge) * np.eye(M), P).T
        seen = np.concatenate(tasks[:t + 1])
        evalrows = np.flatnonzero(np.isin(yte, seen))
        W_seen = Wo[: len(seen)]
        pred = off.stsa_cosine_predict(Xte[evalrows].astype(np.float64), W_rand.astype(np.float64), W_seen)
        arrays[f"evalrows__{t}"] = evalrows
        arrays[f"pred__{variant}__{t}"] = pred
        arrays[f"W__{variant}__{t}"], arrays[f"Wcls__{variant}__{t}"] = W_seen, seen
        arrays[f"group__{t}"] = group[t]
        accs.append(100 * float(np.mean(pred == yte[evalrows])))
    hp = {"rp_dim": M, "ridge": ridge}
    if variant == "STSA_E":
        hp["subsets"] = KD
    entry = {"algorithm": variant, "label": variant, "hparams": hp, "compare_to": variant,
             "compare_mode": "own_groups" if own_groups else "replay"}
    meta = {"format": DUMP_FORMAT, "source": {"repo": "synthetic-stsa", "commit": "0" * 40},
            "setting": {"dataset": "synthetic", "backbone": "none", "num_classes": C, "num_clients": K,
                        "partition_label": "dir", "alpha": 0.5, "tasks": tasks, "seed": 0},
            "official": {variant: {"acc": accs}}, "ours": [entry], "paper": []}
    save_dump(path, meta, arrays)


def run(tmp_path, dump, *extra):
    import run_ours
    out = tmp_path / "out"
    code = run_ours.main([str(dump), "--dataset_root", str(tmp_path / "data"), "--out", str(out), *extra])
    with open(out / (os.path.splitext(os.path.basename(dump))[0] + ".json")) as f:
        return code, json.load(f)


@pytest.mark.parametrize("kind", ["fedcof", "afl"])
def test_single_task_dumps_match_exactly(tmp_path, kind):
    dump = tmp_path / f"{kind}_seed0.npz"
    single_task_dump(dump, kind)
    code, rep = run(tmp_path, dump)
    assert code == 0
    exact = [r for r in rep["results"] if r["precision"] == "exact"]
    assert exact and all(r["verdict"] == "MATCH" for r in exact)
    for r in exact:
        assert r["min_agreement"] == 1.0
        assert max(t["W_rel_diff"] for t in r["tasks"]) < 1e-8
        assert all(t.get("b_rel_diff", 0.0) < 1e-8 for t in r["tasks"])
    cache = [r for r in rep["results"] if r["precision"] == "cache"]
    assert cache and all(r["verdict"].startswith("not judged") for r in cache)


@pytest.mark.parametrize("variant", ["STSA", "STSA_E"])
def test_stsa_dumps_match_exactly_with_official_projection_and_groups(tmp_path, variant):
    dump = tmp_path / f"{variant.lower()}_dump.npz"
    stsa_dump(dump, variant)
    code, rep = run(tmp_path, dump, "--precision", "exact")
    assert code == 0
    (r,) = rep["results"]
    assert r["verdict"] == "MATCH" and r["min_agreement"] == 1.0
    assert len(r["tasks"]) == 3
    assert max(t["W_rel_diff"] for t in r["tasks"]) < 1e-8


def test_mismatch_is_reported(tmp_path):
    dump = tmp_path / "fedcof_bad.npz"
    single_task_dump(dump, "fedcof", corrupt=True)              # official side with another gamma
    code, rep = run(tmp_path, dump, "--precision", "exact", "--only", "FedCOF")
    (r,) = rep["results"]
    assert max(t["W_rel_diff"] for t in r["tasks"]) > 1e-3
    assert code == (1 if r["verdict"] == "MISMATCH" else 0)


def test_other_ridge_shows_in_the_classifier_difference(tmp_path):
    """Predictions can survive a small hyperparameter change; the classifier difference does not."""
    dump = tmp_path / "stsa_bad.npz"
    stsa_dump(dump, "STSA", corrupt=True)
    code, rep = run(tmp_path, dump, "--precision", "exact")
    (r,) = rep["results"]
    assert max(t["W_rel_diff"] for t in r["tasks"]) > 1e-3


def test_own_groups_not_judged(tmp_path):
    dump = tmp_path / "stsae_own.npz"
    stsa_dump(dump, "STSA_E", own_groups=True)
    code, rep = run(tmp_path, dump, "--precision", "exact")
    (r,) = rep["results"]
    assert r["verdict"].startswith("not judged") and code == 0


def test_dump_checks(tmp_path):
    dump = tmp_path / "x.npz"
    single_task_dump(dump, "afl")
    meta, a = load_dump(str(dump))
    a["train_y"] = a["train_y"].copy()
    a["train_y"][0] = C + 3                                    # a label outside every task
    with pytest.raises(ValueError):
        save_dump(str(tmp_path / "y.npz"), meta, a)


def test_reimport_same_dump_is_reused_and_other_dump_refused(tmp_path):
    from import_dump import import_dump
    dump = tmp_path / "afl_seed0.npz"
    single_task_dump(dump, "afl")
    root = str(tmp_path / "data")
    a = import_dump(str(dump), root)
    assert import_dump(str(dump), root) == a
    single_task_dump(dump, "afl", seed=1)                      # same file name, other content
    with pytest.raises(FileExistsError):
        import_dump(str(dump), root)


def test_entry_without_official_code(tmp_path):
    dump = tmp_path / "fed3r_seed0.npz"
    single_task_dump(dump, "fed3r")
    code, rep = run(tmp_path, dump, "--precision", "exact")
    assert code == 0
    by = {r["label"]: r for r in rep["results"]}
    assert by["Fed3R"]["verdict"] == "MATCH"
    assert by["Fed3R-RF5k"]["verdict"].startswith("no official code") and by["Fed3R-RF5k"]["ours_final"] > 0
    keys = {pl["paper_key"]: pl for pl in rep["paper"]}
    assert keys["fed3r_icml24_t6_fed3r"]["compared"] == "official run"
    assert keys["fed3r_icml24_t6_fed3r_rf5k"]["compared"].startswith("ours")
