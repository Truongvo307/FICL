#!/usr/bin/env python
"""Run our FICL implementation on the exact inputs of an official run and compare, task by task.

    python repro/run_ours.py repro/work/dumps/fedcof_cifar100_vit_seed0.npz
    python repro/run_ours.py DUMP.npz --precision exact          # only the float32-feature replay

Needs only the FICL environment (numpy): the dump holds the official features, split and predictions.

For every entry of the dump's "ours" list (algorithm + the hyperparameters of the official run), the dump is
imported as a FICL dataset (import_dump.py) and FICL's own server and clients run through the shared loop
(main.py's classes, no shortcut). Two precisions:

    exact   the official float32 features are given to the run (FICL's feature store reads them instead of
            its float16 cache): the decisive comparison, implementation against implementation
    cache   the run reads FICL's float16 cache of the same features, as every FICL experiment does
            (shows the effect of FICL's storage precision; not part of the verdict)

Replays that remove randomness which is not part of the method:
    STSA / STSA-E   the official random projection W_rand is used instead of ours (same distribution)
    STSA-E          the official dummy groups are used (compare_mode "replay"); entries with compare_mode
                    "own_groups" draw our own groups and are reported without a verdict

Verdict per entry (pre-declared, repro/common.py): MATCH when, at every task, at least 99.5 % of the test
predictions equal the official ones and the accuracies differ by at most 0.2 points.
Results: <out>/<dump stem>.json and a printed table.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (MATCH_MAX_ACC_DIFF, MATCH_MIN_AGREEMENT, PAPER_FLAG_DIFF, REPO_ROOT,  # noqa: E402
                    accuracy, json_default, load_dump, load_paper_numbers, rel_diff)
from import_dump import BACKBONE, GROUP, import_dump  # noqa: E402

SYSTEM_DIR = os.path.join(REPO_ROOT, "system")
W_LAYOUT = {"FedCOF": "classes_dim", "STSA": "classes_dim", "STSA_E": "classes_dim",
            "Fed3R": "dim_classes", "AFL": "dim_classes"}


def write_config(path: str, info: dict, entry: dict) -> None:
    lines = ["# written by repro/run_ours.py: our method on the inputs of an official run",
             "scenario:", f"  dataset: {info['dataset']}", f"  partition: {info['partition']}",
             f"  num_clients: {info['num_clients']}"]
    if info["partition"] == "dir":
        lines.append(f"  alpha: {info['alpha']}")
    lines += [f"  class_per_task: {info['class_per_task']}",
              "features:", f"  backbone: {BACKBONE}", f"  group: {GROUP}", "  blocks: [0]", "  tokens: [cls]",
              "schedule:", "  regime: single_round",
              "algorithm:", f"  name: {entry['algorithm']}", "  hparams:"]
    for k, v in entry.get("hparams", {}).items():
        lines.append(f"    {k}: {json.dumps(v)}")
    lines += ["run:", "  seed: 0", "  times: 1", "  goal: repro", "  device: cpu"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ours_classifier(server, algorithm: str):
    """(class ids, W as [classes, dim], bias or None) of our head after a task."""
    W = np.asarray(server.W, dtype=np.float64)
    if W_LAYOUT[algorithm] == "dim_classes":
        W = W.T
    bias = getattr(server, "bias", None)
    return np.asarray(server.head_classes), W, (None if bias is None else np.asarray(bias, dtype=np.float64))


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def replay_groups_by_client(meta: dict, a: dict, K_D: int) -> dict:
    """{(t, k): [positions in client k's sorted task-t rows, one array per official dummy group 0..K_D-1]}."""
    out = {}
    K = int(meta["setting"]["num_clients"])
    for t in range(len(meta["setting"]["tasks"])):
        g = a[f"group__{t}"]
        for k in range(K):
            rows = np.flatnonzero((a["train_task"] == t) & (a["train_client"] == k))   # = client's sorted rows
            if len(rows) == 0:
                continue
            gid = g[rows]
            if np.any(gid < 0) or np.any(gid >= K_D):
                raise ValueError(f"task {t}, client {k}: rows without an official dummy group in 0..{K_D - 1}")
            out[(t, k)] = [np.flatnonzero(gid == i) for i in range(K_D)]
    return out


def run_entry(meta: dict, a: dict, info: dict, entry: dict, precision: str, work: str) -> dict:
    """Run one of our methods through FICL on the dump; return per-task predictions and classifiers."""
    if SYSTEM_DIR not in sys.path:
        sys.path.insert(0, SYSTEM_DIR)
    from main import resolve_args

    algorithm = entry["algorithm"]
    tag = f"{entry.get('label', algorithm)}_{precision}"
    cfg = os.path.join(work, f"{tag}.yaml")
    write_config(cfg, info, entry)
    args, server_cls, _ = resolve_args(["-cfg", cfg, "-droot", info["dataset_root"],
                                        "-rroot", os.path.join(work, "ficl_results")])
    args.exp_dir = tempfile.mkdtemp(prefix=f"{tag}_", dir=work)

    T = len(meta["setting"]["tasks"])
    test64 = a["test_x"].astype(np.float64) if precision == "exact" else \
        a["test_x"].astype(np.float16).astype(np.float64)
    records = []

    class Recorder(server_cls):
        def on_end_task(self, t):
            super().on_end_task(t)
            rows = a[f"evalrows__{t}"]
            seen = self.scenario.seen_classes(t)
            pred = np.asarray(self.predict(test64[rows], seen))
            records.append({"pred": pred, "head": ours_classifier(self, algorithm)})

    stack = contextlib.ExitStack()
    with stack:
        if algorithm in ("STSA", "STSA_E") and "W_rand" in a:
            import flcore.clients.clientstsa as cmod
            import flcore.servers.serverstsa as smod
            W_rand = np.asarray(a["W_rand"], dtype=np.float64)

            def official_projection(seed, in_dim, M):
                if (int(in_dim), int(M)) != W_rand.shape:
                    raise ValueError(f"official W_rand is {W_rand.shape}, the run asks for ({in_dim}, {M})")
                return W_rand
            stack.enter_context(patched(cmod, "random_projection", official_projection))
            stack.enter_context(patched(smod, "random_projection", official_projection))
        if algorithm == "STSA_E" and entry.get("compare_mode") == "replay":
            import flcore.clients.clientstsa as cmod
            groups = replay_groups_by_client(meta, a, int(entry["hparams"]["subsets"]))
            original = cmod.clientSTSAE.train_round

            def train_round(self, t, r, _orig=original):
                key = (int(t), int(self.id))
                if key not in groups:
                    return _orig(self, t, r)

                def official_groups(n, K_D, rng):
                    gs = groups[key]
                    if sum(len(g) for g in gs) != n or len(gs) != int(K_D):
                        raise ValueError(f"task {t}, client {self.id}: official groups cover "
                                         f"{sum(len(g) for g in gs)} of {n} rows in {len(gs)} groups, K_D {K_D}")
                    return gs
                with patched(cmod, "dummy_groups", official_groups):
                    return _orig(self, t, r)
            stack.enter_context(patched(cmod.clientSTSAE, "train_round", train_round))

        server = Recorder(args, 0)
        if precision == "exact":
            exact = np.concatenate([a["train_x"], a["test_x"]]).astype(np.float32)[:, None, None, :]
            if exact.shape != server.store._feats.shape:
                raise RuntimeError(f"store shape {server.store._feats.shape} != dump {exact.shape}")
            server.store._feats = exact
        server.train()
    if len(records) != T:
        raise RuntimeError(f"{tag}: {len(records)} tasks recorded, expected {T}")
    return {"records": records, "exp_dir": args.exp_dir, "config": cfg}


def ours_only(meta: dict, a: dict, entry: dict, out: dict) -> dict:
    """Entries without official code (Fed3R-RF): our accuracy only, for the paper comparison."""
    T = len(meta["setting"]["tasks"])
    accs = [accuracy(out["records"][t]["pred"], a["test_y"][a[f"evalrows__{t}"]]) for t in range(T)]
    return {"label": entry.get("label", entry["algorithm"]), "algorithm": entry["algorithm"], "compare_to": None,
            "hparams": entry.get("hparams", {}), "tasks": [{"task": t, "acc_ours": x} for t, x in enumerate(accs)],
            "official_final": float("nan"), "official_avg": float("nan"), "ours_final": accs[-1],
            "ours_avg": float(np.mean(accs)), "min_agreement": float("nan"), "max_abs_acc_diff": float("nan"),
            "verdict": "no official code (paper comparison only)"}


def compare(meta: dict, a: dict, entry: dict, out: dict) -> dict:
    key = entry.get("compare_to")
    if key is None:
        return ours_only(meta, a, entry, out)
    off = meta["official"][key]
    T = len(meta["setting"]["tasks"])
    rows = []
    for t in range(T):
        er = a[f"evalrows__{t}"]
        y = a["test_y"][er]
        p_off = a[f"pred__{key}__{t}"]
        rec = out["records"][t]
        r = {"task": t, "n_test": int(len(er)), "acc_official": accuracy(p_off, y), "acc_ours": accuracy(rec["pred"], y),
             "agreement": float(np.mean(rec["pred"] == p_off))}
        r["acc_diff"] = r["acc_ours"] - r["acc_official"]
        if off.get("acc") and abs(off["acc"][t] - r["acc_official"]) > 0.005:
            raise RuntimeError(f"official {key}, task {t}: logged accuracy {off['acc'][t]} but its predictions give "
                               f"{r['acc_official']:.3f} (dump inconsistent)")
        if f"W__{key}__{t}" in a:
            cls, W, b = rec["head"]
            ocls = a[f"Wcls__{key}__{t}"]
            pos = {int(c): i for i, c in enumerate(cls)}
            if sorted(pos) != sorted(int(c) for c in ocls):
                raise RuntimeError(f"task {t}: our head covers other classes than the official one")
            order = [pos[int(c)] for c in ocls]
            r["W_rel_diff"] = rel_diff(a[f"W__{key}__{t}"], W[order])
            if f"b__{key}__{t}" in a and b is not None:
                r["b_rel_diff"] = rel_diff(a[f"b__{key}__{t}"], b[order])
        rows.append(r)
    accs_o = [r["acc_official"] for r in rows]
    accs_u = [r["acc_ours"] for r in rows]
    ok = all(r["agreement"] >= MATCH_MIN_AGREEMENT and abs(r["acc_diff"]) <= MATCH_MAX_ACC_DIFF for r in rows)
    judged = entry.get("compare_mode", "replay") != "own_groups"
    return {"label": entry.get("label", entry["algorithm"]), "algorithm": entry["algorithm"], "compare_to": key,
            "hparams": entry.get("hparams", {}), "tasks": rows,
            "official_final": accs_o[-1], "official_avg": float(np.mean(accs_o)),
            "ours_final": accs_u[-1], "ours_avg": float(np.mean(accs_u)),
            "min_agreement": min(r["agreement"] for r in rows),
            "max_abs_acc_diff": max(abs(r["acc_diff"]) for r in rows),
            "verdict": ("MATCH" if ok else "MISMATCH") if judged else "not judged (own random groups)"}


def paper_lines(meta: dict, results: list) -> list:
    """Official run (or ours, where no official code exists) vs the paper, for this one seed.
    repro/summarize.py averages over the seeds."""
    papers = load_paper_numbers()["numbers"]
    official, ours = {}, {}
    for res in results:
        if res["precision"] == "exact":
            if res["compare_to"] is not None:
                official[res["compare_to"]] = (res["official_final"], res["official_avg"])
            ours[res["label"]] = (res["ours_final"], res["ours_avg"])
    out = []
    for keyp in meta.get("paper", []):
        p = papers[keyp]
        if p.get("official_key") in official:
            who, pair = "official run", official[p["official_key"]]
        elif p.get("ours_label") in ours:
            who, pair = "ours (no official code)", ours[p["ours_label"]]
        else:
            continue
        val = pair[0 if p["metric"] == "final" else 1]
        out.append({"paper_key": keyp, "metric": p["metric"], "paper": p["value"], "compared": who, "value": val,
                    "diff": val - p["value"], "flag": abs(val - p["value"]) > PAPER_FLAG_DIFF})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump")
    p.add_argument("--precision", nargs="+", choices=["exact", "cache"], default=["exact", "cache"])
    p.add_argument("--only", nargs="*", help="labels of the dump's 'ours' entries to run (default: all)")
    p.add_argument("--dataset_root", default=os.path.join(HERE, "work", "ficl_data"))
    p.add_argument("--out", default=os.path.join(HERE, "work", "results"))
    a_ = p.parse_args(argv)

    meta, arrays = load_dump(a_.dump)
    info = import_dump(a_.dump, a_.dataset_root)
    stem = os.path.splitext(os.path.basename(a_.dump))[0]
    work = os.path.join(a_.out, stem)
    os.makedirs(work, exist_ok=True)
    results = []
    for entry in meta["ours"]:
        if a_.only and entry.get("label", entry["algorithm"]) not in a_.only:
            continue
        for prec in a_.precision:
            out = run_entry(meta, arrays, info, entry, prec, work)
            res = compare(meta, arrays, entry, out)
            res.update(precision=prec, exp_dir=out["exp_dir"])
            if prec == "cache" and res["compare_to"] is not None:
                res["verdict"] = "not judged (float16 cache)"
            results.append(res)
    report = {"dump": os.path.abspath(a_.dump), "source": meta.get("source"), "setting": meta.get("setting"),
              "official": {k: {kk: v for kk, v in o.items() if kk != "acc"} for k, o in meta["official"].items()},
              "criteria": {"min_agreement": MATCH_MIN_AGREEMENT, "max_abs_acc_diff_points": MATCH_MAX_ACC_DIFF},
              "results": results, "paper": paper_lines(meta, results)}
    with open(os.path.join(a_.out, f"{stem}.json"), "w") as f:
        json.dump(report, f, indent=1, default=json_default)

    print(f"\n{stem}: {meta['source'].get('repo')} @ {meta['source'].get('commit', '?')[:7]}  "
          f"({meta['setting'].get('dataset')}, {meta['setting'].get('backbone')}, "
          f"{meta['setting']['num_clients']} clients, {len(meta['setting']['tasks'])} task(s))")
    print(f"  {'ours':<22}{'vs official':<16}{'prec':<7}{'acc official':>13}{'acc ours':>10}{'max |diff|':>11}"
          f"{'min agree':>10}{'max W diff':>11}  verdict")
    for r in results:
        wd = max((t.get("W_rel_diff", float("nan")) for t in r["tasks"]), default=float("nan"))
        print(f"  {r['label']:<22}{str(r['compare_to']):<16}{r['precision']:<7}{r['official_final']:>13.2f}"
              f"{r['ours_final']:>10.2f}{r['max_abs_acc_diff']:>11.2f}{100 * r['min_agreement']:>9.2f}%"
              f"{wd:>11.1e}  {r['verdict']}")
    for pl in report["paper"]:
        print(f"  paper {pl['paper_key']} ({pl['metric']}): {pl['paper']:.2f} [R] | {pl['compared']}, this seed "
              f"{pl['value']:.2f} | diff {pl['diff']:+.2f}" + ("  <- more than 1 point" if pl["flag"] else ""))
    bad = [r for r in results if r["verdict"] == "MISMATCH"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
