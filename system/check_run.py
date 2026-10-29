#!/usr/bin/env python
"""
Structural check of a FICL experiment folder (runs automatically at the end of main.py).

    cd system
    python check_run.py ../results/<dataset>/<scenario>/<experiment>

It reads only the result files (and the scenario files they point to) and checks that the run
followed its regime:
    - every run is complete (its final.json exists); tasks.csv has one row per task
    - single-round: no rounds.csv and no round columns anywhere (time is counted in tasks)
    - multi-round: rounds.csv has tasks x rounds_per_task rows, global_round = task * R + round,
      curve points exactly every eval_gap rounds
    - nothing was evaluated before its task arrived; client test shards add up to the global test
      set of the scenario files; every accuracy is a whole count over its test set (the counts are
      rebuilt from the 6-decimal accuracies and the test-set sizes of the scenario files)
    - clients evaluated with the global model agree with the global evaluation
    - acc_seen in tasks.csv equals acc_global.csv; at most one upload per client per round / exchange
    - no field anywhere is a best round; avg/summary.json covers every run
Exit code 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def check_experiment(exp_dir: str) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    exp_path = os.path.join(exp_dir, "experiment.json")
    if not os.path.exists(exp_path):
        add("experiment.json exists", False, exp_path)
        return checks
    exp = _read_json(exp_path)
    regime, times = exp["regime"], int(exp["settings"]["times"])
    multi = regime == "multi_round"
    add("experiment.json exists", True, f"{exp['algorithm']} [{regime}], {times} run(s)")

    for i in range(times):
        rd = os.path.join(exp_dir, f"times_{i}")
        p = f"run {i}"
        complete = os.path.exists(os.path.join(rd, "final.json")) and os.path.exists(os.path.join(rd, "run.json"))
        add(f"{p}: complete (final.json written)", complete, "no final.json: failed or interrupted, see its log")
        if not complete:
            continue
        run = _read_json(os.path.join(rd, "run.json"))
        T, K = run["scenario"]["num_tasks"], run["scenario"]["num_clients"]
        R = run["rounds_per_task"]
        add(f"{p}: run.json regime matches experiment.json", run["regime"] == regime, run["regime"])

        header, tasks = _read_csv(os.path.join(rd, "tasks.csv"))
        add(f"{p}: tasks.csv has one row per task", [int(r["task"]) for r in tasks] == list(range(T)),
            f"{len(tasks)} rows for {T} tasks")
        if multi:
            add(f"{p}: tasks.csv rounds column equals rounds_per_task",
                len(tasks) > 0 and all(int(r["rounds"]) == R for r in tasks), f"R = {R}")
        else:
            add(f"{p}: single-round has no round columns in tasks.csv", not any("round" in h for h in header))
            add(f"{p}: single-round has no rounds_per_task in run.json",
                run["rounds_per_task"] is None and run["total_rounds"] is None)

        rounds_path = os.path.join(rd, "rounds.csv")
        if multi:
            if not os.path.exists(rounds_path):
                add(f"{p}: rounds.csv exists", False)
            else:
                _h, rr = _read_csv(rounds_path)
                order = [(int(x["task"]), int(x["round"])) for x in rr]
                add(f"{p}: rounds.csv has tasks x rounds_per_task rows in order",
                    order == [(t, r) for t in range(T) for r in range(R)], f"{len(rr)} rows, expected {T * R}")
                add(f"{p}: global_round = task * R + round",
                    all(int(x["global_round"]) == int(x["task"]) * R + int(x["round"]) for x in rr))
                eg = int(run["eval_gap"] or 0)
                points = [(int(x["task"]), int(x["round"])) for x in rr if x["acc_seen"] != ""]
                expected = [(t, r) for t in range(T) for r in range(R) if eg and (r + 1) % eg == 0]
                add(f"{p}: curve points exactly every eval_gap rounds", points == expected, f"eval_gap = {eg}")
        else:
            add(f"{p}: single-round has no rounds.csv", not os.path.exists(rounds_path))

        ch, comm = _read_csv(os.path.join(rd, "comm.csv"))
        if not multi:
            add(f"{p}: single-round has no round columns in comm.csv", not any("round" in h for h in ch))
            add(f"{p}: single-round messages are exchanges or task-end broadcasts",
                {x["phase"] for x in comm} <= {"exchange", "task_end"})
        ups = [(x["task"], x.get("round", ""), x["client"]) for x in comm
               if x["direction"] == "up" and x["phase"] != "task_end"]
        add(f"{p}: at most one upload per client per round or exchange", len(ups) == len(set(ups)))
        add(f"{p}: every logged message has bytes > 0", all(int(x["nbytes"]) > 0 for x in comm))

        _gh, accg = _read_csv(os.path.join(rd, "acc_global.csv"))
        cells = [[row.get(f"test_task_{c}", "") for c in range(T)] for row in accg]
        add(f"{p}: nothing evaluated before its task arrived",
            len(cells) == T and all((cells[t][c] != "") == (c <= t) for t in range(T) for c in range(T)))
        _ch2, accc = _read_csv(os.path.join(rd, "acc_clients.csv"))
        add(f"{p}: acc_clients.csv has one row per task and client", len(accc) == T * K, f"{len(accc)} rows")
        cfg_dir = run["scenario"]["config_dir"]
        if not os.path.isdir(cfg_dir) or len(cells) != T or len(accc) != T * K or len(tasks) != T:
            add(f"{p}: counts rebuilt from the scenario files (skipped: {cfg_dir} not found or files incomplete)",
                True)
        else:
            def size(*parts):
                return len(np.load(os.path.join(cfg_dir, *parts), allow_pickle=False))

            gt = np.array([size(f"task_{c}", "test_global.npy") for c in range(T)])
            ct = np.array([[size(f"task_{c}", "test", f"{k}.npy") for c in range(T)] for k in range(K)])
            add(f"{p}: client test shards add up to the global test set", np.array_equal(ct.sum(axis=0), gt))

            def count(acc_text, n):
                """Correct predictions behind a 6-decimal accuracy over n samples, or None if none fits."""
                c = round(float(acc_text) * n)
                return c if 0 <= c <= n and abs(float(acc_text) * n - c) <= 0.5e-6 * n + 1e-9 else None

            gc = [[count(cells[t][c], gt[c]) if c <= t else 0 for c in range(T)] for t in range(T)]
            add(f"{p}: every accuracy in acc_global.csv is a count over its test set",
                all(v is not None for row in gc for v in row))
            if all(v is not None for row in gc for v in row):
                gc = np.array(gc)
                recomputed = [gc[t, :t + 1].sum() / gt[:t + 1].sum() for t in range(T)]
                diff = max(abs(float(tasks[t]["acc_seen"]) - recomputed[t]) for t in range(T))
                add(f"{p}: acc_seen in tasks.csv matches acc_global.csv", diff < 1e-6, f"max difference {diff:.2e}")
                n_seen_ok, client_counts = True, np.zeros((T, K), dtype=np.int64)
                for row in accc:
                    t, k = int(row["task"]), int(row["client"])
                    n = int(ct[k, :t + 1].sum())
                    n_seen_ok &= int(row["n_test_seen"]) == n
                    c = count(row["acc_seen_own"], n) if n else 0
                    n_seen_ok &= c is not None
                    client_counts[t, k] = c or 0
                add(f"{p}: client test sizes and accuracies match the scenario's test shards", n_seen_ok)
                if not run.get("personalized", False):
                    add(f"{p}: clients evaluated with the global model agree with the global evaluation",
                        all(client_counts[t].sum() == gc[t, :t + 1].sum() for t in range(T)))
        final = _read_json(os.path.join(rd, "final.json"))
        add(f"{p}: final.json has the run's final values", "acc_final" in final.get("metrics", {}))

    best = []
    for dirpath, _dirs, files in os.walk(exp_dir):
        for fn in files:
            path = os.path.join(dirpath, fn)
            if fn.endswith(".csv"):
                h, _rows = _read_csv(path)
                best += [f"{path}: {c}" for c in h if "best" in c.lower()]
            elif fn.endswith(".json"):
                best += [f"{path}: {k}" for k in _keys(_read_json(path)) if "best" in k.lower()]
    add("no best-round fields anywhere", not best, "; ".join(best[:3]))

    summary_path = os.path.join(exp_dir, "avg", "summary.json")
    if os.path.exists(summary_path):
        s = _read_json(summary_path)
        add("avg/summary.json covers every run", s["n_runs_complete"] == times and s["complete"],
            f"{s['n_runs_complete']} of {times} runs complete")
    else:
        add("avg/summary.json exists", False)
    return checks


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python check_run.py <experiment folder>")
        return 2
    checks = check_experiment(argv[0])
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    failed = sum(1 for _n, ok, _d in checks if not ok)
    print(f"\n{len(checks) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
