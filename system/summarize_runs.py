#!/usr/bin/env python
"""
One table of finished experiments, for comparing algorithms run on the same scenario and features.

    cd system
    python summarize_runs.py ../results                       # every experiment below a folder
    python summarize_runs.py ../results/CIFAR100/cl_10clients_dir_0.5_cpt10 --csv table.csv

Reads only what the runs wrote: experiment.json (algorithm, flags, scenario, features, hyperparameters) and
avg/summary.json (the mean and SD over the seeds of the final values, never a selected round). Incomplete
experiments are listed and marked, never averaged over fewer seeds. Rows are grouped by scenario and
features, so a row is only compared with rows that read the same data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

COLUMNS = ("scenario", "features", "algorithm", "flags", "hparams", "runs", "acc_final", "acc_avg_incremental",
           "client_acc_final_min", "MB_up", "MB_down", "seconds")


def find_experiments(paths):
    for p in paths:
        if os.path.isfile(os.path.join(p, "experiment.json")):
            yield p
            continue
        for root, dirs, files in os.walk(p):
            if "experiment.json" in files:
                dirs[:] = []
                yield root


def fmt(m: dict, scale: float = 100.0) -> str:
    if not m or m.get("mean") is None:
        return "-"
    sd = m.get("sd")
    return f"{scale * m['mean']:.2f}" + (f" +- {scale * sd:.2f}" if sd is not None else "")


def row(exp_dir: str) -> dict:
    with open(os.path.join(exp_dir, "experiment.json")) as f:
        exp = json.load(f)
    s = exp["settings"]
    scen = (f"{s['dataset']} {s['partition']}" + (f" a={s['data_alpha']:g}" if s.get("data_alpha") else "")
            + f" K={s.get('num_clients')} cpt={s['class_per_task']}")
    feats = (f"{s.get('backbone')}/{s.get('feature_group')} b{s.get('blocks')} {s.get('tokens')}"
             + (f" proj{s['proj_rank']}" if s.get("proj_rank") else "")) if s.get("backbone") else f"model {s.get('model')}"
    out = {"scenario": scen, "features": feats, "algorithm": exp["algorithm"], "flags": "+".join(exp["run_flags"]),
           "hparams": " ".join(f"{k}={v}" for k, v in sorted((s.get("hparams") or {}).items())), "dir": exp_dir}
    path = os.path.join(exp_dir, "avg", "summary.json")
    if not os.path.exists(path):
        out.update(runs="no summary (unfinished?)")
        return out
    with open(path) as f:
        sm = json.load(f)
    m = sm["metrics"]
    out.update(runs=f"{sm['n_runs_complete']}/{sm['n_runs_expected']}" + ("" if sm["complete"] else " INCOMPLETE"),
               acc_final=fmt(m.get("acc_final")) if sm["complete"] else "-",
               acc_avg_incremental=fmt(m.get("acc_avg_incremental")) if sm["complete"] else "-",
               client_acc_final_min=fmt(m.get("client_acc_final_min")) if sm["complete"] else "-",
               MB_up=fmt(m.get("bytes_up_total"), 1e-6), MB_down=fmt(m.get("bytes_down_total"), 1e-6),
               seconds=fmt(m.get("seconds_total"), 1.0))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Table of finished experiments (accuracies in %)")
    p.add_argument("paths", nargs="+", help="experiment folders or folders above them")
    p.add_argument("--csv", help="also write the table to this CSV file")
    a = p.parse_args(argv)
    rows = [row(d) for d in sorted(set(find_experiments(a.paths)))]
    if not rows:
        print("no experiment.json found below " + ", ".join(a.paths))
        return 1
    rows.sort(key=lambda r: (r["scenario"], r["features"], r["algorithm"], r["hparams"]))
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS[2:]}
    group = None
    for r in rows:
        if (r["scenario"], r["features"]) != group:
            group = (r["scenario"], r["features"])
            print(f"\n{r['scenario']} | {r['features']}")
            print("  " + "  ".join(c.ljust(widths[c]) for c in COLUMNS[2:]))
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLUMNS[2:]))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=(*COLUMNS, "dir"))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
