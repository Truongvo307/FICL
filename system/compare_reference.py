#!/usr/bin/env python
"""
Compare a reference-check experiment with the fed-mammoth re-run and the published number.

    cd system
    python compare_reference.py configs/reference/ref_cifar100_fedavg_b0.05.yaml ../results/<...>/<experiment> \
        --mammoth ../../fed-mammoth/logs/cifar100_fedavg_b0.05_seed*.log

It reads only final values: our avg/summary.json (acc_final of every seed, never a selected round) and,
for each fed-mammoth log, the last "Mean accuracy up to task T" line of the last task (the evaluation
after the last task's end_task). The published number comes from reports/reference_numbers.json [R].

Agreement rule (the protocol's rule for a gap): two means agree when they differ by at most
max(2 points, 2 x the larger seed SD). A difference is a reason to look for a bug or a protocol difference
first (reports/REFERENCE_CHECK.md lists the known ones), not a result.
Exit code 0 when ours agrees with the fed-mammoth re-run (or no log was given), 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(HERE, "..", "reports", "reference_numbers.json")
MAMMOTH_LINE = re.compile(r"Mean accuracy up to task (\d+):\s*([0-9.]+)\s*%")


def mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else None)


def read_ours(exp_dir: str) -> dict:
    path = os.path.join(exp_dir, "avg", "summary.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: give the experiment folder (the one with experiment.json and avg/)")
    with open(path) as f:
        s = json.load(f)
    if not s.get("complete"):
        raise SystemExit(f"{exp_dir}: {s['n_runs_complete']} of {s['n_runs_expected']} runs complete; "
                         f"every seed must finish before comparing (no seed is dropped)")
    vals = [100.0 * v for v in s["metrics"]["acc_final"]["values"]]
    return {"values": vals, "flags": s.get("run_flags", []), "algorithm": s.get("algorithm")}


def read_mammoth(path: str, num_tasks: int) -> float:
    with open(path, errors="replace") as f:
        text = f.read().replace("\r", "\n")
    if "Total training time" not in text:
        raise SystemExit(f"{path}: no 'Total training time' line, so the run did not finish")
    hits = [(int(t), float(a)) for t, a in MAMMOTH_LINE.findall(text)]
    last = [a for t, a in hits if t == num_tasks]
    if not last:
        found = sorted({t for t, _ in hits})
        raise SystemExit(f"{path}: no evaluation after task {num_tasks} (tasks found: {found}); "
                         f"was it run with the same number of tasks?")
    return last[-1]


def verdict(a: dict, b: dict) -> tuple[bool, float, float]:
    sds = [x for x in (a["sd"], b["sd"]) if x is not None]
    tol = max(2.0, 2.0 * max(sds)) if sds else 2.0
    gap = a["mean"] - b["mean"]
    return abs(gap) <= tol, gap, tol


def fmt(m: dict) -> str:
    sd = f" +- {m['sd']:.2f}" if m["sd"] is not None else ""
    return f"{m['mean']:.2f}{sd}  (n={m['n']}: {', '.join(f'{v:.2f}' for v in m['values'])})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare a reference-check run with fed-mammoth and the paper")
    p.add_argument("config", help="configs/reference/<name>.yaml (its file name selects the reference entry)")
    p.add_argument("experiment", help="our experiment folder (holds experiment.json, times_<i>/ and avg/)")
    p.add_argument("--mammoth", nargs="*", default=[], help="fed-mammoth logs of the same setting, one per seed")
    p.add_argument("--out", help="also write the comparison as JSON to this file")
    p.add_argument("--any_algorithm", action="store_true", help=argparse.SUPPRESS)  # the tool's own test
    args = p.parse_args(argv)

    key = os.path.splitext(os.path.basename(args.config))[0]
    with open(REFERENCE_FILE) as f:
        ref = json.load(f)
    if key not in ref["entries"]:
        raise SystemExit(f"{key}: not in {REFERENCE_FILE}; known: {sorted(ref['entries'])}")
    entry = ref["entries"][key]
    with open(os.path.join(args.experiment, "experiment.json")) as f:
        exp = json.load(f)
    if exp.get("algorithm") != "FedAvg" and not args.any_algorithm:
        raise SystemExit(f"{args.experiment}: algorithm is {exp.get('algorithm')!r}; the reference check runs FedAvg")
    num_tasks = int(exp["plan"]["num_tasks"])
    ours_raw = read_ours(args.experiment)
    m, sd = mean_sd(ours_raw["values"])
    ours = {"mean": m, "sd": sd, "n": len(ours_raw["values"]), "values": ours_raw["values"]}

    lines = [f"reference check  {key}   ({num_tasks} task(s); accuracy after the last task, %, all seen classes)",
             f"  ours (this repository)        {fmt(ours)}   flags: {' '.join(ours_raw['flags'])}"]
    report = {"config": key, "num_tasks": num_tasks, "ours": ours, "rule": "agree if |gap| <= max(2, 2 x larger SD)"}
    status = 0
    if args.mammoth:
        vals = [read_mammoth(path, num_tasks) for path in args.mammoth]
        mm, msd = mean_sd(vals)
        mam = {"mean": mm, "sd": msd, "n": len(vals), "values": vals, "logs": args.mammoth}
        ok, gap, tol = verdict(ours, mam)
        lines.append(f"  fed-mammoth re-run            {fmt(mam)}")
        lines.append(f"    -> gap {gap:+.2f}, tolerance {tol:.2f}: "
                     + ("AGREE" if ok else "DIFFER: look for a bug or a protocol difference before anything else"))
        report["fed_mammoth"] = {**mam, "gap": gap, "tolerance": tol, "agree": ok}
        status = 0 if ok else 1
        if len(vals) != ours["n"]:
            lines.append(f"    note: {len(vals)} fed-mammoth seeds vs {ours['n']} of ours")
    else:
        lines.append("  fed-mammoth re-run            not given (--mammoth): the decisive comparison is missing")
    if entry.get("paper") is not None:
        pap = {"mean": float(entry["paper"]), "sd": None, "n": 1, "values": [float(entry["paper"])]}
        ok, gap, tol = verdict(ours, pap)
        checked = "" if entry.get("checked_against_pdf") else "  (not yet checked against the PDF)"
        lines.append(f"  paper [R]                     {entry['paper']:.2f}   {entry['source']}: {entry['what']}{checked}")
        lines.append(f"    -> gap {gap:+.2f}, tolerance {tol:.2f}: " + ("within" if ok else "outside")
                     + " (indicative only: the paper does not report every setting; see REFERENCE_CHECK.md)")
        report["paper"] = {**entry, "gap": gap, "tolerance": tol, "within": ok}
    else:
        lines.append(f"  paper [R]                     none: {entry.get('what', '')}")
    print("\n".join(lines))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
