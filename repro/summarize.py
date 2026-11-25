#!/usr/bin/env python
"""One table over all reproduction results (repro/work/results/*.json written by run_ours.py).

    python repro/summarize.py [--results repro/work/results] [--md repro/work/REPRO_SUMMARY.md]

Groups the seeds of one setting (dump names that differ only in _seed<n>) and reports, per our method and
precision: the verdicts (MATCH k/n), the smallest prediction agreement and largest accuracy difference over
all seeds and tasks, the official and our accuracies (mean +- SD over seeds, all seeds shown, none dropped),
and the paper value [R] next to the mean of the official runs (or of ours where no official code exists).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import PAPER_FLAG_DIFF, load_paper_numbers  # noqa: E402


def fmt(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "-"
    return f"{np.mean(xs):.2f} +- {np.std(xs, ddof=1) if len(xs) > 1 else 0.0:.2f} (n={len(xs)})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=os.path.join(HERE, "work", "results"))
    p.add_argument("--md", default=None, help="also write the table as markdown")
    a = p.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.results, "*.json")))
    if not files:
        print(f"no results in {a.results}")
        return 1
    groups = {}
    for f in files:
        with open(f) as fh:
            rep = json.load(fh)
        setting = re.sub(r"_seed\d+$", "", os.path.splitext(os.path.basename(f))[0])
        g = groups.setdefault(setting, {"rows": {}, "paper": {}, "seeds": []})
        g["seeds"].append(rep["setting"].get("seed"))
        for r in rep["results"]:
            g["rows"].setdefault((r["label"], str(r["compare_to"]), r["precision"]), []).append(r)
        for pl in rep.get("paper", []):
            g["paper"].setdefault(pl["paper_key"], []).append(pl)
    papers = load_paper_numbers()["numbers"]
    lines = [("| setting | ours | vs official | precision | verdicts | min agreement | max abs diff | "
              "official final | ours final | official avg | ours avg |"), "|---" * 11 + "|"]
    plines = ["| setting | paper value [R] | compared | mean over seeds | diff | note |", "|---" * 6 + "|"]
    for setting, g in groups.items():
        for (label, key, prec), rs in sorted(g["rows"].items()):
            verd = [r["verdict"] for r in rs]
            judged = [v for v in verd if v in ("MATCH", "MISMATCH")]
            vtxt = f"MATCH {judged.count('MATCH')}/{len(judged)}" if judged else verd[0]
            agree = [r["min_agreement"] for r in rs if not math.isnan(r["min_agreement"])]
            diff = [r["max_abs_acc_diff"] for r in rs if not math.isnan(r["max_abs_acc_diff"])]
            agree_txt = f"{100 * min(agree):.2f} %" if agree else "-"
            diff_txt = f"{max(diff):.2f}" if diff else "-"
            lines.append(f"| {setting} | {label} | {key} | {prec} | {vtxt} | {agree_txt} | {diff_txt} | "
                         f"{fmt([r['official_final'] for r in rs])} | {fmt([r['ours_final'] for r in rs])} | "
                         f"{fmt([r['official_avg'] for r in rs])} | {fmt([r['ours_avg'] for r in rs])} |")
        for keyp, pls in g["paper"].items():
            vals = [pl["value"] for pl in pls]
            m = float(np.mean(vals))
            pv = papers[keyp]["value"]
            note = f"differs by more than {PAPER_FLAG_DIFF} point" if abs(m - pv) > PAPER_FLAG_DIFF else ""
            plines.append(f"| {setting} | {keyp}: {pv:.2f} ({papers[keyp]['metric']}) | {pls[0]['compared']} | "
                          f"{fmt(vals)} | {m - pv:+.2f} | {note} |")
    text = ("## Implementation check: ours vs official on identical inputs\n\n" + "\n".join(lines) +
            "\n\n## Official runs vs the papers ([R] values, orientation only)\n\n" + "\n".join(plines) + "\n")
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
