"""Result files of the FCL track: one folder per run (times_<i>, as in HtFL) and avg/ per experiment.

    times_<i>/run.json         what ran: seeds, scenario (name, sha256), features (cache sha256), schedule
              status.json      initialized | running | complete | failed | interrupted (+ error, traceback)
              tasks.csv        one row per task: accuracy on all seen classes and on the new task, client
                               mean / min, bytes up / down, seconds (multi-round also: rounds, last global round)
              acc_global.csv   accuracy matrix: row = after task t, column = test set of task i (i <= t)
              acc_clients.csv  per task and client: accuracy on the client's own test shards of seen tasks
              counts.npz       the correct / total counts behind every accuracy, int64:
                               glob_correct [T, T], glob_total [T], client_correct [K, T, T], client_total [K, T]
              comm.csv         every message: task, phase, client, direction, bytes, keys
                               (multi-round also: round, global_round)
              rounds.csv       multi-round only: one row per round (selected clients, uploads, train loss,
                               curve accuracy every eval_gap rounds, bytes, seconds)
              clients/client_<k>.csv   per-task rows of client k
              final.json       final values of the run (definitions included); nothing is a best round
    avg/summary.json, avg/tasks.csv   mean and SD (ddof=1) over all complete runs

Single-round runs have no rounds.csv and no round columns: their time is counted in tasks.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone

import numpy as np

SINGLE_ROUND, MULTI_ROUND = "single_round", "multi_round"      # same values as utils/cl_config.py

METRIC_DEFINITIONS = {
    "acc_final": "accuracy on the global test set of all classes, after the last task (sample-weighted)",
    "acc_avg_incremental": "mean over tasks t of the accuracy on all classes seen up to t (A_avg)",
    "client_acc_final_mean": "mean over clients of the accuracy on their own test shards after the last task "
                             "(clients without test data are left out)",
    "client_acc_final_min": "lowest client accuracy on its own test shards after the last task",
    "clients_with_test_final": "number of clients with test data after the last task",
    "bytes_up_total": "bytes sent by clients, counted with the wire rule in run.json",
    "bytes_down_total": "bytes sent by the server, counted with the wire rule in run.json",
    "seconds_total": "wall-clock seconds of the run",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(obj):
    """numpy values to plain Python; NaN and infinity become None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj


def write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else f"{float(v):.6f}"
    return str(v)


def write_csv(path: str, header: list[str], rows: list[dict]) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([_fmt(row.get(h)) for h in header])
    os.replace(tmp, path)


def _ratio(num, den) -> float:
    return float(num) / float(den) if den else float("nan")


class CLResultSaver:
    """Collects one run's measurements and rewrites its files after every task (so partial runs are readable)."""

    def __init__(self, run_dir: str, regime: str, num_tasks: int, num_clients: int,
                 rounds_per_task: int | None = None, run_flags=()):
        if regime not in (SINGLE_ROUND, MULTI_ROUND):
            raise ValueError(f"unknown regime {regime!r}")
        self.multi = regime == MULTI_ROUND
        if self.multi and not rounds_per_task:
            raise ValueError("multi-round results need rounds_per_task")
        self.run_dir, self.regime = run_dir, regime
        self.T, self.K = int(num_tasks), int(num_clients)
        self.R = int(rounds_per_task) if self.multi else None
        self.run_flags = list(run_flags)
        self.glob_correct = np.zeros((self.T, self.T), dtype=np.int64)
        self.glob_total = np.zeros(self.T, dtype=np.int64)
        self.client_correct = np.zeros((self.K, self.T, self.T), dtype=np.int64)
        self.client_total = np.zeros((self.K, self.T), dtype=np.int64)
        self.tasks_done = 0
        self.task_rows: list[dict] = []
        self.round_rows: list[dict] = []
        self.comm_rows: list[dict] = []
        self.client_rows: list[list[dict]] = [[] for _ in range(self.K)]
        self.long_rows: list[dict] = []
        self.task_bytes = np.zeros((self.T, 2), dtype=np.int64)            # [:, 0] up, [:, 1] down
        self.client_task_bytes = np.zeros((self.K, self.T, 2), dtype=np.int64)
        self.round_bytes: dict[tuple[int, int], list[int]] = {}
        self._started = None
        os.makedirs(os.path.join(run_dir, "clients"), exist_ok=True)

    # ---- small files
    def write_json(self, name: str, obj) -> None:
        write_json(os.path.join(self.run_dir, name), obj)

    def set_status(self, status: str, **extra) -> None:
        if self._started is None:
            self._started = now()
        self.write_json("status.json", {"status": status, "started": self._started, "updated": now(), **extra})

    # ---- communication
    def log_comm(self, t: int, phase: str, r: int | None, client: int, direction: str, nbytes: int,
                 keys: str) -> None:
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be up or down, got {direction!r}")
        if phase not in ("exchange", "round", "task_end"):
            raise ValueError(f"unknown phase {phase!r}")
        if phase == "round" and not self.multi:
            raise ValueError("single-round runs have no rounds")
        if phase == "exchange" and self.multi:
            raise ValueError("multi-round runs communicate in rounds")
        d = 0 if direction == "up" else 1
        nbytes = int(nbytes)
        self.task_bytes[t, d] += nbytes
        self.client_task_bytes[client, t, d] += nbytes
        row = {"task": t, "phase": phase, "client": client, "direction": direction, "nbytes": nbytes, "keys": keys}
        if self.multi:
            if phase == "round":
                row["round"], row["global_round"] = r, t * self.R + r
                self.round_bytes.setdefault((t, r), [0, 0])[d] += nbytes
        self.comm_rows.append(row)

    # ---- rounds (multi-round only)
    def log_round(self, t: int, r: int, selected: list[int], uploads: int, train_loss: float | None,
                  acc_seen: float | None, seconds: float) -> None:
        if not self.multi:
            raise ValueError("single-round runs have no rounds")
        up, down = self.round_bytes.get((t, r), [0, 0])
        self.round_rows.append({
            "task": t, "round": r, "global_round": t * self.R + r, "clients_selected": len(selected),
            "selected": " ".join(str(k) for k in selected), "uploads": uploads, "train_loss": train_loss,
            "acc_seen": acc_seen, "seconds": seconds, "bytes_up": up, "bytes_down": down})

    # ---- tasks
    def log_task(self, t: int, *, classes_seen: int, glob_correct: np.ndarray, glob_total: np.ndarray,
                 client_correct: np.ndarray, client_total: np.ndarray, n_train: np.ndarray,
                 seconds: float) -> dict:
        if t != self.tasks_done:
            raise RuntimeError(f"task {t} logged out of order (expected task {self.tasks_done})")
        gc, gt = np.asarray(glob_correct, np.int64), np.asarray(glob_total, np.int64)
        cc, ct = np.asarray(client_correct, np.int64), np.asarray(client_total, np.int64)
        if np.any(gc[t + 1:]) or np.any(gt[t + 1:]) or np.any(cc[:, t + 1:]) or np.any(ct[:, t + 1:]):
            raise RuntimeError(f"task {t}: future tasks must not be evaluated")
        if np.any(gc > gt) or np.any(cc > ct):
            raise RuntimeError(f"task {t}: more correct predictions than test samples")
        if t > 0 and (not np.array_equal(gt[:t], self.glob_total[:t]) or
                      not np.array_equal(ct[:, :t], self.client_total[:, :t])):
            raise RuntimeError(f"task {t}: the test sets of earlier tasks changed size")
        self.glob_correct[t], self.glob_total[:t + 1] = gc, gt[:t + 1]
        self.client_correct[:, t], self.client_total[:, :t + 1] = cc, ct[:, :t + 1]

        acc_seen = _ratio(gc[:t + 1].sum(), gt[:t + 1].sum())
        acc_new = _ratio(gc[t], gt[t])
        c_tot, c_cor = ct[:, :t + 1].sum(axis=1), cc[:, :t + 1].sum(axis=1)
        has = c_tot > 0
        c_acc = np.where(has, c_cor / np.maximum(c_tot, 1), np.nan)
        c_mean = float(np.mean(c_acc[has])) if has.any() else float("nan")
        c_min = float(np.min(c_acc[has])) if has.any() else float("nan")
        up, down = int(self.task_bytes[t, 0]), int(self.task_bytes[t, 1])
        row = {"task": t, "classes_seen": classes_seen, "acc_seen": acc_seen, "acc_task_new": acc_new,
               "client_acc_mean": c_mean, "client_acc_min": c_min, "clients_with_test": int(has.sum()),
               "seconds": seconds, "bytes_up": up, "bytes_down": down,
               "bytes_up_cum": int(self.task_bytes[:t + 1, 0].sum()),
               "bytes_down_cum": int(self.task_bytes[:t + 1, 1].sum())}
        if self.multi:
            row["rounds"], row["last_global_round"] = self.R, (t + 1) * self.R - 1
        self.task_rows.append(row)
        for k in range(self.K):
            crow = {"task": t, "client": k, "n_train_task": int(n_train[k]), "n_test_seen": int(c_tot[k]),
                    "acc_seen_own": float(c_acc[k]), "bytes_up": int(self.client_task_bytes[k, t, 0]),
                    "bytes_down": int(self.client_task_bytes[k, t, 1])}
            self.client_rows[k].append(crow)
            self.long_rows.append(crow)
        self.tasks_done += 1
        self.flush()
        return row

    def acc_seen_curve(self) -> np.ndarray:
        n = self.tasks_done
        return np.array([_ratio(self.glob_correct[t, :t + 1].sum(), self.glob_total[:t + 1].sum())
                         for t in range(n)])

    # ---- files
    def flush(self) -> None:
        task_header = ["task"] + (["rounds", "last_global_round"] if self.multi else []) + [
            "classes_seen", "acc_seen", "acc_task_new", "client_acc_mean", "client_acc_min", "clients_with_test",
            "seconds", "bytes_up", "bytes_down", "bytes_up_cum", "bytes_down_cum"]
        write_csv(os.path.join(self.run_dir, "tasks.csv"), task_header, self.task_rows)

        acc_rows = []
        for t in range(self.tasks_done):
            row = {"task": t, "acc_seen": self.task_rows[t]["acc_seen"]}
            for i in range(self.T):
                row[f"test_task_{i}"] = _ratio(self.glob_correct[t, i], self.glob_total[i]) if i <= t else None
            acc_rows.append(row)
        write_csv(os.path.join(self.run_dir, "acc_global.csv"),
                  ["task", "acc_seen"] + [f"test_task_{i}" for i in range(self.T)], acc_rows)

        client_header = ["task", "n_train_task", "n_test_seen", "acc_seen_own", "bytes_up", "bytes_down"]
        for k in range(self.K):
            write_csv(os.path.join(self.run_dir, "clients", f"client_{k}.csv"), client_header, self.client_rows[k])
        write_csv(os.path.join(self.run_dir, "acc_clients.csv"), ["task", "client"] + client_header[1:],
                  self.long_rows)

        comm_header = (["task", "phase", "round", "global_round", "client", "direction", "nbytes", "keys"]
                       if self.multi else ["task", "phase", "client", "direction", "nbytes", "keys"])
        write_csv(os.path.join(self.run_dir, "comm.csv"), comm_header, self.comm_rows)
        if self.multi:
            write_csv(os.path.join(self.run_dir, "rounds.csv"),
                      ["task", "round", "global_round", "clients_selected", "selected", "uploads", "train_loss",
                       "acc_seen", "seconds", "bytes_up", "bytes_down"], self.round_rows)

        tmp = os.path.join(self.run_dir, f"counts.tmp-{os.getpid()}.npz")
        np.savez(tmp, glob_correct=self.glob_correct, glob_total=self.glob_total,
                 client_correct=self.client_correct, client_total=self.client_total)
        os.replace(tmp, os.path.join(self.run_dir, "counts.npz"))

    def finalize(self, seconds_total: float) -> dict:
        if self.tasks_done != self.T:
            raise RuntimeError(f"finalize after {self.tasks_done} of {self.T} tasks")
        acc = self.acc_seen_curve()
        T = self.T
        c_tot = self.client_total.sum(axis=1)
        c_cor = self.client_correct[:, T - 1, :].sum(axis=1)
        has = c_tot > 0
        c_acc = c_cor[has] / c_tot[has]
        metrics = {
            "acc_final": float(acc[-1]),
            "acc_avg_incremental": float(np.mean(acc)),
            "client_acc_final_mean": float(np.mean(c_acc)) if has.any() else None,
            "client_acc_final_min": float(np.min(c_acc)) if has.any() else None,
            "clients_with_test_final": int(has.sum()),
            "bytes_up_total": int(self.task_bytes[:, 0].sum()),
            "bytes_down_total": int(self.task_bytes[:, 1].sum()),
            "seconds_total": float(seconds_total),
        }
        self.write_json("final.json", {"format": "fcl-final/1", "regime": self.regime, "run_flags": self.run_flags,
                                       "metrics": metrics, "definitions": METRIC_DEFINITIONS})
        self.set_status("complete")
        return metrics


# ---------------------------------------------------------------------------
# Averages over runs
# ---------------------------------------------------------------------------
def compute_cl_averages(exp_dir: str) -> dict:
    """avg/summary.json and avg/tasks.csv: mean and SD (ddof=1) of final values over all complete runs."""
    exp = read_json(os.path.join(exp_dir, "experiment.json"))
    expected = int(exp["settings"]["times"])
    runs, complete = [], []
    for i in range(expected):
        rd = os.path.join(exp_dir, f"times_{i}")
        status_path = os.path.join(rd, "status.json")
        status = read_json(status_path)["status"] if os.path.exists(status_path) else "missing"
        info = {"run": i, "status": status}
        if os.path.exists(os.path.join(rd, "run.json")):
            run = read_json(os.path.join(rd, "run.json"))
            info.update(seed=run["run_seed"], scenario=run["scenario"]["config_name"])
        runs.append(info)
        if status == "complete":
            complete.append(rd)

    metrics, task_rows = {}, []
    if complete:
        finals = [read_json(os.path.join(rd, "final.json"))["metrics"] for rd in complete]
        for name in finals[0]:
            vals = [f.get(name) for f in finals]
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in vals):
                continue
            arr = np.asarray(vals, dtype=np.float64)
            metrics[name] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                             "n": int(len(arr)), "values": [float(x) for x in vals]}
        curves = []
        for rd in complete:
            with open(os.path.join(rd, "tasks.csv"), newline="") as f:
                curves.append([float(r["acc_seen"]) for r in csv.DictReader(f)])
        if len({len(c) for c in curves}) == 1:
            cur = np.asarray(curves, dtype=np.float64)
            for t in range(cur.shape[1]):
                task_rows.append({"task": t, "acc_seen_mean": float(cur[:, t].mean()),
                                  "acc_seen_sd": float(cur[:, t].std(ddof=1)) if len(cur) > 1 else None,
                                  "n_runs": int(len(cur))})

    summary = {"format": "fcl-summary/1", "algorithm": exp["algorithm"], "regime": exp["regime"],
               "run_flags": exp["run_flags"], "n_runs_expected": expected, "n_runs_complete": len(complete),
               "complete": len(complete) == expected, "runs": runs,
               "rule": "mean and SD (ddof=1) over all complete runs; final values only, never a selected round",
               "metrics": metrics, "definitions": METRIC_DEFINITIONS, "created": now()}
    avg_dir = os.path.join(exp_dir, "avg")
    os.makedirs(avg_dir, exist_ok=True)
    write_json(os.path.join(avg_dir, "summary.json"), summary)
    write_csv(os.path.join(avg_dir, "tasks.csv"), ["task", "acc_seen_mean", "acc_seen_sd", "n_runs"], task_rows)
    return summary
