"""Tests for the result files (utils/result_utils.py), checked through real runs on synthetic data."""
import os

import numpy as np

import helpers as H
from flcore.servers.serverclassmean import SmokeClassMean
from flcore.servers.serverfedavglin import SmokeFedAvgLinear


def _single(sr, tmp_path):
    return H.run_server(SmokeClassMean, H.make_cfg(sr["root"], sr["results"], "SmokeClassMean"),
                        str(tmp_path / "single"))


def _multi(sr, tmp_path, rounds=4, eval_gap=2):
    schedule = {**H.MULTI_SCHEDULE, "rounds_per_task": rounds, "eval_gap": eval_gap}
    return H.run_server(SmokeFedAvgLinear, H.make_cfg(sr["root"], sr["results"], "SmokeFedAvgLinear",
                                                      schedule=schedule), str(tmp_path / "multi"))


def test_single_round_outputs_have_no_round_concept(synthetic_root, tmp_path):
    s = _single(synthetic_root, tmp_path)
    assert not os.path.exists(os.path.join(s.run_dir, "rounds.csv"))
    for name in ("tasks.csv", "comm.csv", "acc_global.csv", "acc_clients.csv"):
        header, _ = H.read_csv(os.path.join(s.run_dir, name))
        assert not any("round" in h for h in header), (name, header)
    _, comm = H.read_csv(os.path.join(s.run_dir, "comm.csv"))
    assert {x["phase"] for x in comm} <= {"exchange", "task_end"}
    run = H.read_json(os.path.join(s.run_dir, "run.json"))
    assert run["regime"] == "single_round" and run["rounds_per_task"] is None and run["total_rounds"] is None


def test_multi_round_outputs_have_round_curves(synthetic_root, tmp_path):
    s = _multi(synthetic_root, tmp_path, rounds=4, eval_gap=2)
    _, rounds = H.read_csv(os.path.join(s.run_dir, "rounds.csv"))
    T = s.num_tasks
    assert [(int(x["task"]), int(x["round"])) for x in rounds] == [(t, r) for t in range(T) for r in range(4)]
    assert all(int(x["global_round"]) == 4 * int(x["task"]) + int(x["round"]) for x in rounds)
    evaluated = [(int(x["task"]), int(x["round"])) for x in rounds if x["acc_seen"] != ""]
    assert evaluated == [(t, r) for t in range(T) for r in (1, 3)]
    _, tasks = H.read_csv(os.path.join(s.run_dir, "tasks.csv"))
    assert all(int(x["rounds"]) == 4 for x in tasks)
    assert [int(x["last_global_round"]) for x in tasks] == [4 * t + 3 for t in range(T)]


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def test_no_best_round_anywhere(synthetic_root, tmp_path):
    for s in (_single(synthetic_root, tmp_path), _multi(synthetic_root, tmp_path)):
        for dirpath, _dirs, files in os.walk(s.run_dir):
            for fn in files:
                path = os.path.join(dirpath, fn)
                if fn.endswith(".csv"):
                    header, _ = H.read_csv(path)
                    assert not any("best" in h.lower() for h in header), path
                elif fn.endswith(".json"):
                    assert not any("best" in k.lower() for k in _all_keys(H.read_json(path))), path


def test_accuracy_matrix_and_counts_agree(synthetic_root, tmp_path):
    s = _single(synthetic_root, tmp_path)
    T, K = s.num_tasks, s.num_clients
    gc, gt = s.results.glob_correct, s.results.glob_total            # the exact counts, kept in memory
    cc, ct = s.results.client_correct, s.results.client_total
    assert not os.path.exists(os.path.join(s.run_dir, "counts.npz"))
    assert not os.path.exists(os.path.join(s.run_dir, "status.json"))
    assert gc.shape == (T, T) and cc.shape == (K, T, T) and ct.shape == (K, T)
    upper = np.triu_indices(T, 1)
    assert np.all(gc[upper] == 0) and np.all(cc[:, upper[0], upper[1]] == 0)
    assert np.array_equal(gt, [len(r) for r in s.scenario.test_global_rows])
    assert np.array_equal(ct.sum(axis=0), gt)                 # client shards partition the test set
    assert np.array_equal(cc.sum(axis=0), gc)                 # one global model: both evaluations agree
    _, acc = H.read_csv(os.path.join(s.run_dir, "acc_global.csv"))
    for t, row in enumerate(acc):
        for i in range(T):
            cell = row[f"test_task_{i}"]
            if i > t:
                assert cell == ""
            else:
                assert abs(float(cell) - gc[t, i] / gt[i]) < 1e-6
        assert abs(float(row["acc_seen"]) - gc[t, :t + 1].sum() / gt[:t + 1].sum()) < 1e-6
    _, per_client = H.read_csv(os.path.join(s.run_dir, "acc_clients.csv"))
    assert len(per_client) == T * K
