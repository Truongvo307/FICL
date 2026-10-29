"""Tests for the shared task loop (serverbase.py, clientbase.py) and the two smoke algorithms."""
import logging
import os

import numpy as np
import pytest

import helpers as H
from flcore.servers.serverbase import Server
from flcore.servers.serverclassmean import SmokeClassMean
from flcore.servers.serverfedavglin import SmokeFedAvgLinear
from utils.data_utils import TestDataAccessError


def _cfg(sr, name, **kw):
    return H.make_cfg(sr["root"], sr["results"], name, **kw)


def test_hook_order_single_round(synthetic_root, tmp_path):
    H.EVENTS.clear()
    server = H.build_server(H.RecordingSingle, _cfg(synthetic_root, "RecordingSingle"), str(tmp_path / "exp"))
    H.record_evaluations(server)
    server.train()
    assert H.EVENTS == H.expected_events(server, rounds=1)


def test_hook_order_multi_round(synthetic_root, tmp_path):
    H.EVENTS.clear()
    schedule = {**H.MULTI_SCHEDULE, "rounds_per_task": 3, "eval_gap": 2}
    server = H.build_server(H.RecordingMulti, _cfg(synthetic_root, "RecordingMulti", schedule=schedule),
                            str(tmp_path / "exp"))
    H.record_evaluations(server)
    server.train()
    assert H.EVENTS == H.expected_events(server, rounds=3, eval_gap=2)


def test_algorithms_fill_hooks_and_never_replace_the_loop():
    class OwnLoop(H.RecordingSingle):
        def train(self):
            pass

    class OwnRound(H.RecordingMulti):
        def _run_round(self, t, r):
            pass

    class NoRegime(Server):
        CLIENT_CLASS = H.RecordingClient

    class NotAClient(H.RecordingSingle):
        CLIENT_CLASS = object

    class UnknownFlag(H.RecordingSingle):
        RUN_FLAGS = ("BEST_ROUND",)

    class GatingClient(H.RecordingClient):
        def _begin_task(self, t):
            pass

    class BypassGating(H.RecordingSingle):
        CLIENT_CLASS = GatingClient

    for cls, word in ((OwnLoop, "train"), (OwnRound, "_run_round"), (NoRegime, "REGIME"),
                      (NotAClient, "CLIENT_CLASS"), (UnknownFlag, "RUN_FLAGS"), (BypassGating, "_begin_task")):
        with pytest.raises(TypeError, match=word):
            cls.check_contract()


def test_single_round_server_refuses_more_rounds(synthetic_root, tmp_path):
    args = H.build_args(H.RecordingSingle, _cfg(synthetic_root, "RecordingSingle"), str(tmp_path / "exp"))
    args.rounds_per_task = 3          # bypasses the config checks on purpose
    with pytest.raises(ValueError, match="single-round"):
        H.RecordingSingle(args, 0)


class _PeekingClient(H.RecordingClient):
    def train_round(self, t, r):
        self.store.features(self.data.test_rows(t))      # training code must not read test data


class _PeekingServer(H.RecordingSingle):
    CLIENT_CLASS = _PeekingClient


def test_training_code_cannot_read_test_data(synthetic_root, tmp_path):
    server = H.build_server(_PeekingServer, _cfg(synthetic_root, "_PeekingServer"), str(tmp_path / "exp"))
    with pytest.raises(TestDataAccessError):
        server.train()
    assert not os.path.exists(os.path.join(server.run_dir, "final.json"))      # not complete
    assert "failed during task 0: TestDataAccessError" in _log_text(server.run_dir)


def _log_text(run_dir):
    logs = [f for f in os.listdir(run_dir) if f.endswith(".log")]
    assert len(logs) == 1
    with open(os.path.join(run_dir, logs[0])) as f:
        return f.read()


class _BrokenInit(H.RecordingSingle):
    def init_algorithm(self):
        raise RuntimeError("model setup failed")


def test_a_run_that_fails_during_setup_leaves_the_error_in_its_log(synthetic_root, tmp_path):
    exp = str(tmp_path / "exp")
    with pytest.raises(RuntimeError, match="model setup failed"):
        H.build_server(_BrokenInit, _cfg(synthetic_root, "_BrokenInit"), exp)
    run_dir = os.path.join(exp, "times_0")
    assert not os.path.exists(os.path.join(run_dir, "final.json"))
    text = _log_text(run_dir)
    assert "failed during setup: RuntimeError: model setup failed" in text and "Traceback" in text


class _FutureServer(H.RecordingSingle):
    def predict(self, Z, seen):
        return np.full(len(Z), self.scenario.tasks[-1][0], dtype=np.int64)    # a class that has not arrived


def test_predicting_unseen_classes_is_refused(synthetic_root, tmp_path):
    server = H.build_server(_FutureServer, _cfg(synthetic_root, "_FutureServer"), str(tmp_path / "exp"))
    with pytest.raises(RuntimeError, match="not seen"):
        server.train()


def _central_stats(sr):
    index = sr["index"]
    feats = np.load(sr["cache_file"], mmap_mode="r")[:, 3, 0, :].astype(np.float64)
    train = index.split == 0
    sums = np.stack([feats[train & (index.labels == c)].sum(axis=0) for c in range(index.num_classes)])
    counts = np.array([np.count_nonzero(train & (index.labels == c)) for c in range(index.num_classes)])
    return sums, counts


class _SnapshotClassMean(SmokeClassMean):
    def init_algorithm(self):
        super().init_algorithm()
        self.snapshots = []

    def on_end_task(self, t):
        super().on_end_task(t)
        self.snapshots.append(self.counts.copy())


def test_classmean_federated_equals_centralized_for_any_client_split(synthetic_root, tmp_path):
    sums, counts = _central_stats(synthetic_root)
    servers = [H.run_server(_SnapshotClassMean, _cfg(synthetic_root, "_SnapshotClassMean", scenario=scen),
                            str(tmp_path / f"exp{i}"))
               for i, scen in enumerate((H.DIR_SCENARIO, H.NATURAL_SCENARIO))]
    for s in servers:
        assert np.array_equal(s.counts, counts)
        assert np.max(np.abs(s.sums - sums)) <= 1e-12 * np.max(np.abs(sums))
        for t, snap in enumerate(s.snapshots):          # after task t: exactly the classes seen so far
            seen = s.scenario.seen_classes(t)
            assert np.array_equal(snap[seen], counts[seen])
            assert np.all(np.delete(snap, seen) == 0)
    a, b = servers
    assert np.array_equal(a.scenario.class_order, b.scenario.class_order)    # same seed, same tasks
    ma, mb = a.class_means(np.arange(len(counts))), b.class_means(np.arange(len(counts)))
    assert np.max(np.abs(ma - mb)) <= 1e-12 * np.max(np.abs(ma))


def test_bytes_are_logged_for_every_transfer(synthetic_root, tmp_path):
    schedule = {**H.MULTI_SCHEDULE, "rounds_per_task": 2}
    s = H.run_server(SmokeFedAvgLinear, _cfg(synthetic_root, "SmokeFedAvgLinear", schedule=schedule),
                     str(tmp_path / "exp"))
    _, comm = H.read_csv(os.path.join(s.run_dir, "comm.csv"))
    _, tasks = H.read_csv(os.path.join(s.run_dir, "tasks.csv"))
    d, K = s.store.dim, s.num_clients
    for t in range(s.num_tasks):
        m = len(s.scenario.seen_classes(t))
        model = 8 * m + 4 * d * m + 4 * m              # class ids (int64) + W and b (fp32 on the wire)
        upload = 4 * d * m + 4 * m + 8                 # W, b and the sample count
        with_data = sorted(k for k in range(K) if len(s.scenario.train_rows[t][k]) > 0)
        for r in range(2):
            rows = [x for x in comm if int(x["task"]) == t and x["phase"] == "round" and x["round"] == str(r)]
            down = [x for x in rows if x["direction"] == "down"]
            up = [x for x in rows if x["direction"] == "up"]
            assert sorted(int(x["client"]) for x in down) == list(range(K))
            assert all(int(x["nbytes"]) == model for x in down)
            assert sorted(int(x["client"]) for x in up) == with_data
            assert all(int(x["nbytes"]) == upload for x in up)
        end = [x for x in comm if int(x["task"]) == t and x["phase"] == "task_end"]
        assert sorted(int(x["client"]) for x in end) == list(range(K))
        assert all(int(x["nbytes"]) == model for x in end)
        in_task = [x for x in comm if int(x["task"]) == t]
        assert int(tasks[t]["bytes_up"]) == sum(int(x["nbytes"]) for x in in_task if x["direction"] == "up")
        assert int(tasks[t]["bytes_down"]) == sum(int(x["nbytes"]) for x in in_task if x["direction"] == "down")


def test_partial_participation_is_seeded(synthetic_root, tmp_path):
    schedule = {**H.MULTI_SCHEDULE, "rounds_per_task": 4, "join_ratio": 0.4}
    cfg = _cfg(synthetic_root, "SmokeFedAvgLinear", schedule=schedule)
    a = H.run_server(SmokeFedAvgLinear, cfg, str(tmp_path / "a"))
    b = H.run_server(SmokeFedAvgLinear, cfg, str(tmp_path / "b"))
    sel_a = [x["selected"] for x in H.read_csv(os.path.join(a.run_dir, "rounds.csv"))[1]]
    sel_b = [x["selected"] for x in H.read_csv(os.path.join(b.run_dir, "rounds.csv"))[1]]
    assert sel_a == sel_b
    assert all(len(s.split()) == 2 for s in sel_a)          # int(5 * 0.4) clients per round
    assert len(set(sel_a)) > 1


def test_multi_round_run_is_reproducible(synthetic_root, tmp_path):
    cfg = _cfg(synthetic_root, "SmokeFedAvgLinear", schedule={**H.MULTI_SCHEDULE, "eval_gap": 1})
    a = H.run_server(SmokeFedAvgLinear, cfg, str(tmp_path / "a"))
    b = H.run_server(SmokeFedAvgLinear, cfg, str(tmp_path / "b"))
    for key in ("glob_correct", "glob_total", "client_correct", "client_total"):
        assert np.array_equal(getattr(a.results, key), getattr(b.results, key)), key
    for name in ("acc_global.csv", "acc_clients.csv"):
        assert H.read_csv(os.path.join(a.run_dir, name)) == H.read_csv(os.path.join(b.run_dir, name)), name

    def no_time(rows):
        return [{k: v for k, v in x.items() if k != "seconds"} for x in rows]

    assert no_time(H.read_csv(os.path.join(a.run_dir, "rounds.csv"))[1]) == \
        no_time(H.read_csv(os.path.join(b.run_dir, "rounds.csv"))[1])
    assert H.read_csv(os.path.join(a.run_dir, "comm.csv")) == H.read_csv(os.path.join(b.run_dir, "comm.csv"))


def test_smoke_algorithms_learn_on_synthetic_data(synthetic_root, tmp_path):
    ncm = H.run_server(SmokeClassMean, _cfg(synthetic_root, "SmokeClassMean"), str(tmp_path / "ncm"))
    assert H.read_json(os.path.join(ncm.run_dir, "final.json"))["metrics"]["acc_final"] > 0.8
    schedule = {**H.MULTI_SCHEDULE, "rounds_per_task": 10}
    fed = H.run_server(SmokeFedAvgLinear, _cfg(synthetic_root, "SmokeFedAvgLinear", schedule=schedule),
                       str(tmp_path / "fed"))
    _, tasks = H.read_csv(os.path.join(fed.run_dir, "tasks.csv"))
    assert float(tasks[0]["acc_task_new"]) > 0.8                 # it learns the first task


def test_the_run_log_is_written_and_closed(synthetic_root, tmp_path):
    server = H.build_server(SmokeClassMean, _cfg(synthetic_root, "SmokeClassMean"), str(tmp_path / "exp"))
    files = [h for h in server.logger.handlers if isinstance(h, logging.FileHandler)]
    server.train()
    assert len(files) == 1 and files[0].stream is None                  # closed when the run ended
    assert not any(isinstance(h, logging.FileHandler) for h in server.logger.handlers)
    logs = [f for f in os.listdir(server.run_dir) if f.endswith(".log")]
    assert logs == ["Synthetic_SmokeClassMean_test_run0.log"]
    with open(os.path.join(server.run_dir, logs[0])) as f:
        text = f.read()
    assert "class_per_task: 2" in text and f"task {server.num_tasks - 1}:" in text and "complete" in text
