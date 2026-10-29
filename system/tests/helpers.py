"""Shared helpers for the FICL loop tests (imported by the test files, not collected as tests)."""
import csv
import json
import os

import numpy as np

from flcore.clients.clientbase import Client
from flcore.servers.serverbase import Server
from utils.config_utils import MULTI_ROUND, SINGLE_ROUND, config_from_dict, finalize_config, merge_sources

DIR_SCENARIO = {"dataset": "Synthetic", "partition": "dir", "num_clients": 5, "alpha": 0.5, "class_per_task": 2}
NATURAL_SCENARIO = {"dataset": "Synthetic", "partition": "natural", "class_per_task": 2}
MULTI_SCHEDULE = {"rounds_per_task": 3, "local_epochs": 1, "batch_size": 16, "learning_rate": 0.01}
IMAGE_SCHEDULE = {"rounds_per_task": 2, "local_epochs": 1, "batch_size": 16, "learning_rate": 0.05}

EVENTS = []


def make_cfg(root, results, name, scenario=None, features=None, schedule=None, run=None, hparams=None):
    """A config dict with the same sections as a YAML file, pointing at the synthetic dataset."""
    return {
        "scenario": {**(scenario or DIR_SCENARIO), "dataset_root": root},
        "features": {"backbone": "synthetic", "group": "pre", "blocks": [3], "tokens": ["cls"], **(features or {})},
        "schedule": dict(schedule or {}),
        "algorithm": {"name": name, "hparams": dict(hparams or {})},
        "run": {"seed": 0, "times": 1, "goal": "test", "device": "cpu", "results_root": results, **(run or {})},
    }


def make_image_cfg(root, results, name, model=None, scenario=None, schedule=None, run=None, hparams=None):
    """A config dict for an image algorithm: the model section instead of features (tiny CNN by default)."""
    return {
        "scenario": {**(scenario or DIR_SCENARIO), "dataset_root": root},
        "model": {"backbone": "tiny_cnn", **(model or {})},
        "schedule": dict(schedule or IMAGE_SCHEDULE),
        "algorithm": {"name": name, "hparams": dict(hparams or {})},
        "run": {"seed": 0, "times": 1, "goal": "test", "device": "cpu", "results_root": results, "num_workers": 0,
                **(run or {})},
    }


def build_args(server_cls, cfg, exp_dir):
    values, hparams = config_from_dict(cfg, "test")
    v, s, h, hs = merge_sources({}, {}, values, hparams, "test")
    args = finalize_config(v, s, h, hs, server_cls)
    args.exp_dir = exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    return args


def build_server(server_cls, cfg, exp_dir, run_index=0):
    return server_cls(build_args(server_cls, cfg, exp_dir), run_index)


def run_server(server_cls, cfg, exp_dir, run_index=0):
    server = build_server(server_cls, cfg, exp_dir, run_index)
    server.train()
    return server


def load_dataset_raw_utils():
    """dataset/utils/raw_utils.py, loaded by path like the fixtures in conftest.py do."""
    from conftest import load_dataset_module

    return load_dataset_module("raw_utils.py", "_ds_raw_utils")


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_json(path):
    with open(path) as f:
        return json.load(f)


def record_evaluations(server):
    """Wrap the server's evaluation steps so they appear in EVENTS (instance attributes only)."""
    task_eval, round_eval = server._evaluate_task, server._evaluate_round

    def _task(t):
        EVENTS.append(("evaluate_task", t))
        return task_eval(t)

    def _round(t, r):
        EVENTS.append(("evaluate_round", t, r))
        return round_eval(t, r)

    server._evaluate_task = _task
    server._evaluate_round = _round


class RecordingClient(Client):
    def on_begin_task(self, t):
        EVENTS.append(("client.begin_task", self.id, t))
        self._payload = None

    def receive(self, t, r, payload):
        EVENTS.append(("client.receive", self.id, t, r))

    def train_round(self, t, r):
        EVENTS.append(("client.train_round", self.id, t, r))
        rows = self.train_rows()
        labels = self.store.labels(rows)
        assert np.isin(labels, self.task_classes()).all(), "a client saw training data of another task"
        self._payload = {"n": len(rows)} if len(rows) else None

    def upload_payload(self, t, r):
        EVENTS.append(("client.upload", self.id, t, r))
        return self._payload

    def on_end_task(self, t):
        EVENTS.append(("client.end_task", self.id, t))


class _RecordingServer(Server):
    CLIENT_CLASS = RecordingClient

    def init_algorithm(self):
        EVENTS.append(("server.init",))

    def on_begin_task(self, t):
        EVENTS.append(("server.begin_task", t))

    def download_payload(self, t, r, client):
        EVENTS.append(("server.download", t, r, client.id))
        return {"w": np.zeros(3)} if self.regime == MULTI_ROUND else None

    def aggregate(self, t, r, uploads):
        EVENTS.append(("server.aggregate", t, r, tuple(cid for cid, _ in uploads)))

    def on_end_task(self, t):
        EVENTS.append(("server.end_task", t))

    def predict(self, Z, seen):
        return np.full(len(Z), seen[0], dtype=np.int64)


class RecordingSingle(_RecordingServer):
    REGIME = SINGLE_ROUND


class RecordingMulti(_RecordingServer):
    REGIME = MULTI_ROUND


def expected_events(server, rounds, eval_gap=0):
    """The exact hook sequence the shared loop must produce for the recording algorithms."""
    scn, K = server.scenario, server.num_clients
    multi = server.regime == MULTI_ROUND
    ev = [("server.init",)]
    for t in range(scn.num_tasks):
        ev += [("client.begin_task", k, t) for k in range(K)] + [("server.begin_task", t)]
        with_data = tuple(k for k in range(K) if len(scn.train_rows[t][k]) > 0)
        for r in range(rounds):
            for k in range(K):
                ev.append(("server.download", t, r, k))
                if multi:
                    ev.append(("client.receive", k, t, r))
                ev.append(("client.train_round", k, t, r))
            ev += [("client.upload", k, t, r) for k in range(K)]
            ev.append(("server.aggregate", t, r, with_data))
            if multi and eval_gap and (r + 1) % eval_gap == 0:
                ev.append(("evaluate_round", t, r))
        ev += [("server.end_task", t)] + [("client.end_task", k, t) for k in range(K)] + [("evaluate_task", t)]
    return ev
