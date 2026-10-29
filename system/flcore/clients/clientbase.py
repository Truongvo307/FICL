"""Base client of FICL.

A client never decides when tasks start or what data it may read: the shared loop in
serverbase.py calls _begin_task(t), which moves the client's ClientData to task t (the
"new data arrived" event). From then on train_rows() gives the current task's rows only;
past tasks are closed unless the algorithm is an upper bound (RUN_FLAGS contains UPPER_BOUND).

Hooks an algorithm's client fills (all optional except train_round):
    on_begin_task(t)            new data has arrived (e.g. load the task's features)
    receive(t, r, payload)      a message from the server (r is None for task-end messages)
    train_round(t, r)           the local work: one statistics pass (single-round) or SGD epochs (multi-round)
    upload_payload(t, r)        the message to send back, or None to send nothing
    round_report()              optional training numbers for rounds.csv, e.g. {"train_loss": ..., "n": ...}
    on_end_task(t)              after the server's on_end_task (e.g. solve a personal head)
    receive_final(t, payload)   the model the server broadcasts for inference after task t
    predict(Z, seen)            only for personalized clients (PERSONALIZED = True)

Attribute names follow HtFL's Client where they exist (id, batch_size, learning_rate, local_epochs).
"""
from __future__ import annotations

import numpy as np


class Client:
    PERSONALIZED: bool = False     # True: evaluated with its own predict() instead of the server's global model

    def __init__(self, args, client_id: int, data, store, hp: dict, algo_seed: int):
        self.args = args
        self.id = int(client_id)
        self.data = data                 # utils.data_utils.ClientData (task-gated rows)
        self.store = store               # utils.data_utils.FeatureStore (read-only features, float64)
        self.hp = dict(hp)               # resolved algorithm hyperparameters
        self.num_classes = int(data.scenario.num_classes)
        self.local_epochs = getattr(args, "local_epochs", None)
        self.batch_size = getattr(args, "batch_size", None)
        self.learning_rate = getattr(args, "local_learning_rate", None)
        self._algo_seed = int(algo_seed)
        self.current_task = -1

    # ---- called by the shared loop only (algorithms must not override these two)
    def _begin_task(self, t: int) -> None:
        self.data.begin_task(t)
        self.current_task = t
        self.on_begin_task(t)

    def _end_task(self, t: int) -> None:
        self.on_end_task(t)

    # ---- helpers
    def rng(self, t: int, r: int = 0) -> np.random.Generator:
        """A generator that depends only on (algorithm seed, client, task, round): call order does not matter."""
        return np.random.default_rng([self._algo_seed, 0x434C4945, self.id, int(t), int(r)])

    def train_rows(self) -> np.ndarray:
        return self.data.train_rows()

    def num_train_samples(self) -> int:
        return int(len(self.data.train_rows()))

    def task_classes(self) -> np.ndarray:
        return self.data.task_classes()

    def seen_classes(self) -> np.ndarray:
        return self.data.seen_classes()

    def load_task_data(self) -> tuple[np.ndarray, np.ndarray]:
        """(features float64 [n, d], labels int64 [n]) of the current task's training rows."""
        rows = self.train_rows()
        return self.store.features(rows), self.store.labels(rows)

    # ---- hooks
    def on_begin_task(self, t: int) -> None:
        pass

    def receive(self, t: int, r, payload: dict) -> None:
        pass

    def train_round(self, t: int, r: int) -> None:
        raise NotImplementedError

    def upload_payload(self, t: int, r: int):
        return None

    def round_report(self) -> dict:
        return {}

    def on_end_task(self, t: int) -> None:
        pass

    def receive_final(self, t: int, payload: dict) -> None:
        pass

    def predict(self, Z: np.ndarray, seen: np.ndarray) -> np.ndarray:
        raise NotImplementedError("only personalized clients (PERSONALIZED = True) predict on their own")
