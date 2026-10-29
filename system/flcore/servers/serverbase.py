"""Base server of FICL: one task loop for every algorithm.

    train():
        for t in tasks:                        # a task is an event: the classes of task t arrive at the clients
            _begin_task(t)                     #   every client's data moves to task t, then on_begin_task(t)
            for r in range(rounds_per_task):   #   single_round: exactly one exchange; multi_round: -rpt rounds
                _run_round(t, r)               #     select -> download -> client.train_round -> upload -> aggregate
                                               #     multi_round: curve point every eval_gap rounds
            _end_task(t)                       #   on_end_task(t), clients' on_end_task(t), final broadcast
            _evaluate_task(t)                  #   accuracy-matrix row t and each client's own test shards
        _end_run()                             #   final.json (the run is complete)

An algorithm subclasses Server and Client and fills the hooks below; it never writes its own
loop (check_contract() refuses a class that overrides the loop). Class attributes it sets:
    REGIME         SINGLE_ROUND (closed-form: one exchange per task, no rounds) or MULTI_ROUND
    INPUT          "features" (cached features of a frozen backbone; self.store is a FeatureStore) or
                   "images" (raw images; self.store is an ImageStore, see flcore/servers/serverimage.py)
    CLIENT_CLASS   its Client subclass
    HPARAMS        {name: HParam(kind, default, help, choices, low, high)}: its own hyperparameters. They
                   become --NAME flags (python main.py -algo NAME --help), YAML algorithm.hparams, and
                   self.hp["NAME"] on the server and every client; unknown names are refused
    RUN_FLAGS      e.g. ("SMOKE_TEST",); UPPER_BOUND opens past training data, DIAGNOSTIC_ORACLE marks
                   the only arm that may look at test data; flags go into the folder name and into
                   experiment.json, run.json, final.json and avg/summary.json
    USES_TORCH     True if it needs torch (then -dev / -did matter)

Server hooks: init_algorithm(), on_begin_task(t), download_payload(t, r, client), aggregate(t, r, uploads)
(required), on_end_task(t) (may call self.send(client, t, payload)), final_broadcast_payload(t),
predict(X, seen) (required: the global model; X is a batch of features or of images).
Class method check_hparams(hp, sources): extra checks of the resolved hyperparameters (raise ConfigError).

The structure follows HtFLlib / PFLlib's Server (set_clients, select_clients, send / receive,
aggregate, evaluate) with these differences: one loop for every algorithm instead of a train() per
algorithm; tasks around rounds; clients see only the current task's training rows; test features
can be read only inside evaluation; every message is counted in bytes; randomness comes from
seeded generators (no global RNG state); evaluation after every task with final values only
(no best round, no auto_break on test accuracy).
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback
import zlib

import numpy as np

from flcore.clients.clientbase import Client
from utils.comm_utils import WIRE_RULE, payload_keys, wire_nbytes
from utils.config_utils import (
    FEATURES,
    INPUTS,
    MULTI_ROUND,
    REGIMES,
    SGD_SCHEDULE,
    HParam,
    check_hparam_names,
    model_summary,
)
from utils.data_utils import (
    DataConfig,
    ClientData,
    FeatureStore,
    FeatureView,
    check_compatible,
    load_scenario,
    set_data_config,
)
from utils.log_utils import close_log_files, log_config, setup_logger
from utils.result_utils import ResultSaver

KNOWN_RUN_FLAGS = ("SMOKE_TEST", "REFERENCE", "UPPER_BOUND", "DIAGNOSTIC_ORACLE")
LOOP_METHODS = ("train", "set_clients", "send", "_begin_task", "_run_round", "_end_task", "_evaluate_task",
                "_evaluate_round", "_end_run", "_log_comm", "_predict_rows")
CLIENT_BASE_METHODS = ("_begin_task", "_end_task")


def derive_run_seeds(run_seed: int) -> dict:
    """Independent seeds for the algorithm, client selection and feature projections of one run."""
    children = np.random.SeedSequence([int(run_seed), 0x46434C]).spawn(3)
    return {name: int(c.generate_state(1)[0]) for name, c in zip(("algorithm", "selection", "projection"), children)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Server:
    REGIME: str = ""
    INPUT: str = FEATURES
    CLIENT_CLASS: type = Client
    HPARAMS: dict = {}
    RUN_FLAGS: tuple = ()
    USES_TORCH: bool = False
    EVAL_CHUNK: int = 8192

    # ------------------------------------------------------------------ declarations
    @classmethod
    def schedule_required(cls) -> tuple:
        """Schedule settings the algorithm needs. Multi-round with local SGD: rounds, epochs, batch size, lr."""
        return SGD_SCHEDULE if cls.REGIME == MULTI_ROUND else ()

    @classmethod
    def schedule_allowed(cls) -> tuple:
        if cls.REGIME != MULTI_ROUND:
            return ()
        return tuple(dict.fromkeys(("rounds_per_task",) + tuple(cls.schedule_required()) + ("join_ratio", "eval_gap")))

    @classmethod
    def check_contract(cls) -> None:
        """Refuse algorithm classes that would change the shared protocol."""
        name = cls.__name__
        if cls.REGIME not in REGIMES:
            raise TypeError(f"{name}.REGIME must be one of {REGIMES}, got {cls.REGIME!r}")
        if cls.INPUT not in INPUTS:
            raise TypeError(f"{name}.INPUT must be one of {INPUTS}, got {cls.INPUT!r}")
        client = cls.CLIENT_CLASS
        if not (isinstance(client, type) and issubclass(client, Client)) or client is Client:
            raise TypeError(f"{name}.CLIENT_CLASS must be a subclass of Client")
        for m in LOOP_METHODS:
            if getattr(cls, m) is not getattr(Server, m):
                raise TypeError(f"{name} overrides {m}(); algorithms fill the hooks, the task loop is shared")
        for m in CLIENT_BASE_METHODS:
            if getattr(client, m) is not getattr(Client, m):
                raise TypeError(f"{client.__name__} overrides {m}(); task gating of client data is shared")
        bad = [f for f in cls.RUN_FLAGS if f not in KNOWN_RUN_FLAGS]
        if bad:
            raise TypeError(f"{name}.RUN_FLAGS has unknown flags {bad}; known flags: {KNOWN_RUN_FLAGS}")
        for k, v in cls.HPARAMS.items():
            if not isinstance(v, HParam):
                raise TypeError(f"{name}.HPARAMS[{k!r}] must be an HParam")
        check_hparam_names(cls.HPARAMS, f"{name}.HPARAMS")
        for m in ("aggregate", "predict"):
            if getattr(cls, m) is getattr(Server, m):
                raise TypeError(f"{name} must implement {m}()")
        if client.train_round is Client.train_round:
            raise TypeError(f"{client.__name__} must implement train_round()")
        if client.PERSONALIZED and client.predict is Client.predict:
            raise TypeError(f"{client.__name__} is PERSONALIZED and must implement predict()")

    # ------------------------------------------------------------------ setup
    def __init__(self, args, run_index: int):
        cls = type(self)
        cls.check_contract()
        self.args = args
        self.algorithm = args.algorithm
        self.regime = cls.REGIME
        self.multi = self.regime == MULTI_ROUND
        self.run_index = int(run_index)
        self.run_seed = int(args.seed) + self.run_index
        self.seeds = derive_run_seeds(self.run_seed)
        self.hp = dict(getattr(args, "hparams", {}) or {})
        self.run_flags = tuple(getattr(args, "run_flags", None) or cls.RUN_FLAGS)
        if not set(cls.RUN_FLAGS) <= set(self.run_flags):
            raise ValueError(f"args.run_flags {self.run_flags} lacks the flags of {cls.__name__} {cls.RUN_FLAGS}")
        self.past_train_access = "UPPER_BOUND" in self.run_flags

        # schedule: how time runs inside a task
        rpt = getattr(args, "rounds_per_task", None)
        if self.multi:
            if rpt is None or int(rpt) < 1:
                raise ValueError(f"{self.algorithm} is multi-round and needs rounds_per_task >= 1")
            self.rounds_per_task = int(rpt)
            self.eval_gap = int(getattr(args, "eval_gap", 0) or 0)
            jr = getattr(args, "join_ratio", None)
            self.join_ratio = 1.0 if jr is None else float(jr)
        else:
            if rpt not in (None, 1):
                raise ValueError(f"{self.algorithm} is single-round (closed-form): exactly one exchange per task, "
                                 f"got rounds_per_task={rpt}")
            self.rounds_per_task, self.eval_gap, self.join_ratio = 1, 0, 1.0
        self.local_epochs = getattr(args, "local_epochs", None)
        self.batch_size = getattr(args, "batch_size", None)
        self.learning_rate = getattr(args, "local_learning_rate", None)

        # the scenario of this run: scenario seed = run seed
        self.data_config = DataConfig(
            dataset=args.dataset, partition=args.partition, num_clients=args.num_clients or 0,
            alpha=args.data_alpha if args.data_alpha is not None else 0.0, class_per_task=args.class_per_task,
            seed=self.run_seed, use_val=args.use_val, min_train_per_client=args.min_train_per_client,
            dataset_root=args.dataset_root)
        set_data_config(self.data_config)
        self.scenario = load_scenario(self.data_config)
        self.num_clients = self.scenario.num_clients
        self.num_tasks = self.scenario.num_tasks
        self.num_classes = self.scenario.num_classes

        # what the clients read: cached features or raw images
        self.view, self.proj_seed = None, None
        if cls.INPUT == FEATURES:
            self.proj_seed = args.proj_seed
            if args.proj_rank is not None and self.proj_seed is None:
                self.proj_seed = self.seeds["projection"]
            self.view = FeatureView(group=args.feature_group,
                                    blocks=tuple(args.blocks) if args.blocks is not None else None,
                                    tokens=tuple(args.tokens) if args.tokens is not None else None,
                                    proj_rank=args.proj_rank, proj_seed=self.proj_seed)
            self.store = FeatureStore(args.dataset, args.backbone, self.view, dataset_root=args.dataset_root)
        else:
            from utils.image_utils import ImageStore

            self.store = ImageStore(args.dataset, dataset_root=args.dataset_root, num_workers=args.num_workers,
                                    eval_batch_size=args.eval_batch_size,
                                    pin_memory=str(getattr(args, "device_used", args.device)).startswith("cuda"))
        check_compatible(self.scenario, self.store)

        # outputs
        exp_dir = getattr(args, "exp_dir", None)
        if not exp_dir:
            raise ValueError("args.exp_dir is not set (main.py creates the experiment folder)")
        self.run_dir = os.path.join(exp_dir, f"times_{self.run_index}")
        if os.path.exists(self.run_dir):
            raise FileExistsError(f"{self.run_dir} already exists; results are never overwritten")
        os.makedirs(self.run_dir)
        self.results = ResultSaver(self.run_dir, self.regime, self.num_tasks, self.num_clients,
                                     self.rounds_per_task if self.multi else None, self.run_flags)
        self.logger = setup_logger(args, self.run_index, self.run_dir)
        self.clients: list[Client] = []
        self.current_task = -1
        self._task_t0 = time.time()
        try:
            log_config(self.logger, args)
            self.set_clients(cls.CLIENT_CLASS)
            self.results.write_json("run.json", self.run_info())
            self.init_algorithm()
            extra = self.extra_run_info()
            if extra:                              # what only the algorithm knows, e.g. the network built
                self.results.write_json("run.json", {**self.run_info(), **extra})
        except BaseException as exc:
            self.logger.error(f"run {self.run_index} failed during setup: {type(exc).__name__}: {exc}\n"
                              f"{traceback.format_exc()}")
            close_log_files(self.logger)
            raise

    def set_clients(self, clientObj) -> None:
        """One client per scenario client, each with its own task-gated ClientData."""
        for k in range(self.num_clients):
            data = ClientData(self.scenario, k, past_train_access=self.past_train_access)
            self.clients.append(clientObj(self.args, k, data, self.store, self.hp, self.seeds["algorithm"]))

    def run_info(self) -> dict:
        T, R = self.num_tasks, self.rounds_per_task
        return {
            "format": "fcl-run/1", "run_index": self.run_index, "run_seed": self.run_seed, "seeds": self.seeds,
            "algorithm": self.algorithm, "algorithm_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "regime": self.regime, "run_flags": list(self.run_flags), "past_train_access": self.past_train_access,
            "personalized": bool(type(self).CLIENT_CLASS.PERSONALIZED),
            "rounds_per_task": R if self.multi else None, "total_rounds": T * R if self.multi else None,
            "eval_gap": self.eval_gap if self.multi else None, "join_ratio": self.join_ratio if self.multi else None,
            "local_epochs": self.local_epochs if self.multi else None,
            "batch_size": self.batch_size if self.multi else None,
            "learning_rate": self.learning_rate if self.multi else None,
            "hparams": self.hp,
            "scenario": {
                "config_name": self.scenario.config_name, "config_dir": os.path.abspath(self.scenario.config_dir),
                "config_sha256": _sha256(os.path.join(self.scenario.config_dir, "config.json")),
                "content_sha256": self.scenario.content_sha256,
                "raw_index_hash": self.scenario.raw_index_hash, "num_tasks": T, "num_clients": self.num_clients,
                "num_classes": self.num_classes, "class_order": self.scenario.class_order.tolist(),
                "tasks": [c.tolist() for c in self.scenario.tasks]},
            "input": type(self).INPUT,
            **self._input_info(),
            "wire_rule": WIRE_RULE,
        }

    def _input_info(self) -> dict:
        if type(self).INPUT == FEATURES:
            man = self.store.manifest
            return {"features": {
                "backbone": self.args.backbone, "group": self.view.group, "blocks": self.store.blocks,
                "tokens": self.store.tokens, "dim": self.store.dim, "depth_groups": self.store.depth_groups,
                "proj_rank": self.view.proj_rank, "proj_seed": self.proj_seed, "final_ln": man.get("final_ln"),
                "cache_dir": os.path.abspath(self.store.group_dir), "cache_sha256": man.get("sha256")}}
        a = self.args
        keys = ("model", "pretrained", "trainable", "last_k", "lora_rank", "lora_alpha", "lora_targets",
                "prompt_length", "prompt_deep", "image_size", "augment", "num_workers", "eval_batch_size")
        return {"model": {"summary": model_summary(a), **{k: getattr(a, k, None) for k in keys},
                          "images": os.path.abspath(os.path.join(a.dataset_root, a.dataset, "rawdata")),
                          "index_sha256": self.store.raw_index_hash}}

    def extra_run_info(self) -> dict:
        """Entries the algorithm adds to run.json after init_algorithm() (default: none)."""
        return {}

    # ------------------------------------------------------------------ the shared loop
    def train(self) -> None:
        """The shared task loop. The run is complete when final.json is written (in _end_run); a failure
        or an interruption is written to the run's log, and the run then has no final.json."""
        t_run = time.time()
        try:
            for t in range(self.num_tasks):
                self._begin_task(t)
                for r in range(self.rounds_per_task):
                    self._run_round(t, r)
                self._end_task(t)
                self._evaluate_task(t)
            self._end_run(time.time() - t_run)
        except KeyboardInterrupt:
            self.logger.error(f"run {self.run_index} interrupted during task {self.current_task}")
            raise
        except BaseException as exc:
            self.logger.error(f"run {self.run_index} failed during task {self.current_task}: "
                              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            raise
        finally:
            close_log_files(self.logger)

    def _begin_task(self, t: int) -> None:
        self.current_task = t
        self._task_t0 = time.time()
        for c in self.clients:
            c._begin_task(t)
        self.on_begin_task(t)
        new, seen = self.scenario.tasks[t], self.scenario.seen_classes(t)
        if self.multi:
            R = self.rounds_per_task
            when = f"rounds {t * R}..{(t + 1) * R - 1} (global)"
        else:
            when = "one exchange (single-round)"
        self.logger.info(f"=== task {t}/{self.num_tasks - 1}: {len(new)} new classes, {len(seen)} seen, {when}")

    def _run_round(self, t: int, r: int) -> None:
        t0 = time.time()
        selected = self.select_clients(t, r)
        if not self.multi:
            active = [c.id for c in self.clients if self.client_is_active(c.id, t)]
            if [c.id for c in selected] != active:
                raise RuntimeError("a single-round algorithm must hear from every active client once per task")
        phase = "round" if self.multi else "exchange"
        for c in selected:
            payload = self.download_payload(t, r, c)
            if payload is not None:
                self._log_comm(t, phase, r, c.id, "down", payload)
                c.receive(t, r, payload)
            c.train_round(t, r)
        uploads, losses, weights = [], [], []
        for c in selected:
            payload = c.upload_payload(t, r)
            if payload is not None:
                self._log_comm(t, phase, r, c.id, "up", payload)
                uploads.append((c.id, payload))
            report = c.round_report() or {}
            if report.get("train_loss") is not None and report.get("n"):
                losses.append(float(report["train_loss"]))
                weights.append(float(report["n"]))
        self.aggregate(t, r, uploads)
        if self.multi:
            curve = self.eval_gap > 0 and (r + 1) % self.eval_gap == 0
            acc = self._evaluate_round(t, r) if curve else None
            loss = float(np.average(losses, weights=weights)) if weights else None
            self.results.log_round(t, r, selected=[c.id for c in selected], uploads=len(uploads), train_loss=loss,
                                   acc_seen=acc, seconds=time.time() - t0)
            if acc is not None:
                self.logger.info(f"    round {r} (global {t * self.rounds_per_task + r}): accuracy on seen classes "
                                 f"{acc:.4f}" + (f", train loss {loss:.4f}" if loss is not None else ""))

    def _end_task(self, t: int) -> None:
        self.on_end_task(t)
        for c in self.clients:
            c._end_task(t)
        payload = self.final_broadcast_payload(t)
        if payload is not None:
            for c in self.clients:
                if self.client_is_active(c.id, t):
                    self._log_comm(t, "task_end", None, c.id, "down", payload)
                    c.receive_final(t, payload)

    def _evaluate_task(self, t: int) -> None:
        T, K = self.num_tasks, self.num_clients
        seen = self.scenario.seen_classes(t)
        gc, gt = np.zeros(T, np.int64), np.zeros(T, np.int64)
        cc, ct = np.zeros((K, T), np.int64), np.zeros((K, T), np.int64)
        with self.store.evaluation():
            for i in range(t + 1):
                rows = self.scenario.test_global_rows[i]
                ok = self._predict_rows(self.predict, rows, seen) == self.store.labels(rows)
                gc[i], gt[i] = int(ok.sum()), len(rows)
                for c in self.clients:
                    crow = c.data.test_rows(i)
                    if len(crow) == 0:
                        continue
                    if c.PERSONALIZED:
                        cok = self._predict_rows(c.predict, crow, seen) == self.store.labels(crow)
                    else:
                        pos = np.searchsorted(rows, crow)
                        if np.any(pos >= len(rows)) or not np.array_equal(rows[pos], crow):
                            raise RuntimeError(f"client {c.id}: test rows of task {i} are not in the global test set")
                        cok = ok[pos]
                    cc[c.id, i], ct[c.id, i] = int(cok.sum()), len(crow)
        n_train = np.array([len(self.scenario.train_rows[t][k]) for k in range(K)], dtype=np.int64)
        row = self.results.log_task(t, classes_seen=len(seen), glob_correct=gc, glob_total=gt, client_correct=cc,
                                    client_total=ct, n_train=n_train, seconds=time.time() - self._task_t0)
        self.logger.info(
            f"task {t}: accuracy on all {len(seen)} seen classes {row['acc_seen']:.4f} | new task "
            f"{row['acc_task_new']:.4f} | clients mean {row['client_acc_mean']:.4f} min {row['client_acc_min']:.4f} "
            f"| up {row['bytes_up'] / 1e6:.3f} MB down {row['bytes_down'] / 1e6:.3f} MB | {row['seconds']:.1f}s")

    def _evaluate_round(self, t: int, r: int) -> float:
        """Curve point: accuracy of the global model on the global test set of all seen classes."""
        seen = self.scenario.seen_classes(t)
        correct = total = 0
        with self.store.evaluation():
            for i in range(t + 1):
                rows = self.scenario.test_global_rows[i]
                correct += int((self._predict_rows(self.predict, rows, seen) == self.store.labels(rows)).sum())
                total += len(rows)
        return correct / total if total else float("nan")

    def _predict_rows(self, fn, rows: np.ndarray, seen: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.int64)
        chunk = self.EVAL_CHUNK if type(self).INPUT == FEATURES else None
        pos = 0
        for X in self.store.prediction_batches(rows, chunk):
            n = int(X.shape[0])
            pred = fn(X, seen)
            if hasattr(pred, "detach"):                  # a torch tensor
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred)
            if pred.shape != (n,) or not np.issubdtype(pred.dtype, np.integer):
                raise TypeError(f"predict() must return {n} integer class ids, got {pred.dtype} {pred.shape}")
            out[pos:pos + n] = pred
            pos += n
        if pos != len(rows):
            raise RuntimeError(f"prediction covered {pos} of {len(rows)} rows")
        bad = ~np.isin(out, seen)
        if bad.any():
            raise RuntimeError(f"predict() returned classes {np.unique(out[bad])[:10].tolist()} that are not seen "
                               f"yet (seen so far: {sorted(seen.tolist())})")
        return out

    def _end_run(self, seconds: float) -> None:
        m = self.results.finalize(seconds_total=seconds)
        self.logger.info(f"run {self.run_index} complete: final accuracy {m['acc_final']:.4f}, A_avg "
                         f"{m['acc_avg_incremental']:.4f}, up {m['bytes_up_total'] / 1e6:.3f} MB, "
                         f"down {m['bytes_down_total'] / 1e6:.3f} MB, {seconds:.1f}s")

    def _log_comm(self, t: int, phase: str, r, client_id: int, direction: str, payload) -> None:
        self.results.log_comm(t, phase, r if phase == "round" else None, client_id, direction,
                              wire_nbytes(payload), payload_keys(payload))

    def send(self, client: Client, t: int, payload) -> None:
        """Task-end message from the server to one client (logged like every other message)."""
        if payload is None:
            return
        self._log_comm(t, "task_end", None, client.id, "down", payload)
        client.receive(t, None, payload)

    # ------------------------------------------------------------------ policies an algorithm may change
    def client_is_active(self, client_id: int, t: int) -> bool:
        return True                    # join schedules (late joiners) come with the late-joiner metric

    def select_clients(self, t: int, r: int) -> list:
        """All active clients; multi-round with join_ratio < 1: a seeded sample of int(K * join_ratio)."""
        active = [c for c in self.clients if self.client_is_active(c.id, t)]
        if not self.multi or self.join_ratio >= 1.0:
            return active
        m = max(1, int(len(active) * self.join_ratio))
        rng = np.random.default_rng([self.seeds["selection"], int(t), int(r)])
        pick = np.sort(rng.choice(len(active), size=m, replace=False))
        return [active[i] for i in pick]

    def rng(self, tag: str, *keys: int) -> np.random.Generator:
        """A server-side generator for (tag, keys) that does not depend on call order."""
        return np.random.default_rng([self.seeds["algorithm"], zlib.crc32(tag.encode()), *(int(k) for k in keys)])

    # ------------------------------------------------------------------ hooks for algorithms
    def init_algorithm(self) -> None:
        """Set up the global model or statistics (self.store.dim, self.num_classes are known)."""

    def on_begin_task(self, t: int) -> None:
        """The classes of task t have arrived (clients already see the new data)."""

    def download_payload(self, t: int, r: int, client: Client):
        """Message for a selected client before its local work in round r (None: nothing)."""
        return None

    def aggregate(self, t: int, r: int, uploads: list) -> None:
        """Combine the uploads [(client_id, payload)] of round r (required)."""
        raise NotImplementedError

    def on_end_task(self, t: int) -> None:
        """After the last round of task t (may call self.send(client, t, payload))."""

    def final_broadcast_payload(self, t: int):
        """The model clients use for inference after task t, sent to every active client (None: nothing)."""
        return None

    def predict(self, X, seen: np.ndarray) -> np.ndarray:
        """Class ids (int) of the global model for a batch X (features [n, d], or images [n, 3, H, W] for
        image algorithms); every id must be in `seen` (required)."""
        raise NotImplementedError

    @classmethod
    def check_hparams(cls, hp: dict, sources: dict) -> None:
        """Extra checks of the resolved hyperparameters, before anything runs (raise ConfigError)."""
