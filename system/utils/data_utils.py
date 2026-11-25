"""Reading FICL data in system/ (scenarios, raw index, fp16 feature caches).

Mirrors utils/data_utils.py: files are read by path from ../dataset/<Name>/..., and this
module never imports anything from dataset/ (with both folders on the path, dataset/utils
would shadow this utils package). The file formats are the ones written by
dataset/utils/raw_utils.py, scenario_utils.py and cache_utils.py; the tests in
system/tests/ build fixtures with those writers and read them back with this module.

    cfg = DataConfig(dataset="TissueMNIST", partition="dir", num_clients=10, alpha=0.1,
                       class_per_task=2, seed=0)
    set_data_config(cfg)                      # like set_data_config: finds the folder or lists what exists
    scenario = load_scenario(cfg)
    store = FeatureStore("TissueMNIST", "dinov2_vitb14", FeatureView(group="pre"))
    check_compatible(scenario, store)
    data = ClientData(scenario, client_id=0)
    data.begin_task(0)                           # the "new data arrived" event
    Z = store.features(data.train_rows())        # float64 [n, d]
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

SCENARIO_FORMAT = "fcl-scenario/1"
CACHE_FORMAT = "fcl-feature-cache/1"
INDEX_FORMAT = "fcl-raw-index/1"
DEFAULT_DATASET_ROOT = os.path.join("..", "dataset")
SPLIT_TEST = 2          # same code as dataset/utils/raw_utils.py


class TestDataAccessError(RuntimeError):
    """Test-split features or labels were requested outside FeatureStore.evaluation() (invariant I5)."""

    __test__ = False    # not a pytest test class


# ---------------------------------------------------------------------------
# Finding a scenario (like DataConfig / set_data_config in data_utils.py)
# ---------------------------------------------------------------------------
def format_alpha(alpha: float) -> str:
    return f"{float(alpha):g}"


def get_config_name(partition: str, num_clients: int, alpha: float, class_per_task: int, seed: int,
                       use_val: str = "ignore", min_train_per_client: int = 0,
                       class_order: str = "shuffled") -> str:
    """Must stay identical to dataset/utils/scenario_utils.py:get_config_name (a test compares them)."""
    if partition == "natural":
        if min_train_per_client > 0:
            raise ValueError("natural partitions have no minimum-size rule")
        name = f"cl_natural_cpt{class_per_task}_seed{seed}"
    elif partition == "iid":
        name = f"cl_{num_clients}clients_iid_cpt{class_per_task}_seed{seed}"
    elif partition == "dir":
        name = f"cl_{num_clients}clients_dir_{format_alpha(alpha)}_cpt{class_per_task}_seed{seed}"
    else:
        raise ValueError(f"unknown partition {partition!r}")
    if use_val == "merge":
        name += "_valmerge"
    if min_train_per_client > 0:
        name += f"_min{min_train_per_client}"
    if class_order == "sorted":
        name += "_sorted"
    elif class_order != "shuffled":
        raise ValueError(f"unknown class_order {class_order!r}")
    return name


@dataclass
class DataConfig:
    dataset: str
    partition: str = "dir"
    num_clients: int = 10
    alpha: float = 0.1
    class_per_task: int = 2
    seed: int = 0
    use_val: str = "ignore"
    min_train_per_client: int = 0
    class_order: str = "shuffled"
    dataset_root: str = DEFAULT_DATASET_ROOT

    # filled in by set_data_config
    config_name: str = field(default="", init=False)
    config_dir: str = field(default="", init=False)
    num_classes: int = field(default=0, init=False)
    num_tasks: int = field(default=0, init=False)


def _available_configs(dataset_root: str, dataset: str) -> list[str]:
    base = os.path.join(dataset_root, dataset)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if d.startswith("cl_") and os.path.exists(os.path.join(base, d, "config.json")))


def set_data_config(cfg: DataConfig) -> None:
    """Find ../dataset/<Name>/<config_name>/ and fill in config_name, num_classes, num_tasks
    (and num_clients for natural scenarios). Raises with the list of existing scenarios."""
    name = get_config_name(cfg.partition, cfg.num_clients, cfg.alpha, cfg.class_per_task, cfg.seed,
                              cfg.use_val, cfg.min_train_per_client, cfg.class_order)
    path = os.path.join(cfg.dataset_root, cfg.dataset, name)
    if not os.path.exists(os.path.join(path, "config.json")):
        if cfg.dataset == "Synthetic":
            hint = "cd dataset && python make_synthetic.py   (writes the Synthetic scenarios of the smoke configs)"
        else:
            hint = f"cd dataset && python generate_scenario.py {cfg.dataset} {cfg.partition}"
            if cfg.partition != "natural":
                hint += f" --num_clients {cfg.num_clients}"
            if cfg.partition == "dir":
                hint += f" --alpha {cfg.alpha:g}"
            hint += f" --class_per_task {cfg.class_per_task} --seed {cfg.seed}"
            if cfg.use_val != "ignore":
                hint += f" --use_val {cfg.use_val}"
            if cfg.min_train_per_client:
                hint += f" --min_train_per_client {cfg.min_train_per_client}"
            if cfg.class_order != "shuffled":
                hint += f" --class_order {cfg.class_order}"
            if not os.path.exists(os.path.join(cfg.dataset_root, cfg.dataset, "cl_index", "index.json")):
                hint = (f"cd dataset && python prepare_data.py {cfg.dataset} --download  (there is no index yet), "
                        f"then  " + hint.replace("cd dataset && ", ""))
        raise FileNotFoundError(
            f"Scenario not found: {path}\n"
            f"Available for {cfg.dataset}: {_available_configs(cfg.dataset_root, cfg.dataset)}\n"
            f"Generate it first: {hint}")
    with open(os.path.join(path, "config.json")) as f:
        meta = json.load(f)
    if meta.get("format") != SCENARIO_FORMAT:
        raise ValueError(f"{path}: unknown scenario format {meta.get('format')!r}")
    cfg.config_name, cfg.config_dir = name, path
    cfg.num_classes, cfg.num_tasks = int(meta["num_classes"]), int(meta["num_tasks"])
    if cfg.partition == "natural":
        cfg.num_clients = int(meta["num_clients"])
    elif int(meta["num_clients"]) != cfg.num_clients:
        raise ValueError(f"{path}: config.json says {meta['num_clients']} clients, expected {cfg.num_clients}")
    print(f"[DataConfig] {cfg.dataset}: scenario={name}, {cfg.num_clients} clients, "
          f"{cfg.num_tasks} tasks, {cfg.num_classes} classes")


# ---------------------------------------------------------------------------
# Scenario reader
# ---------------------------------------------------------------------------
def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Scenario:
    config_name: str
    config_dir: str
    spec: dict
    num_clients: int
    num_classes: int
    raw_index_hash: str
    class_order: np.ndarray
    tasks: list[np.ndarray]
    train_rows: list[list[np.ndarray]]      # [t][k] int64 rows of cl_index
    test_rows: list[list[np.ndarray]]       # [t][k]
    test_global_rows: list[np.ndarray]      # [t]
    content_sha256: str = ""                # what the scenario contains, without its provenance fields

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def seen_classes(self, t: int) -> np.ndarray:
        return np.concatenate(self.tasks[:t + 1]).astype(np.int64)


CONTENT_KEYS = ("spec", "raw_index_hash", "class_order", "tasks", "files")


def scenario_content_sha256(meta: dict) -> str:
    """sha256 of a scenario's content: spec, raw index hash, class order, tasks and the sha256 of every
    split file. Unlike the sha256 of config.json it leaves out where and when the scenario was written
    (creation time, library versions, git commit), so equal splits give equal values on every machine."""
    content = {k: meta[k] for k in CONTENT_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_scenario(cfg_or_dir, verify: bool = True) -> Scenario:
    """Load a scenario from a DataConfig (calls set_data_config if needed) or a folder path.
    With verify, every array file must match its sha256 in config.json."""
    if isinstance(cfg_or_dir, DataConfig):
        if not cfg_or_dir.config_dir:
            set_data_config(cfg_or_dir)
        config_dir = cfg_or_dir.config_dir
    else:
        config_dir = str(cfg_or_dir)
    with open(os.path.join(config_dir, "config.json")) as f:
        meta = json.load(f)
    if meta.get("format") != SCENARIO_FORMAT:
        raise ValueError(f"{config_dir}: unknown scenario format {meta.get('format')!r}")

    def get(rel: str) -> np.ndarray:
        path = os.path.join(config_dir, *rel.split("/"))
        if verify and _sha256(path) != meta["files"][rel]:
            raise ValueError(f"{path}: checksum does not match config.json (edited or corrupted)")
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.int64 or arr.ndim != 1:
            raise ValueError(f"{path}: expected a 1-D int64 array")
        return arr

    n_tasks, n_clients = int(meta["num_tasks"]), int(meta["num_clients"])
    return Scenario(
        config_name=meta["config_name"], config_dir=config_dir, spec=dict(meta["spec"]), num_clients=n_clients,
        num_classes=int(meta["num_classes"]), raw_index_hash=meta["raw_index_hash"],
        class_order=np.asarray(meta["class_order"], dtype=np.int64),
        tasks=[np.asarray(c, dtype=np.int64) for c in meta["tasks"]],
        train_rows=[[get(f"task_{t}/train/{k}.npy") for k in range(n_clients)] for t in range(n_tasks)],
        test_rows=[[get(f"task_{t}/test/{k}.npy") for k in range(n_clients)] for t in range(n_tasks)],
        test_global_rows=[get(f"task_{t}/test_global.npy") for t in range(n_tasks)],
        content_sha256=scenario_content_sha256(meta))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureView:
    """Which part of a cache group a run uses.

    blocks / tokens: None = everything stored in the group, otherwise block ids (as stored in
    the manifest) and token names, in the order given. proj_rank: optional fixed random
    projection of each block's concatenated tokens to proj_rank dims (needs proj_seed).
    """

    group: str
    blocks: tuple[int, ...] | None = None
    tokens: tuple[str, ...] | None = None
    proj_rank: int | None = None
    proj_seed: int | None = None

    def __post_init__(self) -> None:
        if self.proj_rank is not None:
            if int(self.proj_rank) < 1:
                raise ValueError("proj_rank must be >= 1")
            if self.proj_seed is None:
                raise ValueError("a projection needs an explicit proj_seed (it is logged with the run)")


class GuardedIndex:
    """Labels and splits of cl_index, with the guard of invariant I5 (shared by FeatureStore and ImageStore).

    Rows of the official test split can only be read inside `with store.evaluation():`; anywhere
    else labels() (and a store's features or images) raise TestDataAccessError. The shared loop wraps
    evaluation in that context, so training and selection code cannot see test data by accident.
    """

    def _open_index(self, dataset_root: str, dataset: str) -> dict:
        index_dir = os.path.join(dataset_root, dataset, "cl_index")
        path = os.path.join(index_dir, "index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run  python prepare_data.py {dataset}  in dataset/ first")
        with open(path) as f:
            index_meta = json.load(f)
        if index_meta.get("format") != INDEX_FORMAT:
            raise ValueError(f"{index_dir}: unknown index format {index_meta.get('format')!r}")
        self.raw_index_hash = index_meta["hash"]
        self._labels = np.load(os.path.join(index_dir, "labels.npy"), allow_pickle=False)
        self._split = np.load(os.path.join(index_dir, "split.npy"), allow_pickle=False)
        if self._split.shape != self._labels.shape:
            raise ValueError(f"{index_dir}: split.npy and labels.npy have different lengths")
        self._eval_depth = 0
        return index_meta

    @contextlib.contextmanager
    def evaluation(self):
        """Allow reads of test-split rows inside this block (nested blocks are fine)."""
        self._eval_depth += 1
        try:
            yield self
        finally:
            self._eval_depth -= 1

    @property
    def in_evaluation(self) -> bool:
        return self._eval_depth > 0

    def _guard(self, rows: np.ndarray) -> None:
        if self._eval_depth == 0 and len(rows):
            n_test = int(np.count_nonzero(self._split[rows] == SPLIT_TEST))
            if n_test:
                raise TestDataAccessError(
                    f"{n_test} test-split rows were requested outside evaluation; training and selection "
                    f"code may read training rows only (invariant I5)")

    def labels(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        self._guard(rows)
        return self._labels[rows].astype(np.int64)


class FeatureStore(GuardedIndex):
    """Read-only access to one cache group, selected through a FeatureView. Returns float64.

    Rows of the official test split can only be read inside `with store.evaluation():`; anywhere
    else features() and labels() raise TestDataAccessError. The shared loop wraps evaluation in
    that context, so training and selection code cannot see test data by accident (invariant I5).
    """

    input_kind = "features"

    def __init__(self, dataset: str, backbone: str, view: FeatureView, dataset_root: str = DEFAULT_DATASET_ROOT):
        self.dataset, self.backbone, self.view = dataset, backbone, view
        self._open_index(dataset_root, dataset)

        gdir = os.path.join(dataset_root, dataset, "features", backbone, view.group)
        self.group_dir = gdir
        feat_path, man_path = os.path.join(gdir, "features.npy"), os.path.join(gdir, "manifest.json")
        if not (os.path.exists(feat_path) and os.path.exists(man_path)):
            raise FileNotFoundError(f"no finished cache in {gdir}")
        with open(man_path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"{gdir}: unknown cache format {self.manifest.get('format')!r}")
        if self.manifest["raw_index_hash"] != self.raw_index_hash:
            raise ValueError(f"{gdir} was built for another cl_index "
                             f"({self.manifest['raw_index_hash'][:12]} != {self.raw_index_hash[:12]})")
        self._feats = np.load(feat_path, mmap_mode="r", allow_pickle=False)
        if self._feats.dtype != np.float16 or list(self._feats.shape) != list(self.manifest["shape"]):
            raise ValueError(f"{feat_path}: dtype/shape do not match manifest.json")
        if self._feats.shape[0] != len(self._labels):
            raise ValueError(f"{feat_path}: {self._feats.shape[0]} rows but cl_index has {len(self._labels)}")

        stored_blocks, stored_tokens = list(self.manifest["blocks"]), list(self.manifest["tokens"])
        blocks = list(view.blocks) if view.blocks is not None else stored_blocks
        tokens = list(view.tokens) if view.tokens is not None else stored_tokens
        missing_b = [b for b in blocks if b not in stored_blocks]
        missing_t = [t for t in tokens if t not in stored_tokens]
        if missing_b or missing_t or not blocks or not tokens:
            raise ValueError(f"view asks for blocks {missing_b} / tokens {missing_t} that are not in group "
                             f"{view.group} (stored blocks {stored_blocks}, tokens {stored_tokens})")
        self._bi = [stored_blocks.index(b) for b in blocks]
        self._ti = [stored_tokens.index(t) for t in tokens]
        self.blocks, self.tokens = blocks, tokens
        self.embed_dim = int(self._feats.shape[3])
        per_block = len(tokens) * self.embed_dim
        self._proj = None
        if view.proj_rank is not None:
            r = int(view.proj_rank)
            self._proj = [np.random.default_rng([int(view.proj_seed), int(b)]).standard_normal((per_block, r))
                          / np.sqrt(r) for b in blocks]
            self.depth_groups = [r] * len(blocks)
        else:
            self.depth_groups = [per_block] * len(blocks)
        self.dim = int(sum(self.depth_groups))

    @property
    def num_samples(self) -> int:
        return int(self._feats.shape[0])

    def describe(self) -> str:
        return f"cache {self.backbone}/{self.view.group}"

    def verify_checksum(self) -> bool:
        """Recompute the sha256 of features.npy and compare it with manifest.json (reads the whole file)."""
        path = os.path.join(self.group_dir, "features.npy")
        if _sha256(path, chunk=1 << 24) != self.manifest["sha256"]:
            raise ValueError(f"{path}: sha256 does not match manifest.json (edited or corrupted)")
        return True

    def features(self, rows: np.ndarray) -> np.ndarray:
        """float64 [len(rows), dim]: selected blocks x tokens, flattened block by block (then projected)."""
        rows = np.asarray(rows, dtype=np.int64)
        self._guard(rows)
        if len(rows) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        # read only the selected blocks and tokens of each row (not whole rows): float16 -> float64 [n, L, T, D]
        x = np.asarray(self._feats[np.ix_(rows, self._bi, self._ti)]).astype(np.float64)
        n = x.shape[0]
        if self._proj is None:
            return x.reshape(n, -1)
        x = x.reshape(n, len(self.blocks), -1)
        return np.concatenate([x[:, i] @ p for i, p in enumerate(self._proj)], axis=1)

    def iter_chunks(self, rows: np.ndarray, chunk: int = 8192) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (features float64 [b, dim], labels int64 [b]) over rows in chunks."""
        rows = np.asarray(rows, dtype=np.int64)
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            yield self.features(r), self.labels(r)

    def prediction_batches(self, rows: np.ndarray, chunk: int = 8192) -> Iterator[np.ndarray]:
        """The inputs of predict(): features float64 [b, dim] of rows, in row order, in chunks."""
        rows = np.asarray(rows, dtype=np.int64)
        self._guard(rows)
        for i in range(0, len(rows), chunk):
            yield self.features(rows[i:i + chunk])


def check_compatible(scenario: Scenario, store) -> None:
    """The scenario and the store (FeatureStore or ImageStore) must refer to the same cl_index."""
    if scenario.raw_index_hash != store.raw_index_hash:
        raise ValueError(f"scenario {scenario.config_name} and {store.describe()} were built for different "
                         f"cl_index files")


# ---------------------------------------------------------------------------
# What one client may see
# ---------------------------------------------------------------------------
class ClientData:
    """Task-gated rows of one client. begin_task(t) is the "new data arrived" event.

    A client sees only the current task's training rows. past_train_rows needs
    past_train_access=True, which only upper-bound arms may set (it is written to their results).
    Test shards of every task up to the current one stay available for evaluation.
    """

    def __init__(self, scenario: Scenario, client_id: int, past_train_access: bool = False):
        if not 0 <= client_id < scenario.num_clients:
            raise ValueError(f"client {client_id} out of range 0..{scenario.num_clients - 1}")
        self.scenario, self.client_id, self.past_train_access = scenario, int(client_id), bool(past_train_access)
        self.current_task = -1

    def begin_task(self, t: int) -> None:
        if t != self.current_task + 1:
            raise ValueError(f"client {self.client_id}: tasks must arrive in order (got {t} after "
                             f"{self.current_task})")
        if t >= self.scenario.num_tasks:
            raise ValueError(f"task {t} out of range (scenario has {self.scenario.num_tasks})")
        self.current_task = t

    def _require_task(self) -> int:
        if self.current_task < 0:
            raise RuntimeError(f"client {self.client_id}: no task has started (call begin_task first)")
        return self.current_task

    def train_rows(self) -> np.ndarray:
        t = self._require_task()
        return self.scenario.train_rows[t][self.client_id]

    def past_train_rows(self, t_prev: int) -> np.ndarray:
        t = self._require_task()
        if not self.past_train_access:
            raise PermissionError(f"client {self.client_id}: no access to past training data "
                                  f"(only upper-bound arms set past_train_access)")
        if not 0 <= t_prev < t:
            raise ValueError(f"task {t_prev} is not a past task (current task {t})")
        return self.scenario.train_rows[t_prev][self.client_id]

    def test_rows(self, i: int) -> np.ndarray:
        """Row indices of this client's test shard of task i (features need FeatureStore.evaluation())."""
        t = self._require_task()
        if not 0 <= i <= t:
            raise ValueError(f"task {i} has not arrived yet (current task {t})")
        return self.scenario.test_rows[i][self.client_id]

    def seen_classes(self) -> np.ndarray:
        """Class ids of every task up to the current one."""
        return self.scenario.seen_classes(self._require_task())

    def task_classes(self) -> np.ndarray:
        """Class ids of the current task."""
        return self.scenario.tasks[self._require_task()]
