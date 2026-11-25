"""Class-incremental federated scenarios for FICL (the "event schedule").

A scenario fixes which samples each client holds at each task:
  * classes are shuffled once with the seed (class_order "shuffled", the default) or kept in
    label-id order 0..C-1 (class_order "sorted", the order of fed-mammoth and most benchmark
    code), then cut into tasks of class_per_task classes (the last task may be smaller);
  * the training samples of each class go to clients by
      dir      Dirichlet(alpha) proportions per class (plain draw, no size cap),
      iid      an even random split,
      natural  client k = domain k (e.g. DomainNet);
  * each client's test shard follows its own training mix: the official test samples of
    class c are shared out in proportion to how many training samples of c each client
    holds (largest remainder, exact integers). Together the shards are exactly the
    official test set. In natural mode, client k gets domain k's test samples.

Nothing is dropped and nothing is duplicated: within a task, client shards are disjoint
and cover every training sample of the task's classes (tests check this).

Layout written by save_scenario (one <Dataset>/<config_name>/ folder per split, as in HtFLlib):
    dataset/<Name>/<config_name>/config.json
                                /task_<t>/train/<k>.npy   int64 rows of cl_index
                                /task_<t>/test/<k>.npy
                                /task_<t>/test_global.npy

The folder name includes the seed, so every seed gets its own split. Files are loaded
with allow_pickle=False and checked against the sha256 values in config.json.

This module must stay standalone (numpy only, no `utils.*` imports): the system-side
tests load it by file path. Keep get_config_name identical to
system/utils/data_utils.py:get_config_name (a test compares them).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2      # same codes as raw_utils
PARTITIONS = ("dir", "iid", "natural")
CLASS_ORDERS = ("shuffled", "sorted")             # seeded permutation (default) | label ids 0..C-1
SCENARIO_FORMAT = "fcl-scenario/1"
MAX_REDRAWS = 1000


def format_alpha(alpha: float) -> str:
    """Alpha as it appears in folder names: 1.0 -> '1', 0.1 -> '0.1'."""
    return f"{float(alpha):g}"


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything that defines a scenario. Two equal specs on the same raw index give identical files."""

    dataset: str
    partition: str                         # "dir" | "iid" | "natural"
    num_clients: int = 10                  # ignored for natural (one client per domain)
    alpha: float = 0.1                     # Dirichlet concentration, dir only
    class_per_task: int = 2
    seed: int = 0
    use_val: str = "ignore"                # "ignore" the official val split, or "merge" it into train
    min_train_per_client: int = 0          # 0 = no rule; otherwise redraw a task until every client has this many
    class_order: str = "shuffled"          # "shuffled" (seeded) | "sorted" (label ids in order, as fed-mammoth)
    join_task: tuple[int, ...] | None = None   # reserved for late joiners (not implemented yet)

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset name is empty")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, got {self.partition!r}")
        if self.partition == "dir" and not float(self.alpha) > 0:
            raise ValueError("alpha must be > 0 for a Dirichlet partition")
        if int(self.class_per_task) < 1:
            raise ValueError("class_per_task must be >= 1")
        if self.partition != "natural" and int(self.num_clients) < 1:
            raise ValueError("num_clients must be >= 1")
        if self.use_val not in ("ignore", "merge"):
            raise ValueError("use_val must be 'ignore' or 'merge'")
        if int(self.min_train_per_client) < 0:
            raise ValueError("min_train_per_client must be >= 0")
        if self.partition == "natural" and int(self.min_train_per_client) > 0:
            raise ValueError("natural partitions are fixed by the domains and cannot be redrawn")
        if self.class_order not in CLASS_ORDERS:
            raise ValueError(f"class_order must be one of {CLASS_ORDERS}, got {self.class_order!r}")
        if self.join_task is not None:
            object.__setattr__(self, "join_task", tuple(int(j) for j in self.join_task))


def get_config_name(spec: ScenarioSpec) -> str:
    """Folder name, e.g. cl_10clients_dir_0.1_cpt2_seed0 (natural: cl_natural_cpt35_seed0)."""
    if spec.partition == "natural":
        name = f"cl_natural_cpt{spec.class_per_task}_seed{spec.seed}"
    elif spec.partition == "iid":
        name = f"cl_{spec.num_clients}clients_iid_cpt{spec.class_per_task}_seed{spec.seed}"
    else:
        name = f"cl_{spec.num_clients}clients_dir_{format_alpha(spec.alpha)}_cpt{spec.class_per_task}_seed{spec.seed}"
    if spec.use_val == "merge":
        name += "_valmerge"
    if spec.min_train_per_client > 0:
        name += f"_min{spec.min_train_per_client}"
    if spec.class_order == "sorted":
        name += "_sorted"
    if spec.join_task is not None:
        name += "_join" + hashlib.sha256(",".join(map(str, spec.join_task)).encode()).hexdigest()[:8]
    return name


def derive_seeds(seed: int) -> dict[str, int]:
    """Independent random streams for the class order, the partition and the test allocation."""
    children = np.random.SeedSequence(int(seed)).spawn(3)
    return {name: int(child.generate_state(1)[0])
            for name, child in zip(("class_order", "partition", "test_alloc"), children)}


def make_class_order(num_classes: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(int(num_classes)).astype(np.int64)


def split_tasks(class_order: np.ndarray, class_per_task: int) -> list[np.ndarray]:
    """Consecutive groups of class_per_task classes; the last group may be smaller."""
    if int(class_per_task) < 1:
        raise ValueError("class_per_task must be >= 1")
    order = np.asarray(class_order, dtype=np.int64)
    return [order[i:i + class_per_task].copy() for i in range(0, len(order), class_per_task)]


def dirichlet_partition(rows: np.ndarray, num_clients: int, alpha: float,
                        rng: np.random.Generator, max_redraws: int = 100) -> list[np.ndarray]:
    """Split the rows of ONE class over clients with p ~ Dirichlet(alpha). Exact cover, sorted parts."""
    rows = np.asarray(rows, dtype=np.int64)
    shuffled = rng.permutation(rows)
    for _ in range(max_redraws):
        p = rng.dirichlet(np.full(int(num_clients), float(alpha)))
        if np.all(np.isfinite(p)) and p.sum() > 0:
            break
    else:
        raise RuntimeError(f"Dirichlet({alpha}) gave no usable proportions in {max_redraws} draws")
    p = p / p.sum()
    cuts = np.floor(np.cumsum(p)[:-1] * len(shuffled)).astype(np.int64)
    cuts = np.clip(cuts, 0, len(shuffled))
    return [np.sort(part).astype(np.int64) for part in np.split(shuffled, cuts)]


def iid_partition(rows: np.ndarray, num_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Even random split of the rows of ONE class."""
    shuffled = rng.permutation(np.asarray(rows, dtype=np.int64))
    return [np.sort(part).astype(np.int64) for part in np.array_split(shuffled, int(num_clients))]


def natural_partition(rows: np.ndarray, row_domains: np.ndarray, num_clients: int) -> list[np.ndarray]:
    """Client k gets the rows whose domain is k."""
    rows = np.asarray(rows, dtype=np.int64)
    row_domains = np.asarray(row_domains)
    return [np.sort(rows[row_domains == k]).astype(np.int64) for k in range(int(num_clients))]


def allocate_test_shards(train_counts: np.ndarray, test_rows: np.ndarray,
                         rng: np.random.Generator) -> list[np.ndarray]:
    """Share the test rows of ONE class in proportion to each client's training count.

    Largest remainder with exact integer arithmetic: client k gets floor(m * n_k / n) or one
    more, so |m_k - m n_k / n| < 1, and a client with n_k = 0 gets nothing. Ties are broken
    by a seeded random order.
    """
    counts = np.asarray(train_counts, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    k, m, n = len(counts), len(test_rows), int(counts.sum())
    if m == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(k)]
    if n == 0:
        raise ValueError("test samples of a class that no client trains on cannot be allocated")
    num = m * counts
    quota = num // n
    remainder_num = num % n
    extra = m - int(quota.sum())
    tie = rng.permutation(k)
    order = np.lexsort((tie, -remainder_num))
    quota[order[:extra]] += 1
    shuffled = rng.permutation(test_rows)
    parts = np.split(shuffled, np.cumsum(quota)[:-1])
    return [np.sort(p).astype(np.int64) for p in parts]


@dataclass
class Scenario:
    spec: ScenarioSpec
    config_name: str
    num_clients: int
    num_classes: int
    raw_index_hash: str
    class_order: np.ndarray                      # int64 [C]
    tasks: list[np.ndarray]                      # class ids per task
    train_rows: list[list[np.ndarray]]           # [t][k] sorted int64 rows
    test_rows: list[list[np.ndarray]]            # [t][k]
    test_global_rows: list[np.ndarray]           # [t]
    sub_seeds: dict = field(default_factory=dict)
    redraws: list[int] = field(default_factory=list)
    counts: dict = field(default_factory=dict)   # {"train"|"test": [t][k] {class: n}}

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def seen_classes(self, t: int) -> np.ndarray:
        return np.concatenate(self.tasks[:t + 1]).astype(np.int64)


def _count_table(labels: np.ndarray, rows: list[list[np.ndarray]]) -> list[list[dict]]:
    out = []
    for per_task in rows:
        task_counts = []
        for r in per_task:
            vals, cnt = np.unique(labels[r], return_counts=True)
            task_counts.append({str(int(v)): int(c) for v, c in zip(vals, cnt)})
        out.append(task_counts)
    return out


def build_scenario(index, spec: ScenarioSpec) -> Scenario:
    """Build the scenario for a RawIndex (any object with labels, split, domain, class_names,
    domain_names and hash())."""
    if spec.join_task is not None:
        raise NotImplementedError("join schedules come with the late-joiner metric (checkpoint 7)")
    labels = np.asarray(index.labels, dtype=np.int64)
    split = np.asarray(index.split)
    domain = np.asarray(index.domain)
    num_classes = len(index.class_names)
    if spec.partition == "natural":
        num_clients = len(index.domain_names)
        if num_clients < 1:
            raise ValueError(f"{index.name}: natural partition needs a dataset with domains")
    else:
        num_clients = int(spec.num_clients)

    train_mask = split == SPLIT_TRAIN
    if spec.use_val == "merge":
        train_mask = train_mask | (split == SPLIT_VAL)
    test_mask = split == SPLIT_TEST

    seeds = derive_seeds(spec.seed)
    rng_order = np.random.default_rng(seeds["class_order"])
    rng_part = np.random.default_rng(seeds["partition"])
    rng_test = np.random.default_rng(seeds["test_alloc"])

    if spec.class_order == "sorted":
        class_order = np.arange(num_classes, dtype=np.int64)     # the seed still sets the partition
    else:
        class_order = make_class_order(num_classes, rng_order)
    tasks = split_tasks(class_order, spec.class_per_task)
    train_rows, test_rows, test_global, redraws = [], [], [], []

    for t, classes in enumerate(tasks):
        for attempt in range(MAX_REDRAWS):
            per_client: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
            for c in classes:
                rows_c = np.flatnonzero(train_mask & (labels == c)).astype(np.int64)
                if spec.partition == "dir":
                    parts = dirichlet_partition(rows_c, num_clients, spec.alpha, rng_part)
                elif spec.partition == "iid":
                    parts = iid_partition(rows_c, num_clients, rng_part)
                else:
                    parts = natural_partition(rows_c, domain[rows_c], num_clients)
                for k in range(num_clients):
                    per_client[k].append(parts[k])
            tr = [np.sort(np.concatenate(p)).astype(np.int64) for p in per_client]
            if spec.min_train_per_client == 0 or min(len(r) for r in tr) >= spec.min_train_per_client:
                break
        else:
            raise ValueError(f"task {t}: could not give every client >= {spec.min_train_per_client} training "
                             f"samples in {MAX_REDRAWS} draws; lower --min_train_per_client or raise --alpha")
        redraws.append(attempt)

        te_parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            rows_test_c = np.flatnonzero(test_mask & (labels == c)).astype(np.int64)
            if spec.partition == "natural":
                parts = natural_partition(rows_test_c, domain[rows_test_c], num_clients)
            else:
                counts_c = np.array([np.count_nonzero(labels[tr[k]] == c) for k in range(num_clients)])
                parts = allocate_test_shards(counts_c, rows_test_c, rng_test)
            for k in range(num_clients):
                te_parts[k].append(parts[k])
        te = [np.sort(np.concatenate(p)).astype(np.int64) for p in te_parts]
        glob = np.flatnonzero(test_mask & np.isin(labels, classes)).astype(np.int64)
        if not np.array_equal(np.sort(np.concatenate(te)), glob):
            raise RuntimeError(f"task {t}: client test shards do not add up to the official test set")
        all_train = np.sort(np.concatenate(tr))
        if len(np.unique(all_train)) != len(all_train) or not np.array_equal(
                all_train, np.flatnonzero(train_mask & np.isin(labels, classes))):
            raise RuntimeError(f"task {t}: client training shards overlap or miss samples")
        train_rows.append(tr)
        test_rows.append(te)
        test_global.append(glob)

    return Scenario(spec=spec, config_name=get_config_name(spec), num_clients=num_clients,
                    num_classes=num_classes, raw_index_hash=index.hash(), class_order=class_order, tasks=tasks,
                    train_rows=train_rows, test_rows=test_rows, test_global_rows=test_global, sub_seeds=seeds,
                    redraws=redraws,
                    counts={"train": _count_table(labels, train_rows), "test": _count_table(labels, test_rows)})


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spec_dict(spec: ScenarioSpec) -> dict:
    d = asdict(spec)
    if d["class_order"] == "shuffled":
        del d["class_order"]        # the default: scenario files written before the option stay identical
    if d["join_task"] is not None:
        d["join_task"] = list(d["join_task"])
    return d


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return None


def save_scenario(scn: Scenario, dataset_dir: str) -> str:
    """Write <dataset_dir>/<config_name>/. Returns the folder path.

    An existing folder with the same spec and raw index is kept as it is; a folder with the
    same name but a different spec or raw index is never overwritten.
    """
    name = scn.config_name
    final = os.path.join(dataset_dir, name)
    cfg_path = os.path.join(final, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            existing = json.load(f)
        if existing.get("spec") == _spec_dict(scn.spec) and existing.get("raw_index_hash") == scn.raw_index_hash:
            print(f"Scenario already exists (identical spec and raw index): {final}")
            return final
        raise FileExistsError(f"{final} already exists with a different spec or raw index. "
                              f"Delete it yourself if you really want to regenerate it.")
    if os.path.exists(final):
        raise FileExistsError(f"{final} exists but has no config.json (an interrupted run?); remove it first")

    os.makedirs(dataset_dir, exist_ok=True)
    tmp = os.path.join(dataset_dir, f".{name}.partial-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    files: dict[str, str] = {}

    def put(rel: str, arr: np.ndarray) -> None:
        path = os.path.join(tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.ascontiguousarray(arr, dtype=np.int64))
        files[rel] = _sha256(path)

    for t in range(scn.num_tasks):
        for k in range(scn.num_clients):
            put(f"task_{t}/train/{k}.npy", scn.train_rows[t][k])
            put(f"task_{t}/test/{k}.npy", scn.test_rows[t][k])
        put(f"task_{t}/test_global.npy", scn.test_global_rows[t])

    config = {
        "format": SCENARIO_FORMAT,
        "config_name": name,
        "spec": _spec_dict(scn.spec),
        "num_clients": scn.num_clients,
        "num_classes": scn.num_classes,
        "num_tasks": scn.num_tasks,
        "class_order": scn.class_order.tolist(),
        "tasks": [c.tolist() for c in scn.tasks],
        "sub_seeds": scn.sub_seeds,
        "redraws": scn.redraws,
        "raw_index_hash": scn.raw_index_hash,
        "counts": scn.counts,
        "totals": {"train": int(sum(len(r) for per in scn.train_rows for r in per)),
                   "test": int(sum(len(r) for r in scn.test_global_rows))},
        "files": files,
        "versions": {"numpy": np.__version__, "python": platform.python_version()},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "dataset/generate_scenario.py",
        "git_commit": _git_commit(),
    }
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp, final)
    return final


def load_scenario(config_dir: str, expected_raw_index_hash: str | None = None, verify: bool = True) -> Scenario:
    """Read a scenario folder. With verify, every file must match its sha256 in config.json."""
    with open(os.path.join(config_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("format") != SCENARIO_FORMAT:
        raise ValueError(f"{config_dir}: unknown scenario format {cfg.get('format')!r}")
    if expected_raw_index_hash is not None and cfg["raw_index_hash"] != expected_raw_index_hash:
        raise ValueError(f"{config_dir} was built for another raw index "
                         f"({cfg['raw_index_hash'][:12]} != {expected_raw_index_hash[:12]})")

    def get(rel: str) -> np.ndarray:
        path = os.path.join(config_dir, *rel.split("/"))
        if verify and _sha256(path) != cfg["files"][rel]:
            raise ValueError(f"{path}: checksum does not match config.json (edited or corrupted)")
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.int64 or arr.ndim != 1:
            raise ValueError(f"{path}: expected a 1-D int64 array")
        return arr

    spec_d = dict(cfg["spec"])
    if spec_d.get("join_task") is not None:
        spec_d["join_task"] = tuple(spec_d["join_task"])
    n_tasks, n_clients = cfg["num_tasks"], cfg["num_clients"]
    return Scenario(
        spec=ScenarioSpec(**spec_d), config_name=cfg["config_name"], num_clients=n_clients,
        num_classes=cfg["num_classes"], raw_index_hash=cfg["raw_index_hash"],
        class_order=np.asarray(cfg["class_order"], dtype=np.int64),
        tasks=[np.asarray(c, dtype=np.int64) for c in cfg["tasks"]],
        train_rows=[[get(f"task_{t}/train/{k}.npy") for k in range(n_clients)] for t in range(n_tasks)],
        test_rows=[[get(f"task_{t}/test/{k}.npy") for k in range(n_clients)] for t in range(n_tasks)],
        test_global_rows=[get(f"task_{t}/test_global.npy") for t in range(n_tasks)],
        sub_seeds=cfg.get("sub_seeds", {}), redraws=cfg.get("redraws", []), counts=cfg.get("counts", {}))


def list_configs(dataset_dir: str) -> list[str]:
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(d for d in os.listdir(dataset_dir)
                  if d.startswith("cl_") and os.path.exists(os.path.join(dataset_dir, d, "config.json")))
