#!/usr/bin/env python
"""Turn an official-run dump into a FICL dataset, so that FICL's main.py can run our methods on the exact
features, client split and task split the official code used.

    python repro/import_dump.py repro/work/dumps/afl_cifar100_seed1.npz [--dataset_root repro/work/ficl_data]

Writes <dataset_root>/Repro_<stem>/ with
    cl_index/                         one row per training sample (official order), then one per test sample
    features/official/dump/           the dump's float32 features stored as FICL's float16 cache [N, 1, 1, d]
    <scenario>/                       the official split: task t, client k -> the rows the official run gave it

The scenario folder name follows FICL's naming (cl_<K>clients_dir_<alpha>_cpt<c>_seed0, or ..._iid_... when the
official split has no Dirichlet alpha); it is only a label: the rows come from the dump, not from FICL's
generator. Test rows are shared out to clients in proportion to their training counts (FICL needs per-client
test shards for its per-client metrics; the comparison uses the global test set only).
An existing folder built from the same dump is reused; one built from another dump is never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, load_dump, sha256_file  # noqa: E402

DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
BACKBONE, GROUP = "official", "dump"


def _load(filename: str, name: str):
    """dataset/utils/<filename> under a private name (dataset/utils must not shadow system/utils)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(DATASET_DIR, "utils", filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dataset_name(dump_path: str) -> str:
    stem = os.path.splitext(os.path.basename(dump_path))[0]
    return "Repro_" + re.sub(r"[^A-Za-z0-9_]", "_", stem)


def scenario_params(meta: dict) -> dict:
    s = meta["setting"]
    alpha = s.get("alpha")
    partition = "dir" if (s.get("partition_label") == "dir" and alpha and float(alpha) > 0) else "iid"
    return {"partition": partition, "num_clients": int(s["num_clients"]),
            "alpha": float(alpha) if partition == "dir" else 0.1,
            "class_per_task": len(s["tasks"][0]), "seed": 0}


def import_dump(dump_path: str, dataset_root: str) -> dict:
    raw = _load("raw_utils.py", "_repro_raw_utils")
    scen = _load("scenario_utils.py", "_repro_scenario_utils")
    cache = _load("cache_utils.py", "_repro_cache_utils")

    meta, a = load_dump(dump_path)
    digest = sha256_file(dump_path)
    name = dataset_name(dump_path)
    ds_dir = os.path.join(dataset_root, name)
    stamp_path = os.path.join(ds_dir, "repro_source.json")
    sp = scenario_params(meta)
    info = {"dataset": name, "dataset_root": os.path.abspath(dataset_root), "backbone": BACKBONE, "group": GROUP,
            **sp, "dump": os.path.abspath(dump_path), "dump_sha256": digest}
    if os.path.exists(stamp_path):
        with open(stamp_path) as f:
            old = json.load(f)
        if old.get("dump_sha256") != digest:
            raise FileExistsError(f"{ds_dir} was built from another dump ({old.get('dump')}); delete it to rebuild")
        return info

    s = meta["setting"]
    C, K = int(s["num_classes"]), int(s["num_clients"])
    tasks = [np.asarray(c, dtype=np.int64) for c in s["tasks"]]
    n, m, d = len(a["train_y"]), len(a["test_y"]), int(a["train_x"].shape[1])

    labels = np.concatenate([a["train_y"], a["test_y"]]).astype(np.int64)
    split = np.concatenate([np.full(n, raw.SPLIT_TRAIN), np.full(m, raw.SPLIT_TEST)]).astype(np.int8)
    index = raw.RawIndex(name=name, labels=labels, split=split, domain=np.full(n + m, -1, dtype=np.int8),
                         ids=[f"train{i}" for i in range(n)] + [f"test{j}" for j in range(m)],
                         class_names=[f"class{c}" for c in range(C)],
                         meta={"imported_from": os.path.basename(dump_path), "dump_sha256": digest,
                               "source": meta.get("source", {})})
    os.makedirs(ds_dir, exist_ok=True)
    raw.save_raw_index(index, os.path.join(ds_dir, raw.INDEX_DIRNAME))

    gdir = cache.group_dir(dataset_root, name, BACKBONE, GROUP)
    shape = (n + m, 1, 1, d)
    arr = cache.create_cache(gdir, shape)
    arr[:n, 0, 0] = a["train_x"].astype(np.float16)
    arr[n:, 0, 0] = a["test_x"].astype(np.float16)
    arr.flush()
    del arr
    cache.finalize_cache(gdir, cache.CacheManifest(
        dataset=name, backbone=BACKBONE, group=GROUP, blocks=[0], tokens=["cls"], final_ln=False, embed_dim=d,
        num_samples=n + m, shape=list(shape), raw_index_hash=index.hash(), preprocess="official run (see dump meta)",
        notes=f"float32 features of an official run, stored as float16; dump {os.path.basename(dump_path)}"))

    rng = np.random.default_rng(0)
    train_rows, test_rows, test_global = [], [], []
    for t, cls in enumerate(tasks):
        tr = [np.flatnonzero((a["train_task"] == t) & (a["train_client"] == k)).astype(np.int64) for k in range(K)]
        glob = (n + np.flatnonzero(np.isin(a["test_y"], cls))).astype(np.int64)
        parts = [[] for _ in range(K)]
        for c in cls:
            counts = np.array([np.count_nonzero(a["train_y"][r] == c) for r in tr])
            rows_c = n + np.flatnonzero(a["test_y"] == c)
            if counts.sum() == 0:
                raise ValueError(f"class {c} has test samples but no client trains on it")
            for k, p in enumerate(scen.allocate_test_shards(counts, rows_c, rng)):
                parts[k].append(p)
        train_rows.append(tr)
        test_rows.append([np.sort(np.concatenate(p)).astype(np.int64) for p in parts])
        test_global.append(glob)
    spec = scen.ScenarioSpec(dataset=name, partition=sp["partition"], num_clients=K, alpha=sp["alpha"],
                             class_per_task=sp["class_per_task"], seed=0)
    scenario = scen.Scenario(spec=spec, config_name=scen.get_config_name(spec), num_clients=K, num_classes=C,
                             raw_index_hash=index.hash(), class_order=np.concatenate(tasks), tasks=tasks,
                             train_rows=train_rows, test_rows=test_rows, test_global_rows=test_global,
                             sub_seeds={}, redraws=[0] * len(tasks),
                             counts={"train": scen._count_table(labels, train_rows),
                                     "test": scen._count_table(labels, test_rows)})
    scen.save_scenario(scenario, ds_dir)
    with open(stamp_path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump")
    p.add_argument("--dataset_root", default=os.path.join(REPO_ROOT, "repro", "work", "ficl_data"))
    a = p.parse_args(argv)
    info = import_dump(a.dump, a.dataset_root)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
