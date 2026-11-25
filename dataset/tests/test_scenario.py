"""Tests for dataset/utils/scenario_utils.py (class-incremental client splits)."""
import json
import os

import numpy as np
import pytest

from utils.raw_utils import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, SyntheticAdapter
from utils.scenario_utils import (
    ScenarioSpec,
    allocate_test_shards,
    build_scenario,
    dirichlet_partition,
    get_config_name,
    list_configs,
    load_scenario,
    save_scenario,
    split_tasks,
)


def _spec(**kw):
    base = dict(dataset="Synthetic", partition="dir", num_clients=5, alpha=0.3, class_per_task=2, seed=0)
    base.update(kw)
    return ScenarioSpec(**base)


def _train_mask(idx, use_val="ignore"):
    mask = idx.split == SPLIT_TRAIN
    if use_val == "merge":
        mask = mask | (idx.split == SPLIT_VAL)
    return mask


def test_each_class_in_one_task(synthetic_index):
    scn = build_scenario(synthetic_index, _spec())
    all_cls = np.concatenate(scn.tasks)
    assert sorted(all_cls.tolist()) == list(range(synthetic_index.num_classes))
    assert len(set(all_cls.tolist())) == len(all_cls)


def test_uneven_last_task(synthetic_index):
    scn = build_scenario(synthetic_index, _spec())          # 7 classes, 2 per task
    assert [len(c) for c in scn.tasks] == [2, 2, 2, 1]
    assert [len(t) for t in split_tasks(np.arange(5), 5)] == [5]


def test_train_shards_disjoint_and_cover(synthetic_index):
    idx = synthetic_index
    for partition in ("dir", "iid", "natural"):
        scn = build_scenario(idx, _spec(partition=partition))
        for t, classes in enumerate(scn.tasks):
            union = np.concatenate(scn.train_rows[t])
            assert len(union) == len(np.unique(union)), "a sample is held by two clients"
            expected = np.flatnonzero(np.isin(idx.labels, classes) & _train_mask(idx))
            assert np.array_equal(np.sort(union), expected), "samples dropped or added"


def test_test_shards_equal_official_test(synthetic_index):
    idx = synthetic_index
    for partition in ("dir", "iid", "natural"):
        scn = build_scenario(idx, _spec(partition=partition))
        for t, classes in enumerate(scn.tasks):
            union = np.concatenate(scn.test_rows[t])
            assert len(union) == len(np.unique(union))
            expected = np.flatnonzero(np.isin(idx.labels, classes) & (idx.split == SPLIT_TEST))
            assert np.array_equal(np.sort(union), expected)
            assert np.array_equal(scn.test_global_rows[t], expected)


def test_test_shard_follows_train_mix(synthetic_index):
    idx = synthetic_index
    scn = build_scenario(idx, _spec(alpha=0.1))
    for t, classes in enumerate(scn.tasks):
        for c in classes:
            n_k = np.array([np.sum(idx.labels[r] == c) for r in scn.train_rows[t]])
            m_k = np.array([np.sum(idx.labels[r] == c) for r in scn.test_rows[t]])
            m, n = m_k.sum(), n_k.sum()
            assert np.all(m_k[n_k == 0] == 0), "a client got test images of a class it never trains on"
            assert np.all(np.abs(m_k - m * n_k / n) < 1)


def test_allocate_test_shards_unit():
    rng = np.random.default_rng(0)
    rows = np.arange(100, 110)
    shards = allocate_test_shards(np.array([5, 0, 15, 0]), rows, rng)
    sizes = [len(s) for s in shards]
    assert sum(sizes) == 10 and sizes[1] == 0 and sizes[3] == 0
    assert sizes[0] in (2, 3) and sizes[2] in (7, 8)
    assert sorted(np.concatenate(shards).tolist()) == rows.tolist()
    with pytest.raises(ValueError):
        allocate_test_shards(np.array([0, 0]), rows, rng)


def test_dirichlet_partition_exact_cover():
    rng = np.random.default_rng(1)
    rows = np.arange(1000)
    parts = dirichlet_partition(rows, 10, 0.1, rng)
    assert len(parts) == 10
    assert np.array_equal(np.sort(np.concatenate(parts)), rows)
    assert all(np.array_equal(p, np.sort(p)) for p in parts)


def test_same_seed_identical_files(tmp_path, synthetic_index):
    pa = save_scenario(build_scenario(synthetic_index, _spec(seed=3)), str(tmp_path / "a"))
    pb = save_scenario(build_scenario(synthetic_index, _spec(seed=3)), str(tmp_path / "b"))
    with open(os.path.join(pa, "config.json")) as f:
        ca = json.load(f)
    with open(os.path.join(pb, "config.json")) as f:
        cb = json.load(f)
    assert ca["files"] == cb["files"] and ca["class_order"] == cb["class_order"]
    for rel in ca["files"]:
        with open(os.path.join(pa, rel), "rb") as fa, open(os.path.join(pb, rel), "rb") as fb:
            assert fa.read() == fb.read(), rel


def test_different_seed_differs(synthetic_index):
    s0 = build_scenario(synthetic_index, _spec(seed=0))
    s1 = build_scenario(synthetic_index, _spec(seed=1))
    same_order = np.array_equal(s0.class_order, s1.class_order)
    same_rows = all(np.array_equal(a, b) for t in range(s0.num_tasks)
                    for a, b in zip(s0.train_rows[t], s1.train_rows[t]))
    assert not (same_order and same_rows)


def test_seed_in_config_name():
    assert get_config_name(_spec(seed=2)) == "cl_5clients_dir_0.3_cpt2_seed2"
    assert get_config_name(_spec(seed=0)) != get_config_name(_spec(seed=1))
    assert get_config_name(_spec(alpha=1.0)) == "cl_5clients_dir_1_cpt2_seed0"
    assert get_config_name(_spec(partition="iid")) == "cl_5clients_iid_cpt2_seed0"
    assert get_config_name(_spec(partition="natural")) == "cl_natural_cpt2_seed0"
    assert get_config_name(_spec(use_val="merge")) == "cl_5clients_dir_0.3_cpt2_seed0_valmerge"
    assert get_config_name(_spec(min_train_per_client=3)) == "cl_5clients_dir_0.3_cpt2_seed0_min3"
    assert get_config_name(_spec(class_order="sorted")) == "cl_5clients_dir_0.3_cpt2_seed0_sorted"
    assert get_config_name(_spec(min_train_per_client=6, class_order="sorted")) == \
        "cl_5clients_dir_0.3_cpt2_seed0_min6_sorted"


def test_sorted_class_order(synthetic_index, tmp_path):
    """sorted: tasks are label ids in order for every seed; the seed still changes the partition; the
    spec round-trips through the saved files, and a default (shuffled) spec is saved without the key, as
    scenario folders written before the option were."""
    s0 = build_scenario(synthetic_index, _spec(class_order="sorted", seed=0))
    s1 = build_scenario(synthetic_index, _spec(class_order="sorted", seed=1))
    C = synthetic_index.num_classes
    assert s0.class_order.tolist() == list(range(C)) == s1.class_order.tolist()
    assert [t.tolist() for t in s0.tasks] == [list(range(i, min(i + 2, C))) for i in range(0, C, 2)]
    assert any(not np.array_equal(a, b) for t in range(s0.num_tasks)
               for a, b in zip(s0.train_rows[t], s1.train_rows[t]))
    path = save_scenario(s0, str(tmp_path))
    assert load_scenario(path).spec.class_order == "sorted"
    with open(os.path.join(path, "config.json")) as f:
        assert json.load(f)["spec"]["class_order"] == "sorted"
    plain = save_scenario(build_scenario(synthetic_index, _spec()), str(tmp_path))
    with open(os.path.join(plain, "config.json")) as f:
        assert "class_order" not in json.load(f)["spec"]
    assert load_scenario(plain).spec.class_order == "shuffled"
    with pytest.raises(ValueError):
        _spec(class_order="alphabetical")


def test_natural_mode(synthetic_index):
    idx = synthetic_index
    scn = build_scenario(idx, _spec(partition="natural", num_clients=99))
    assert scn.num_clients == idx.num_domains == 3
    for t in range(scn.num_tasks):
        for k in range(scn.num_clients):
            assert np.all(idx.domain[scn.train_rows[t][k]] == k)
            assert np.all(idx.domain[scn.test_rows[t][k]] == k)


def test_val_ignored_by_default(synthetic_index):
    idx = synthetic_index
    used = np.concatenate([np.concatenate(r) for r in build_scenario(idx, _spec()).train_rows])
    assert not np.any(idx.split[used] == SPLIT_VAL)
    used_m = np.concatenate([np.concatenate(r) for r in build_scenario(idx, _spec(use_val="merge")).train_rows])
    assert np.sum(idx.split[used_m] == SPLIT_VAL) == np.sum(idx.split == SPLIT_VAL)


def test_min_train_rule(synthetic_index):
    scn = build_scenario(synthetic_index, _spec(partition="iid", min_train_per_client=1))
    for t in range(scn.num_tasks):
        assert all(len(r) >= 1 for r in scn.train_rows[t])
    with pytest.raises(ValueError):
        build_scenario(synthetic_index, _spec(alpha=0.3, min_train_per_client=10 ** 6))
    with pytest.raises(ValueError):
        _spec(partition="natural", min_train_per_client=1)


def test_spec_validation():
    with pytest.raises(ValueError):
        _spec(partition="pat")
    with pytest.raises(ValueError):
        _spec(alpha=0.0)
    with pytest.raises(ValueError):
        _spec(class_per_task=0)
    with pytest.raises(ValueError):
        _spec(use_val="keep")


def test_join_schedule_not_implemented_yet(synthetic_index):
    with pytest.raises(NotImplementedError):
        build_scenario(synthetic_index, _spec(join_task=(0, 0, 1, 1, 1)))


def test_roundtrip_and_wrong_raw_index_refused(tmp_path, synthetic_index):
    scn = build_scenario(synthetic_index, _spec())
    path = save_scenario(scn, str(tmp_path))
    back = load_scenario(path, expected_raw_index_hash=synthetic_index.hash())
    assert back.config_name == scn.config_name and back.num_clients == scn.num_clients
    assert np.array_equal(back.class_order, scn.class_order)
    for t in range(scn.num_tasks):
        assert np.array_equal(back.test_global_rows[t], scn.test_global_rows[t])
        for k in range(scn.num_clients):
            assert np.array_equal(back.train_rows[t][k], scn.train_rows[t][k])
            assert np.array_equal(back.test_rows[t][k], scn.test_rows[t][k])
            assert back.train_rows[t][k].dtype == np.int64
    with pytest.raises(ValueError):
        load_scenario(path, expected_raw_index_hash="0" * 64)


def test_tampered_file_detected(tmp_path, synthetic_index):
    path = save_scenario(build_scenario(synthetic_index, _spec()), str(tmp_path))
    with open(os.path.join(path, "task_0", "train", "0.npy"), "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        load_scenario(path)


def test_existing_config_kept_if_identical_refused_if_different(tmp_path, synthetic_index):
    scn = build_scenario(synthetic_index, _spec())
    p1 = save_scenario(scn, str(tmp_path))
    p2 = save_scenario(scn, str(tmp_path))
    assert p1 == p2
    other_index = SyntheticAdapter(num_classes=7, n_train=20, n_val=4, n_test=5, num_domains=3).build_index()
    with pytest.raises(FileExistsError):
        save_scenario(build_scenario(other_index, _spec()), str(tmp_path))


def test_list_configs(tmp_path, synthetic_index):
    save_scenario(build_scenario(synthetic_index, _spec(seed=0)), str(tmp_path))
    save_scenario(build_scenario(synthetic_index, _spec(seed=1)), str(tmp_path))
    assert list_configs(str(tmp_path)) == ["cl_5clients_dir_0.3_cpt2_seed0", "cl_5clients_dir_0.3_cpt2_seed1"]
