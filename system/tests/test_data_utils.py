"""Tests for system/utils/data_utils.py (reading scenarios and caches by path)."""
import json
import os

import numpy as np
import pytest

from conftest import load_dataset_module
from utils.data_utils import (
    DataConfig,
    ClientData,
    FeatureStore,
    FeatureView,
    TestDataAccessError,
    check_compatible,
    get_config_name,
    load_scenario,
    set_data_config,
)


def _cfg(root, **kw):
    base = dict(dataset="Synthetic", partition="dir", num_clients=5, alpha=0.3, class_per_task=2, seed=0,
                dataset_root=root)
    base.update(kw)
    return DataConfig(**base)


def test_name_matches_dataset_side():
    ds = load_dataset_module("scenario_utils.py", "_ds_scenario_utils")
    for partition in ("dir", "iid", "natural"):
        for alpha in (0.1, 1.0, 0.05):
            for cpt in (1, 2, 35):
                for seed in (0, 3):
                    for use_val in ("ignore", "merge"):
                        for m in ((0,) if partition == "natural" else (0, 5)):
                            spec = ds.ScenarioSpec(dataset="X", partition=partition, num_clients=10, alpha=alpha,
                                                   class_per_task=cpt, seed=seed, use_val=use_val,
                                                   min_train_per_client=m)
                            assert get_config_name(partition, 10, alpha, cpt, seed, use_val, m) == \
                                ds.get_config_name(spec)


def test_set_config_finds_folder_and_lists_alternatives(fake_dataset_root):
    root = fake_dataset_root["root"]
    cfg = _cfg(root)
    set_data_config(cfg)
    assert cfg.config_name == "cl_5clients_dir_0.3_cpt2_seed0"
    assert cfg.num_classes == 7 and cfg.num_tasks == 4 and cfg.num_clients == 5
    with pytest.raises(FileNotFoundError, match="cl_5clients_dir_0.3_cpt2_seed0"):
        set_data_config(_cfg(root, seed=9))


def test_missing_scenario_hint_is_a_runnable_command(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        set_data_config(_cfg(str(tmp_path), dataset="TissueMNIST", partition="natural", alpha=0.0, num_clients=0))
    assert "python generate_scenario.py TissueMNIST natural --class_per_task 2 --seed 0" in str(e.value)
    assert "--alpha" not in str(e.value) and "--num_clients" not in str(e.value)
    with pytest.raises(FileNotFoundError) as e:
        set_data_config(_cfg(str(tmp_path), dataset="TissueMNIST", alpha=0.1, use_val="merge"))
    assert ("python generate_scenario.py TissueMNIST dir --num_clients 5 --alpha 0.1 --class_per_task 2 --seed 0 "
            "--use_val merge") in str(e.value)
    with pytest.raises(FileNotFoundError, match="make_synthetic.py"):
        set_data_config(_cfg(str(tmp_path)))
    assert "python prepare_data.py TissueMNIST --download  (there is no index yet)" in str(e.value)


def test_content_hash_ignores_provenance_but_not_content(fake_dataset_root, tmp_path):
    import shutil

    src = os.path.join(fake_dataset_root["root"], "Synthetic", "cl_5clients_dir_0.3_cpt2_seed0")
    dst = os.path.join(str(tmp_path), "copy")
    shutil.copytree(src, dst)
    path = os.path.join(dst, "config.json")
    with open(path) as f:
        meta = json.load(f)
    meta.update(created="2000-01-01T00:00:00+00:00", versions={"numpy": "1.26.4"}, git_commit="0" * 40)
    with open(path, "w") as f:
        json.dump(meta, f)
    a, b = load_scenario(src), load_scenario(dst)
    assert len(a.content_sha256) == 64 and a.content_sha256 == b.content_sha256
    meta["files"]["task_0/train/0.npy"] = "0" * 64
    with open(path, "w") as f:
        json.dump(meta, f)
    assert load_scenario(dst, verify=False).content_sha256 != a.content_sha256


def test_natural_takes_num_clients_from_config(fake_dataset_root):
    cfg = _cfg(fake_dataset_root["root"], partition="natural", num_clients=99)
    set_data_config(cfg)
    assert cfg.config_name == "cl_natural_cpt2_seed0" and cfg.num_clients == 3


def test_rows_via_scenario_equal_direct(fake_dataset_root):
    root, index = fake_dataset_root["root"], fake_dataset_root["index"]
    scn = load_scenario(_cfg(root))
    store = FeatureStore("Synthetic", "fake", FeatureView(group="pre"), dataset_root=root)
    check_compatible(scn, store)
    feats = np.load(fake_dataset_root["cache_file"], mmap_mode="r")
    cd = ClientData(scn, 2)
    for t in range(scn.num_tasks):
        cd.begin_task(t)
        rows = cd.train_rows()
        z = store.features(rows)                       # a client may hold no samples in a task
        assert z.dtype == np.float64 and z.shape == (len(rows), 4 * 2 * 8)
        assert np.array_equal(z, feats[rows].reshape(len(rows), 4 * 2 * 8).astype(np.float64))
        assert np.array_equal(store.labels(rows), index.labels[rows])


def test_view_selects_blocks_and_tokens(fake_dataset_root):
    root = fake_dataset_root["root"]
    scn = load_scenario(_cfg(root))
    view = FeatureView(group="pre", blocks=(1, 3), tokens=("mean",))
    store = FeatureStore("Synthetic", "fake", view, dataset_root=root)
    rows = scn.train_rows[0][0]
    feats = np.load(fake_dataset_root["cache_file"], mmap_mode="r")
    expected = feats[rows][:, [1, 3]][:, :, [1]].reshape(len(rows), -1).astype(np.float64)
    assert np.array_equal(store.features(rows), expected)
    assert store.dim == 16 and store.depth_groups == [8, 8]


def test_unknown_block_or_token_refused(fake_dataset_root):
    root = fake_dataset_root["root"]
    with pytest.raises(ValueError):
        FeatureStore("Synthetic", "fake", FeatureView(group="pre", blocks=(7,)), dataset_root=root)
    with pytest.raises(ValueError):
        FeatureStore("Synthetic", "fake", FeatureView(group="pre", tokens=("max",)), dataset_root=root)
    with pytest.raises(FileNotFoundError):
        FeatureStore("Synthetic", "fake", FeatureView(group="nope"), dataset_root=root)


def test_projection_deterministic_and_shaped(fake_dataset_root):
    root = fake_dataset_root["root"]
    rows = np.arange(10)
    a = FeatureStore("Synthetic", "fake", FeatureView(group="pre", proj_rank=4, proj_seed=11), dataset_root=root)
    b = FeatureStore("Synthetic", "fake", FeatureView(group="pre", proj_rank=4, proj_seed=11), dataset_root=root)
    c = FeatureStore("Synthetic", "fake", FeatureView(group="pre", proj_rank=4, proj_seed=12), dataset_root=root)
    za, zb, zc = a.features(rows), b.features(rows), c.features(rows)
    assert za.shape == (10, 16) and a.depth_groups == [4, 4, 4, 4]
    assert np.array_equal(za, zb) and not np.array_equal(za, zc)
    with pytest.raises(ValueError):
        FeatureView(group="pre", proj_rank=4)          # a projection needs an explicit seed


def test_mismatch_refused(fake_dataset_root):
    root = fake_dataset_root["root"]
    scn = load_scenario(_cfg(root))
    store = FeatureStore("Synthetic", "fake", FeatureView(group="pre"), dataset_root=root)
    scn.raw_index_hash = "0" * 64
    with pytest.raises(ValueError):
        check_compatible(scn, store)


def test_store_refuses_cache_of_another_index(fake_dataset_root):
    root, cache = fake_dataset_root["root"], fake_dataset_root["cache"]
    group = os.path.join(root, "Synthetic", "features", "fake", "other")
    arr = cache.create_cache(group, (3, 1, 1, 8))
    arr[:] = 0
    arr.flush()
    del arr
    cache.finalize_cache(group, cache.CacheManifest(
        dataset="Synthetic", backbone="fake", group="other", blocks=[0], tokens=["cls"], final_ln=False,
        embed_dim=8, num_samples=3, shape=[3, 1, 1, 8], raw_index_hash="f" * 64))
    with pytest.raises(ValueError):
        FeatureStore("Synthetic", "fake", FeatureView(group="other"), dataset_root=root)


def test_no_past_train_access(fake_dataset_root):
    scn = load_scenario(_cfg(fake_dataset_root["root"]))
    cd = ClientData(scn, 0)
    with pytest.raises(RuntimeError):
        cd.train_rows()                                 # no task has started
    cd.begin_task(0)
    cd.begin_task(1)
    with pytest.raises(PermissionError):
        cd.past_train_rows(0)
    with pytest.raises(ValueError):
        cd.test_rows(2)                                 # a future task
    assert np.array_equal(cd.test_rows(0), scn.test_rows[0][0])
    up = ClientData(scn, 0, past_train_access=True)
    up.begin_task(0)
    up.begin_task(1)
    assert np.array_equal(up.past_train_rows(0), scn.train_rows[0][0])


def test_tasks_advance_in_order(fake_dataset_root):
    scn = load_scenario(_cfg(fake_dataset_root["root"]))
    cd = ClientData(scn, 0)
    with pytest.raises(ValueError):
        cd.begin_task(1)
    cd.begin_task(0)
    with pytest.raises(ValueError):
        cd.begin_task(0)
    with pytest.raises(ValueError):
        cd.begin_task(2)


def test_iter_chunks_covers_rows(fake_dataset_root):
    root = fake_dataset_root["root"]
    store = FeatureStore("Synthetic", "fake", FeatureView(group="pre"), dataset_root=root)
    rows = np.arange(0, 40, 3)
    parts = list(store.iter_chunks(rows, chunk=4))
    assert [len(z) for z, _ in parts] == [4, 4, 4, 2]
    assert np.array_equal(np.concatenate([z for z, _ in parts]), store.features(rows))
    assert np.array_equal(np.concatenate([y for _, y in parts]), store.labels(rows))


def test_load_scenario_verifies_files(fake_dataset_root):
    root = fake_dataset_root["root"]
    cfg = _cfg(root)
    set_data_config(cfg)
    with open(os.path.join(cfg.config_dir, "task_1", "test", "0.npy"), "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        load_scenario(cfg)


def test_reader_matches_writer_format(fake_dataset_root):
    root = fake_dataset_root["root"]
    cfg = _cfg(root)
    scn = load_scenario(cfg)
    with open(os.path.join(cfg.config_dir, "config.json")) as f:
        meta = json.load(f)
    assert scn.class_order.tolist() == meta["class_order"]
    assert [t.tolist() for t in scn.tasks] == meta["tasks"]
    assert scn.num_tasks == meta["num_tasks"] == 4


def test_test_rows_need_an_evaluation_context(fake_dataset_root):
    """Invariant I5 at the data layer: training code cannot read test-split features or labels."""
    root = fake_dataset_root["root"]
    scn = load_scenario(_cfg(root))
    store = FeatureStore("Synthetic", "fake", FeatureView(group="pre"), dataset_root=root)
    test_rows = scn.test_global_rows[0]
    with pytest.raises(TestDataAccessError):
        store.features(test_rows)
    with pytest.raises(TestDataAccessError):
        store.labels(test_rows[:1])
    with pytest.raises(TestDataAccessError):
        store.features(np.concatenate([scn.train_rows[0][0], test_rows[:1]]))
    assert store.in_evaluation is False
    with store.evaluation():
        assert store.in_evaluation is True
        assert store.features(test_rows).shape[0] == len(test_rows)
        with store.evaluation():
            store.labels(test_rows)
        store.labels(test_rows)                       # still inside the outer context
    assert store.in_evaluation is False
    with pytest.raises(TestDataAccessError):
        store.labels(test_rows)
    store.features(scn.train_rows[0][0])              # training rows stay readable


def test_verify_checksum(fake_dataset_root):
    root = fake_dataset_root["root"]
    store = FeatureStore("Synthetic", "fake", FeatureView(group="pre"), dataset_root=root)
    assert store.verify_checksum() is True
    with open(fake_dataset_root["cache_file"], "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))
    with pytest.raises(ValueError):
        store.verify_checksum()
