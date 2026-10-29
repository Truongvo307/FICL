"""Tests for dataset/utils/synthetic_utils.py (the data behind the FICL smoke runs and system tests)."""
import os

import numpy as np
import pytest

from utils import cache_utils, raw_utils, scenario_utils
from utils.synthetic_utils import synthetic_features, write_synthetic_dataset


def _write(root, **kw):
    return write_synthetic_dataset(str(root), raw_utils, scenario_utils, cache_utils, **kw)


def test_written_like_the_real_pipeline_and_idempotent(tmp_path):
    info = _write(tmp_path)
    assert info["scenarios"] == ["cl_5clients_dir_0.5_cpt2_seed0", "cl_natural_cpt2_seed0",
                                 "cl_5clients_dir_0.5_cpt2_seed1", "cl_natural_cpt2_seed1",
                                 "cl_5clients_dir_0.5_cpt2_seed2", "cl_natural_cpt2_seed2"]
    feats, manifest = cache_utils.open_cache(info["cache_dir"], expected_raw_index_hash=info["index"].hash(),
                                                verify_checksum=True)
    assert feats.shape == (640, 4, 2, 16) and manifest.tokens == ["cls", "mean"]
    with open(info["cache_file"], "rb") as f:
        first = f.read()
    again = _write(tmp_path)                        # identical files are kept as they are
    with open(again["cache_file"], "rb") as f:
        assert f.read() == first
    scn = scenario_utils.load_scenario(os.path.join(str(tmp_path), "Synthetic", "cl_natural_cpt2_seed0"),
                                          expected_raw_index_hash=info["index"].hash())
    assert scn.num_clients == 3 and scn.num_tasks == 4


def test_different_parameters_are_refused(tmp_path):
    _write(tmp_path)
    with pytest.raises(FileExistsError):
        _write(tmp_path, class_sep=2.0)
    with pytest.raises(ValueError):
        _write(tmp_path, nonsense=1)


def test_later_blocks_separate_classes_better():
    idx = raw_utils.SyntheticAdapter(num_classes=8, n_train=60, n_val=0, n_test=20, num_domains=3).build_index()
    F = synthetic_features(idx.labels, idx.domain, 8, 3).astype(np.float64)
    train, test = idx.split == 0, idx.split == 2
    acc = []
    for b in range(F.shape[1]):
        Z = F[:, b, 0]
        mu = np.stack([Z[train & (idx.labels == c)].mean(axis=0) for c in range(8)])
        pred = np.argmin(((Z[test][:, None, :] - mu[None]) ** 2).sum(axis=-1), axis=1)
        acc.append(float((pred == idx.labels[test]).mean()))
    assert acc == sorted(acc) and acc[0] < 0.5 < 0.85 < acc[-1] < 1.0     # informative, not saturated


def test_class_images_carry_the_class_and_the_settings_are_in_the_index(tmp_path):
    pytest.importorskip("PIL")
    info = _write(tmp_path)
    idx = raw_utils.load_raw_index(os.path.join(str(tmp_path), "Synthetic", "cl_index"))
    params = idx.meta["adapter_params"]
    assert params["pattern"] == "class" and params["image_size"] == 32 and params["num_domains"] == 3
    adapter = raw_utils.adapter_for_index(str(tmp_path), "Synthetic")
    assert adapter.build_index().hash() == info["index"].hash()
    rows = np.arange(0, idx.num_samples, 5)
    for r in rows[:20]:
        assert adapter.label_and_domain(r) == (idx.labels[r], idx.domain[r])
    X = np.stack([np.asarray(adapter.load_image(r), dtype=np.float64).reshape(-1) for r in rows])
    y, split = idx.labels[rows], idx.split[rows]
    tr, te = split == raw_utils.SPLIT_TRAIN, split == raw_utils.SPLIT_TEST
    mu = np.stack([X[tr & (y == c)].mean(axis=0) for c in range(8)])
    pred = np.argmin(((X[te][:, None, :] - mu[None]) ** 2).sum(axis=-1), axis=1)
    assert (pred == y[te]).mean() > 0.9                  # the class is visible in the pixels


def test_an_index_without_image_settings_is_upgraded_in_place(tmp_path):
    info = _write(tmp_path)
    idx_dir = os.path.join(str(tmp_path), "Synthetic", "cl_index")
    old = raw_utils.load_raw_index(idx_dir)
    del old.meta["adapter_params"]                       # what the index of an earlier version holds
    raw_utils.save_raw_index(old, idx_dir)
    with pytest.raises(ValueError, match="make_synthetic.py"):
        raw_utils.adapter_for_index(str(tmp_path), "Synthetic")
    with open(info["cache_file"], "rb") as f:
        cache_before = f.read()
    _write(tmp_path)                                     # same rows: only index.json gains the settings
    again = raw_utils.load_raw_index(idx_dir)
    assert again.hash() == info["index"].hash() and again.meta["adapter_params"]["pattern"] == "class"
    with open(info["cache_file"], "rb") as f:
        assert f.read() == cache_before
    assert raw_utils.verified_adapter(str(tmp_path), "Synthetic").pattern == "class"
