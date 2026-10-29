"""Tests for dataset/utils/cache_utils.py (fp16 feature caches)."""
import os

import numpy as np
import pytest

from utils.cache_utils import (
    FEATURE_FILE,
    PARTIAL_SUFFIX,
    CacheManifest,
    create_cache,
    finalize_cache,
    open_cache,
    sha256_file,
)

RAW = "a" * 64


def _manifest(shape, raw_hash=RAW):
    return CacheManifest(dataset="Synthetic", backbone="fake", group="pre", blocks=[0, 1], tokens=["cls", "mean"],
                         final_ln=False, embed_dim=shape[-1], num_samples=shape[0], shape=list(shape),
                         raw_index_hash=raw_hash)


def _write(group_dir, shape, seed=0):
    arr = create_cache(group_dir, shape)
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float16)
    arr[:] = data
    arr.flush()
    del arr
    return data


def test_roundtrip_bytes(tmp_path):
    g = str(tmp_path / "pre")
    shape = (10, 2, 2, 8)
    data = _write(g, shape)
    man = finalize_cache(g, _manifest(shape))
    feats, man2 = open_cache(g, expected_raw_index_hash=RAW, verify_checksum=True)
    assert feats.dtype == np.float16 and feats.shape == shape
    assert np.array_equal(np.asarray(feats), data)
    assert man.sha256 == man2.sha256 == sha256_file(os.path.join(g, FEATURE_FILE))
    assert not os.path.exists(os.path.join(g, FEATURE_FILE + PARTIAL_SUFFIX))


def test_checksum_mismatch_detected(tmp_path):
    g = str(tmp_path / "pre")
    shape = (4, 2, 2, 8)
    _write(g, shape)
    finalize_cache(g, _manifest(shape))
    with open(os.path.join(g, FEATURE_FILE), "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))
    open_cache(g)                                  # without verification it still opens
    with pytest.raises(ValueError):
        open_cache(g, verify_checksum=True)


def test_index_hash_mismatch_refused(tmp_path):
    g = str(tmp_path / "pre")
    shape = (4, 2, 2, 8)
    _write(g, shape)
    finalize_cache(g, _manifest(shape))
    with pytest.raises(ValueError):
        open_cache(g, expected_raw_index_hash="b" * 64)


def test_readonly(tmp_path):
    g = str(tmp_path / "pre")
    shape = (4, 2, 2, 8)
    _write(g, shape)
    finalize_cache(g, _manifest(shape))
    feats, _ = open_cache(g)
    with pytest.raises((ValueError, OSError)):
        feats[0, 0, 0, 0] = 1


def test_shape_mismatch_refused(tmp_path):
    g = str(tmp_path / "pre")
    _write(g, (4, 2, 2, 8))
    with pytest.raises(ValueError):
        finalize_cache(g, _manifest((5, 2, 2, 8)))


def test_unfinished_cache_refused(tmp_path):
    g = str(tmp_path / "pre")
    _write(g, (4, 2, 2, 8))
    with pytest.raises(FileNotFoundError):
        open_cache(g)
