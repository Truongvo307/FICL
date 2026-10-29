"""Tests for dataset/utils/raw_utils.py (raw sample index)."""
import json
import os

import numpy as np
import pytest

from utils.raw_utils import (
    MEDMNIST_FLAGS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    RawIndex,
    SyntheticAdapter,
    load_raw_index,
    save_raw_index,
    to_rgb_image,
)

DATASET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_order_deterministic():
    a = SyntheticAdapter(seed=0).build_index()
    b = SyntheticAdapter(seed=0).build_index()
    assert a.hash() == b.hash()
    assert np.array_equal(a.labels, b.labels) and a.ids == b.ids
    c = SyntheticAdapter(n_test=7).build_index()
    assert c.hash() != a.hash()


def test_synthetic_layout():
    idx = SyntheticAdapter(num_classes=5, n_train=3, n_val=1, n_test=2, num_domains=2).build_index()
    assert idx.num_classes == 5 and idx.num_domains == 2
    for c in range(5):
        assert np.sum((idx.labels == c) & (idx.split == SPLIT_TRAIN)) == 3
        assert np.sum((idx.labels == c) & (idx.split == SPLIT_VAL)) == 1
        assert np.sum((idx.labels == c) & (idx.split == SPLIT_TEST)) == 2
    assert set(idx.domain.tolist()) == {0, 1}


def test_index_validates_shapes_and_ranges():
    good = dict(name="x", labels=np.array([0, 1], dtype=np.int64), split=np.array([0, 2], dtype=np.int8),
                domain=np.array([-1, -1], dtype=np.int8), ids=["a", "b"], class_names=["c0", "c1"])
    RawIndex(**good)
    with pytest.raises(ValueError):
        RawIndex(**{**good, "split": np.array([0], dtype=np.int8)})
    with pytest.raises(ValueError):
        RawIndex(**{**good, "labels": np.array([0, 5], dtype=np.int64)})
    with pytest.raises(ValueError):
        RawIndex(**{**good, "split": np.array([0, 3], dtype=np.int8)})
    with pytest.raises(ValueError):
        RawIndex(**{**good, "ids": ["a", "a"]})


def test_save_load_roundtrip(tmp_path):
    idx = SyntheticAdapter(num_domains=2).build_index()
    out = str(tmp_path / "cl_index")
    h = save_raw_index(idx, out)
    back = load_raw_index(out)
    assert h == idx.hash() == back.hash()
    assert np.array_equal(back.labels, idx.labels) and back.labels.dtype == np.int64
    assert np.array_equal(back.split, idx.split) and back.split.dtype == np.int8
    assert np.array_equal(back.domain, idx.domain) and back.domain.dtype == np.int8
    assert back.ids == idx.ids and back.class_names == idx.class_names and back.domain_names == idx.domain_names
    with open(os.path.join(out, "index.json")) as f:
        meta = json.load(f)
    assert meta["hash"] == h and meta["num_samples"] == len(idx.labels)


def test_tampered_index_detected(tmp_path):
    idx = SyntheticAdapter().build_index()
    out = str(tmp_path / "cl_index")
    save_raw_index(idx, out)
    labels = np.load(os.path.join(out, "labels.npy"))
    labels[0] = (labels[0] + 1) % idx.num_classes
    np.save(os.path.join(out, "labels.npy"), labels)
    with pytest.raises(ValueError):
        load_raw_index(out)


def test_gray_to_rgb():
    pytest.importorskip("PIL")
    img = to_rgb_image(np.zeros((5, 4), dtype=np.uint8))
    assert img.mode == "RGB" and img.size == (4, 5)
    img = to_rgb_image(np.zeros((5, 4, 1), dtype=np.uint8))
    assert img.mode == "RGB"
    img = to_rgb_image(np.full((5, 4, 3), 7, dtype=np.uint8))
    assert img.mode == "RGB" and np.asarray(img)[0, 0, 0] == 7
    with pytest.raises(ValueError):
        to_rgb_image(np.zeros((5, 4, 2), dtype=np.uint8))


def test_synthetic_images_deterministic():
    pytest.importorskip("PIL")
    a = np.asarray(SyntheticAdapter(seed=0, image_size=16).load_image(3))
    b = np.asarray(SyntheticAdapter(seed=0, image_size=16).load_image(3))
    c = np.asarray(SyntheticAdapter(seed=1, image_size=16).load_image(3))
    assert a.shape == (16, 16, 3) and np.array_equal(a, b) and not np.array_equal(a, c)
    gray = SyntheticAdapter(grayscale=True, image_size=8).load_image(0)
    assert gray.mode == "RGB"


@pytest.mark.data
@pytest.mark.parametrize("name", ["TissueMNIST", "OCTMNIST", "OrganCMNIST", "DermaMNIST"])
def test_medmnist_counts_match_metadata(name):
    medmnist = pytest.importorskip("medmnist")
    idx_dir = os.path.join(DATASET_DIR, name, "cl_index")
    if not os.path.exists(os.path.join(idx_dir, "index.json")):
        pytest.skip(f"run: python prepare_data.py {name} --download")
    idx = load_raw_index(idx_dir)
    info = medmnist.INFO[MEDMNIST_FLAGS[name]]
    for split_name, code in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST)):
        assert int(np.sum(idx.split == code)) == int(info["n_samples"][split_name])
    assert idx.num_classes == len(info["label"])


def _fake_domainnet(root):
    """rawdata/ with the official layout: <domain>/<class>/<file>.png and splits/<domain>_{train,test}.txt."""
    from PIL import Image

    from utils.raw_utils import DOMAINNET_DOMAINS
    raw = os.path.join(str(root), "DomainNet", "rawdata")
    os.makedirs(os.path.join(raw, "splits"))
    classes = ["aircraft_carrier", "zebra"]
    row = 0
    for d in DOMAINNET_DOMAINS:
        for split, n in (("train", 3), ("test", 2)):
            lines = []
            for i in range(n):
                label = i % 2
                rel = f"{d}/{classes[label]}/{d}_{split}_{i}.png"
                os.makedirs(os.path.dirname(os.path.join(raw, rel)), exist_ok=True)
                Image.new("RGB", (4, 4), (row % 256, row // 256, 7)).save(os.path.join(raw, rel))
                lines.append(f"{rel} {label}")
                row += 1
            with open(os.path.join(raw, "splits", f"{d}_{split}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
    return raw, row


def test_domainnet_index_from_official_split_files(tmp_path):
    pytest.importorskip("PIL")
    from utils.raw_utils import DOMAINNET_DOMAINS, DomainNetAdapter
    raw, n = _fake_domainnet(tmp_path)
    adapter = DomainNetAdapter(raw)
    idx = adapter.build_index()
    assert idx.num_samples == n == 30 and idx.class_names == ["aircraft_carrier", "zebra"]
    assert idx.domain_names == list(DOMAINNET_DOMAINS)
    assert idx.domain.tolist() == [d for d in range(6) for _ in range(5)]
    assert idx.split.tolist() == [SPLIT_TRAIN] * 3 + [SPLIT_TEST] * 2 + ([SPLIT_TRAIN] * 3 + [SPLIT_TEST] * 2) * 5
    assert idx.ids[0] == "train:clipart/aircraft_carrier/clipart_train_0.png"
    again = adapter.build_index()                    # building twice gives the same rows and paths
    assert again.hash() == idx.hash() and len(adapter._paths) == n
    for r in (0, 4, 17, 29):
        assert np.asarray(adapter.load_image(r))[0, 0, 0] == r       # each image encodes its row
    with pytest.raises(FileNotFoundError, match="--download"):
        DomainNetAdapter(os.path.join(str(tmp_path), "empty")).build_index()


def test_domainnet_failed_build_leaves_no_partial_paths(tmp_path):
    """A build that stops on a missing split file must not shift the rows of a later build."""
    pytest.importorskip("PIL")
    from utils.raw_utils import DomainNetAdapter
    raw, n = _fake_domainnet(tmp_path)
    missing = os.path.join(raw, "splits", "real_test.txt")
    os.rename(missing, missing + ".away")
    adapter = DomainNetAdapter(raw)
    with pytest.raises(FileNotFoundError):
        adapter.build_index()                        # fails after reading clipart .. quickdraw
    os.rename(missing + ".away", missing)
    adapter.build_index()
    assert len(adapter._paths) == n
    for r in (0, 20, 29):
        assert np.asarray(adapter.load_image(r))[0, 0, 0] == r
