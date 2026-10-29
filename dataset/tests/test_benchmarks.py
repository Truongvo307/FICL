"""Tests for the standard benchmarks in dataset/utils/raw_utils.py.

Offline: each test writes a small fake copy of the dataset's layout. Images are PNG files whose red
value encodes their row, so the tests can check that row i of the index loads image i.
"""
import os
import pickle
import tarfile
import zipfile

import numpy as np
import pytest

import prepare_data
from utils.raw_utils import (
    BENCHMARKS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    CIFAR100Adapter,
    CUB200Adapter,
    ImageFolderAdapter,
    TinyImageNetAdapter,
    _safe_extract,
    available_datasets,
    download_benchmark,
    get_adapter,
    has_seeded_split,
    load_raw_index,
    seeded_stratified_split,
)


def _png(path, value, mode="RGB"):
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new(mode, (5, 4), value if mode == "L" else (value, 1, 2)).save(path)


def _red(img):
    return int(np.asarray(img)[0, 0, 0])


def _fake_eurosat(raw, per_class=5, top="EuroSAT_RGB"):
    """10 class folders under raw/<top>/, per_class images each; returns the class names."""
    classes = [f"Class{c:02d}" for c in range(10)]
    row = 0
    for c in classes:
        for i in range(per_class):
            _png(os.path.join(raw, top, c, f"{c}_{i}.png"), row)
            row += 1
    with open(os.path.join(raw, top, "README.txt"), "w") as f:
        f.write("not a class folder\n")
    return classes


def test_registry_lists_every_benchmark():
    names = available_datasets()
    for name, b in BENCHMARKS.items():
        assert name in names
        assert b["split"] in ("official", "seeded") and has_seeded_split(name) == (b["split"] == "seeded")
        assert b["classes"] > 0 and b["images"] > 0 and b["licence"] and b["homepage"] and b["overlap"]
        assert (b["download"] is None) == ("manual" in b)
    assert {n for n, b in BENCHMARKS.items() if b["split"] == "seeded"} == {"ImageNetR", "ImageNetA", "EuroSAT",
                                                                             "RESISC45"}


def test_seeded_split_is_stratified_and_deterministic():
    labels = np.repeat(np.arange(5), [10, 13, 5, 2, 1])
    a = seeded_stratified_split(labels, 0.2, seed=0)
    assert np.array_equal(a, seeded_stratified_split(labels, 0.2, seed=0))
    n_test = [int((a[labels == c] == SPLIT_TEST).sum()) for c in range(5)]
    assert n_test == [2, 3, 1, 1, 0]              # floor(n * 0.2 + 0.5), at least 1 and at most n - 1
    assert not np.array_equal(a, seeded_stratified_split(labels, 0.2, seed=1))
    with pytest.raises(ValueError, match="test_fraction"):
        seeded_stratified_split(labels, 1.0)


def test_image_folder_adapter_finds_the_class_folders(tmp_path):
    pytest.importorskip("PIL")
    raw = os.path.join(str(tmp_path), "EuroSAT", "rawdata")
    classes = _fake_eurosat(raw)
    adapter = ImageFolderAdapter("EuroSAT", raw)
    idx = adapter.build_index()
    assert idx.class_names == classes and idx.num_samples == 50
    assert idx.ids[0] == "Class00/Class00_0.png" and idx.labels.tolist() == [c for c in range(10) for _ in range(5)]
    assert [int((idx.split[idx.labels == c] == SPLIT_TEST).sum()) for c in range(10)] == [1] * 10
    assert idx.meta["split"]["test_fraction"] == 0.2 and idx.meta["split"]["seed"] == 0
    assert idx.meta["class_root"] == "EuroSAT_RGB" and "MIT" in idx.meta["licence"]
    for r in (0, 17, 49):
        img = adapter.load_image(r)
        assert img.mode == "RGB" and _red(img) == r
    assert ImageFolderAdapter("EuroSAT", raw).build_index().hash() == idx.hash()
    assert ImageFolderAdapter("EuroSAT", raw, split_seed=1).build_index().hash() != idx.hash()


def test_image_folder_layout_is_found_below_extra_folders(tmp_path):
    pytest.importorskip("PIL")
    raw = os.path.join(str(tmp_path), "EuroSAT", "rawdata")
    _fake_eurosat(os.path.join(raw, "extra", "level"), per_class=2, top="2750")
    idx = ImageFolderAdapter("EuroSAT", raw).build_index()
    assert idx.num_samples == 20 and idx.meta["class_root"] == os.path.join("extra", "level", "2750")


def test_missing_raw_files_say_how_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        ImageFolderAdapter("ImageNetR", os.path.join(str(tmp_path), "none")).build_index()
    with pytest.raises(FileNotFoundError, match="NWPU-RESISC45.rar"):
        ImageFolderAdapter("RESISC45", os.path.join(str(tmp_path), "none")).build_index()
    with pytest.raises(SystemExit, match="no automatic download"):
        download_benchmark("RESISC45", os.path.join(str(tmp_path), "raw"))
    with pytest.raises(ValueError, match="official split"):
        ImageFolderAdapter("CUB200", str(tmp_path))


def test_tiny_imagenet_uses_the_labelled_val_set_as_test(tmp_path):
    pytest.importorskip("PIL")
    root = os.path.join(str(tmp_path), "TinyImageNet", "rawdata", "tiny-imagenet-200")
    wnids = ["n02", "n01"]
    os.makedirs(root)
    with open(os.path.join(root, "wnids.txt"), "w") as f:
        f.write("\n".join(wnids) + "\n")
    row = 0
    for w in sorted(wnids):
        for i in range(3):
            _png(os.path.join(root, "train", w, "images", f"{w}_{i}.png"), row)
            row += 1
    lines = []
    for i, w in enumerate(["n02", "n01", "n02"]):
        _png(os.path.join(root, "val", "images", f"val_{i}.png"), 100 + i)
        lines.append(f"val_{i}.png\t{w}\t0\t0\t4\t4")
    with open(os.path.join(root, "val", "val_annotations.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    adapter = TinyImageNetAdapter(os.path.dirname(root))
    idx = adapter.build_index()
    assert idx.class_names == ["n01", "n02"]
    assert idx.split.tolist() == [SPLIT_TRAIN] * 6 + [SPLIT_TEST] * 3
    assert idx.labels.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 1]
    assert idx.ids[6] == "val/val_0.png"
    assert [_red(adapter.load_image(r)) for r in (0, 5, 6, 8)] == [0, 5, 100, 102]


def test_cub_uses_the_official_split_files(tmp_path):
    pytest.importorskip("PIL")
    root = os.path.join(str(tmp_path), "CUB200", "rawdata", "CUB_200_2011")
    rels = ["001.Albatross/a_1.png", "001.Albatross/a_2.png", "002.Auklet/b_1.png", "002.Auklet/b_2.png"]
    for i, rel in enumerate(rels):
        _png(os.path.join(root, "images", rel), 10 + i, mode="L" if i == 3 else "RGB")
    files = {"images.txt": [f"{i + 1} {r}" for i, r in enumerate(rels)],
             "image_class_labels.txt": ["1 1", "2 1", "3 2", "4 2"],
             "train_test_split.txt": ["1 1", "2 0", "3 1", "4 0"],
             "classes.txt": ["1 001.Albatross", "2 002.Auklet"]}
    for name, lines in files.items():
        with open(os.path.join(root, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    adapter = CUB200Adapter(os.path.dirname(root))
    idx = adapter.build_index()
    assert idx.class_names == ["001.Albatross", "002.Auklet"] and idx.labels.tolist() == [0, 0, 1, 1]
    assert idx.split.tolist() == [SPLIT_TRAIN, SPLIT_TEST, SPLIT_TRAIN, SPLIT_TEST] and idx.ids == rels
    gray = adapter.load_image(3)
    assert gray.mode == "RGB" and _red(gray) == 13
    assert "non-commercial" in idx.meta["licence"]


def test_cifar100_reads_the_official_pickles(tmp_path):
    root = os.path.join(str(tmp_path), "CIFAR100", "rawdata", "cifar-100-python")
    os.makedirs(root)

    def part(n, labels, first_value):
        data = np.zeros((n, 3072), dtype=np.uint8)
        data[0, :1024], data[0, 1024:2048], data[0, 2048:] = first_value, first_value + 1, first_value + 2
        return {b"data": data, b"fine_labels": labels}

    for name, obj in (("meta", {b"fine_label_names": [b"apple", b"bear"]}),
                      ("train", part(3, [0, 1, 1], 10)), ("test", part(2, [1, 0], 50))):
        with open(os.path.join(root, name), "wb") as f:
            pickle.dump(obj, f)
    adapter = CIFAR100Adapter(os.path.dirname(root))
    idx = adapter.build_index()
    assert idx.class_names == ["apple", "bear"] and idx.labels.tolist() == [0, 1, 1, 1, 0]
    assert idx.split.tolist() == [SPLIT_TRAIN] * 3 + [SPLIT_TEST] * 2 and idx.ids[3] == "test/0"
    pytest.importorskip("PIL")
    assert np.asarray(adapter.load_image(0))[0, 0].tolist() == [10, 11, 12]      # channel order R, G, B
    assert np.asarray(adapter.load_image(3))[0, 0].tolist() == [50, 51, 52] and adapter.load_image(3).size == (32, 32)


def test_archives_cannot_write_outside_the_target(tmp_path):
    base = str(tmp_path)
    evil = os.path.join(base, "evil.txt")
    with open(evil, "w") as f:
        f.write("x")
    tar_path = os.path.join(base, "bad.tar")
    with tarfile.open(tar_path, "w") as t:
        t.add(evil, arcname="../evil.txt")
    with pytest.raises(RuntimeError, match="outside"):
        _safe_extract(tar_path, os.path.join(base, "out1"))
    link_tar = os.path.join(base, "link.tar")
    with tarfile.open(link_tar, "w") as t:
        info = tarfile.TarInfo("inside/link")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        t.addfile(info)
    with pytest.raises(RuntimeError, match="links"):
        _safe_extract(link_tar, os.path.join(base, "out2"))
    zip_path = os.path.join(base, "bad.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError, match="outside"):
        _safe_extract(zip_path, os.path.join(base, "out3"))
    good = os.path.join(base, "good.tar")
    with tarfile.open(good, "w") as t:
        t.add(evil, arcname="folder/ok.txt")
    _safe_extract(good, os.path.join(base, "out4"))
    assert os.path.exists(os.path.join(base, "out4", "folder", "ok.txt"))


def test_download_unpacks_once_from_a_local_archive(tmp_path):
    pytest.importorskip("PIL")
    src = os.path.join(str(tmp_path), "src")
    _fake_eurosat(src, per_class=2)
    archive = os.path.join(str(tmp_path), "EuroSAT_RGB.zip")
    with zipfile.ZipFile(archive, "w") as z:
        for dirpath, _dirs, files in os.walk(src):
            for fname in files:
                full = os.path.join(dirpath, fname)
                z.write(full, os.path.relpath(full, src))
    raw = os.path.join(str(tmp_path), "EuroSAT", "rawdata")
    saved = BENCHMARKS["EuroSAT"]["download"]
    BENCHMARKS["EuroSAT"]["download"] = [("file://" + archive, "EuroSAT_RGB.zip", "0" * 32)]   # wrong md5: a warning
    try:
        md5s = download_benchmark("EuroSAT", raw)
        assert list(md5s) == ["EuroSAT_RGB.zip"] and os.path.exists(os.path.join(raw, ".unpacked_EuroSAT_RGB.zip"))
        assert download_benchmark("EuroSAT", raw) == {}                    # already unpacked: nothing to do
        idx = get_adapter("EuroSAT", os.path.join(str(tmp_path))).build_index()
        assert idx.num_samples == 20 and idx.meta["class_root"] == "EuroSAT_RGB"
    finally:
        BENCHMARKS["EuroSAT"]["download"] = saved


def test_prepare_data_split_options(tmp_path):
    pytest.importorskip("PIL")
    root = str(tmp_path)
    with pytest.raises(SystemExit, match="official split"):
        prepare_data.main(["CUB200", "--split_seed", "1", "--dataset_root", root])
    _fake_eurosat(os.path.join(root, "EuroSAT", "rawdata"))
    assert prepare_data.main(["EuroSAT", "--dataset_root", root]) == 0
    first = load_raw_index(os.path.join(root, "EuroSAT", "cl_index"))
    assert first.meta["split"]["seed"] == 0
    assert prepare_data.main(["EuroSAT", "--dataset_root", root]) == 0              # same index: nothing to do
    with pytest.raises(SystemExit, match="different index"):
        prepare_data.main(["EuroSAT", "--split_seed", "1", "--dataset_root", root])
    assert prepare_data.main(["EuroSAT", "--split_seed", "1", "--force", "--dataset_root", root]) == 0
    assert load_raw_index(os.path.join(root, "EuroSAT", "cl_index")).meta["split"]["seed"] == 1
    with pytest.raises(SystemExit, match="--download"):
        prepare_data.main(["ImageNetA", "--dataset_root", root])


def test_adapter_from_the_index_and_changed_raw_files(tmp_path):
    pytest.importorskip("PIL")
    root = str(tmp_path)
    raw = os.path.join(root, "EuroSAT", "rawdata")
    _fake_eurosat(raw)
    assert prepare_data.main(["EuroSAT", "--split_seed", "3", "--test_fraction", "0.4", "--dataset_root", root]) == 0
    from utils.raw_utils import adapter_for_index, verified_adapter

    adapter = adapter_for_index(root, "EuroSAT")                 # the split rule comes from index.json
    assert (adapter.test_fraction, adapter.split_seed) == (0.4, 3)
    checked = verified_adapter(root, "EuroSAT")
    assert [_red(checked.load_image(r)) for r in (0, 7, 49)] == [0, 7, 49]
    _png(os.path.join(raw, "EuroSAT_RGB", "Class03", "Class03_0a.png"), 200)    # a file appears: rows shift
    with pytest.raises(ValueError, match="no longer give the rows"):
        verified_adapter(root, "EuroSAT")


def test_downloads_resume_after_a_cut_connection(tmp_path):
    """A local HTTP server that closes the connection half way the first time and honours Range requests."""
    import http.server
    import threading

    from utils.raw_utils import _download

    payload = bytes(range(256)) * 400
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append((self.headers.get("User-Agent"), self.headers.get("Range")))
            start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
            body = payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:len(body) // 2] if len(seen) == 1 else body)    # the first answer is cut

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("no_proxy", "NO_PROXY")}
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"     # never through a proxy
    try:
        target = os.path.join(str(tmp_path), "archive.bin")
        _download(f"http://127.0.0.1:{server.server_port}/archive.bin", target, retries=3, wait=0)
    finally:
        server.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    with open(target, "rb") as f:
        assert f.read() == payload
    assert len(seen) == 2 and seen[1][1] == f"bytes={len(payload) // 2}-" and "Mozilla" in seen[0][0]
    assert not os.path.exists(target + ".part")
