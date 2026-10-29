"""Raw images for algorithms that train a network (INPUT = "images").

ImageStore is the image counterpart of FeatureStore: the same rows of cl_index, the same labels and
the same guard of invariant I5 (test-split rows only inside `with store.evaluation():`). Images are
read through the dataset adapter that built the index (dataset/utils/raw_utils.py, loaded by path),
rebuilt from the settings recorded in index.json and checked against the index first.

    store = ImageStore("CIFAR100", dataset_root="../dataset", num_workers=4)
    store.set_pipeline(pipeline)                 # the network's transforms (flcore/models/transforms.py)
    for x, y, rows in store.train_batches(rows, batch_size=64, key=[seed, client, t, r, epoch]): ...
    with store.evaluation():
        for x in store.prediction_batches(test_rows): ...

Randomness of training batches is fixed by `key` alone: the order of the rows comes from a generator
seeded with key, and the augmentation of each image from a generator seeded with (key, row). The
results therefore do not depend on the number of loader workers or on the order other clients ran in.

This module imports torch only inside the methods that build loaders, so settings checks and dry
runs work without it.
"""
from __future__ import annotations

import math
import os

import numpy as np

from utils import dataset_code
from utils.data_utils import DEFAULT_DATASET_ROOT, GuardedIndex

ORDER_TAG = 0x4F52444552          # "ORDER": generator of the row order of one epoch
_ADAPTERS: dict = {}               # (dataset_root, dataset, index sha256) -> checked adapter, one per process


# ---------------------------------------------------------------------------
# Random augmentation parameters (numpy only; the transforms apply them)
# ---------------------------------------------------------------------------
def random_resized_crop_params(height: int, width: int, rng: np.random.Generator,
                               scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)) -> tuple[int, int, int, int]:
    """(top, left, h, w) of a random resized crop, the algorithm of torchvision 0.15
    RandomResizedCrop.get_params (10 tries, then a centre crop) with draws from `rng`."""
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(rng.uniform(scale[0], scale[1]))
        aspect = math.exp(float(rng.uniform(log_ratio[0], log_ratio[1])))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            return top, left, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def random_crop_params(height: int, width: int, crop: int, rng: np.random.Generator) -> tuple[int, int]:
    """(top, left) of a crop x crop window, uniform over the positions that fit."""
    if crop > height or crop > width:
        raise ValueError(f"a {crop} px crop does not fit a {height} x {width} image")
    return int(rng.integers(0, height - crop + 1)), int(rng.integers(0, width - crop + 1))


def flip_coin(rng: np.random.Generator) -> bool:
    """Horizontal flip with probability 1/2."""
    return bool(rng.random() < 0.5)


# ---------------------------------------------------------------------------
# The dataset adapter of an index
# ---------------------------------------------------------------------------
def open_adapter(dataset_root: str, dataset: str, index_hash: str):
    """The adapter that built dataset_root/<dataset>/cl_index, checked against it and preloaded (once per
    process: every seed of an experiment reuses it)."""
    key = (os.path.abspath(dataset_root), dataset, index_hash)
    if key not in _ADAPTERS:
        raw = dataset_code.load("raw_utils")
        adapter = raw.verified_adapter(dataset_root, dataset)
        adapter.preload()
        _ADAPTERS[key] = adapter
    return _ADAPTERS[key]


class RowImages:
    """Map-style dataset over given rows: item i is (image tensor, label, row) of rows[i].

    A plain class: torch's DataLoader takes any object with __len__ and __getitem__. With train=True the
    image is augmented with a generator seeded by (key, row), so an image's augmentation does not depend
    on which worker loads it or when."""

    def __init__(self, adapter, pipeline, rows: np.ndarray, labels: np.ndarray, train: bool, key=()):
        self.adapter, self.pipeline = adapter, pipeline
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train, self.key = bool(train), [int(k) for k in key]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        img = self.adapter.load_image(row)
        if self.train:
            x = self.pipeline.train(img, np.random.default_rng([*self.key, row]))
        else:
            x = self.pipeline.eval(img)
        return x, int(self.labels[i]), row


# ---------------------------------------------------------------------------
# ImageStore
# ---------------------------------------------------------------------------
class ImageStore(GuardedIndex):
    """Images and labels of one dataset by cl_index row, with the I5 guard."""

    input_kind = "images"

    def __init__(self, dataset: str, dataset_root: str = DEFAULT_DATASET_ROOT, num_workers: int = 0,
                 eval_batch_size: int = 256, pin_memory: bool = False):
        self.dataset, self.dataset_root = dataset, dataset_root
        meta = self._open_index(dataset_root, dataset)
        self.index_meta = meta.get("meta", {})
        self.num_workers, self.eval_batch_size = int(num_workers), int(eval_batch_size)
        self.pin_memory = bool(pin_memory)
        self.adapter = open_adapter(dataset_root, dataset, self.raw_index_hash)
        self.pipeline = None

    @property
    def num_samples(self) -> int:
        return int(len(self._labels))

    def describe(self) -> str:
        return f"the images of {self.dataset}"

    def set_pipeline(self, pipeline) -> None:
        """The network's transforms: pipeline.eval(img) and pipeline.train(img, rng) return image tensors."""
        self.pipeline = pipeline

    def load_image(self, row: int):
        """One raw image (PIL, RGB) by row; test rows only inside evaluation."""
        self._guard(np.asarray([row], dtype=np.int64))
        return self.adapter.load_image(int(row))

    def _require_pipeline(self):
        if self.pipeline is None:
            raise RuntimeError("ImageStore has no transforms yet: the algorithm calls set_pipeline() in "
                               "init_algorithm()")
        return self.pipeline

    def _loader(self, dataset: RowImages, batch_size: int, order: list):
        import torch
        from torch.utils.data import DataLoader

        # a fixed generator: the loader then never draws from torch's global generator (worker seeds only;
        # the augmentation does not use them)
        return DataLoader(dataset, batch_size=int(batch_size), sampler=order, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False, generator=torch.Generator().manual_seed(0))

    def train_batches(self, rows: np.ndarray, batch_size: int, key):
        """Batches (images float32 [b, 3, H, W], labels int64 [b], rows int64 [b]) over training rows, in a
        random order with random augmentation, both fixed by key (a list of ints). The rows are checked
        here, before any image is read."""
        rows = np.asarray(rows, dtype=np.int64)
        labels = self.labels(rows)                        # the guard: training rows only
        pipeline = self._require_pipeline()
        key = [int(k) for k in key]
        order = np.random.default_rng([*key, ORDER_TAG]).permutation(len(rows)).tolist()
        return self._loader(RowImages(self.adapter, pipeline, rows, labels, train=True, key=key), batch_size, order)

    def prediction_batches(self, rows: np.ndarray, chunk: int | None = None):
        """The inputs of predict(): evaluation-transformed images float32 [b, 3, H, W] of rows, in row order.
        The rows are checked here: test rows only inside evaluation."""
        rows = np.asarray(rows, dtype=np.int64)
        labels = self.labels(rows)
        dataset = RowImages(self.adapter, self._require_pipeline(), rows, labels, train=False)
        loader = self._loader(dataset, int(chunk or self.eval_batch_size), list(range(len(rows))))
        return (x for x, _y, _rows in loader)
