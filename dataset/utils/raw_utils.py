"""Raw sample index for FICL.

A RawIndex is the fixed, ordered list of every sample of a dataset: its label, its
official split (train / val / test) and its domain. Every later file refers to samples
by their row in this index: the feature caches store one row per sample in this order,
and the scenarios store row indices. index.json records a sha256 of the index, so a
cache or a scenario built for another index is refused.

This module must stay standalone (numpy + Pillow, no `utils.*` imports): the system-side
tests load it by file path.

Layout written by save_raw_index:
    dataset/<Name>/cl_index/labels.npy   int64 [N]
                            split.npy    int8  [N]   0 train, 1 val, 2 test
                            domain.npy   int8  [N]   -1 when the dataset has no domains
                            ids.txt      one stable sample id per line, row order
                            index.json   names, counts, source, licence, sha256
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}
INDEX_DIRNAME = "cl_index"
INDEX_FORMAT = "fcl-raw-index/1"

# Name used with -data (CamelCase, as in HtFLlib) -> medmnist flag
MEDMNIST_FLAGS = {
    "TissueMNIST": "tissuemnist",
    "OCTMNIST": "octmnist",
    "OrganCMNIST": "organcmnist",
    "DermaMNIST": "dermamnist",
    "PathMNIST": "pathmnist",
    "BloodMNIST": "bloodmnist",
}
# Domain order: it fixes the client order of natural DomainNet scenarios (client k = domain k)
DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
# Download locations, as used by HtFLlib / PFLlib's generate_domainnet.py (some zips sit under groundtruth/)
DOMAINNET_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/"
DOMAINNET_ZIPS = {
    "clipart": DOMAINNET_URL + "groundtruth/clipart.zip",
    "infograph": DOMAINNET_URL + "infograph.zip",
    "painting": DOMAINNET_URL + "groundtruth/painting.zip",
    "quickdraw": DOMAINNET_URL + "quickdraw.zip",
    "real": DOMAINNET_URL + "real.zip",
    "sketch": DOMAINNET_URL + "sketch.zip",
}
DOMAINNET_SPLITS = DOMAINNET_URL + "domainnet/txt/{domain}_{split}.txt"


# ---------------------------------------------------------------------------
# RawIndex
# ---------------------------------------------------------------------------
@dataclass
class RawIndex:
    """Ordered list of all samples. Row i of every cache and scenario refers to sample i."""

    name: str
    labels: np.ndarray                 # int64 [N], classes 0..C-1
    split: np.ndarray                  # int8 [N], SPLIT_TRAIN / SPLIT_VAL / SPLIT_TEST
    domain: np.ndarray                 # int8 [N], 0..num_domains-1 or -1
    ids: list[str]                     # stable sample ids in row order
    class_names: list[str]
    domain_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.ascontiguousarray(self.labels, dtype=np.int64)
        self.split = np.ascontiguousarray(self.split, dtype=np.int8)
        self.domain = np.ascontiguousarray(self.domain, dtype=np.int8)
        n = len(self.labels)
        if self.labels.ndim != 1 or self.split.shape != (n,) or self.domain.shape != (n,) or len(self.ids) != n:
            raise ValueError("labels, split, domain and ids must all have one entry per sample")
        if n and (self.labels.min() < 0 or self.labels.max() >= len(self.class_names)):
            raise ValueError("labels must lie in 0..len(class_names)-1")
        if n and not np.isin(self.split, list(SPLIT_NAMES)).all():
            raise ValueError("split values must be 0 (train), 1 (val) or 2 (test)")
        if self.domain_names:
            if n and (self.domain.min() < 0 or self.domain.max() >= len(self.domain_names)):
                raise ValueError("domain values must lie in 0..len(domain_names)-1")
        elif n and not (self.domain == -1).all():
            raise ValueError("domain must be -1 everywhere when there are no domain_names")
        if len(set(self.ids)) != n:
            raise ValueError("sample ids must be unique")

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_domains(self) -> int:
        return len(self.domain_names)

    def hash(self) -> str:
        """sha256 over name, labels, splits, domains, ids and names (platform independent)."""
        h = hashlib.sha256()
        h.update(INDEX_FORMAT.encode())
        h.update(self.name.encode())
        h.update(self.labels.astype("<i8").tobytes())
        h.update(self.split.astype("<i1").tobytes())
        h.update(self.domain.astype("<i1").tobytes())
        h.update("\n".join(self.ids).encode())
        h.update("\n".join(self.class_names).encode())
        h.update("\n".join(self.domain_names).encode())
        return h.hexdigest()

    def counts(self) -> dict:
        """Samples per split per class (and per domain when there are domains)."""
        out = {}
        for code, sname in SPLIT_NAMES.items():
            m = self.split == code
            out[sname] = {
                "total": int(m.sum()),
                "per_class": np.bincount(self.labels[m], minlength=self.num_classes).astype(int).tolist(),
            }
            if self.domain_names:
                out[sname]["per_domain"] = np.bincount(self.domain[m], minlength=self.num_domains).astype(int).tolist()
        return out


def index_dir(dataset_root: str, name: str) -> str:
    return os.path.join(dataset_root, name, INDEX_DIRNAME)


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_raw_index(idx: RawIndex, out_dir: str) -> str:
    """Write the index files (see module docstring). Returns the index hash."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "labels.npy"), idx.labels)
    np.save(os.path.join(out_dir, "split.npy"), idx.split)
    np.save(os.path.join(out_dir, "domain.npy"), idx.domain)
    with open(os.path.join(out_dir, "ids.txt"), "w") as f:
        f.write("\n".join(idx.ids))
    h = idx.hash()
    _atomic_write_json(os.path.join(out_dir, "index.json"), {
        "format": INDEX_FORMAT,
        "name": idx.name,
        "hash": h,
        "num_samples": idx.num_samples,
        "num_classes": idx.num_classes,
        "class_names": idx.class_names,
        "num_domains": idx.num_domains,
        "domain_names": idx.domain_names,
        "counts": idx.counts(),
        "meta": idx.meta,
    })
    return h


def load_raw_index(out_dir: str, verify: bool = True) -> RawIndex:
    """Read an index written by save_raw_index. With verify, the stored sha256 must match."""
    with open(os.path.join(out_dir, "index.json")) as f:
        meta = json.load(f)
    if meta.get("format") != INDEX_FORMAT:
        raise ValueError(f"{out_dir}: unknown index format {meta.get('format')!r}")
    with open(os.path.join(out_dir, "ids.txt")) as f:
        text = f.read()
    ids = text.split("\n") if text else []
    idx = RawIndex(
        name=meta["name"],
        labels=np.load(os.path.join(out_dir, "labels.npy"), allow_pickle=False),
        split=np.load(os.path.join(out_dir, "split.npy"), allow_pickle=False),
        domain=np.load(os.path.join(out_dir, "domain.npy"), allow_pickle=False),
        ids=ids,
        class_names=list(meta["class_names"]),
        domain_names=list(meta.get("domain_names", [])),
        meta=dict(meta.get("meta", {})),
    )
    if verify and idx.hash() != meta["hash"]:
        raise ValueError(f"{out_dir}: index files do not match the hash in index.json (edited or corrupted)")
    return idx


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def to_rgb_image(arr: np.ndarray):
    """uint8 array [H, W], [H, W, 1] or [H, W, 3] -> RGB PIL image (gray is copied to 3 channels)."""
    from PIL import Image

    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {a.dtype}")
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if not (a.ndim == 2 or (a.ndim == 3 and a.shape[-1] == 3)):
        raise ValueError(f"cannot convert an array of shape {a.shape} to an RGB image")
    img = Image.fromarray(a)          # uint8 [H, W] -> "L", [H, W, 3] -> "RGB"
    return img if img.mode == "RGB" else img.convert("RGB")


# ---------------------------------------------------------------------------
# Adapters: one per dataset. build_index() gives the RawIndex, load_image(row) the
# original image (RGB, no resizing or normalization: each backbone preprocesses).
# ---------------------------------------------------------------------------
SYNTHETIC_PATTERNS = ("noise", "class")


class SyntheticAdapter:
    """Tiny deterministic dataset for tests. Classes cycle over domains.

    pattern "noise": every image is uniform noise, with no class information (feature-extraction tests).
    pattern "class": a bright square on dark noise; the square's position and colour depend on the class,
    every domain adds a colour tint and every image a small random shift. A small CNN learns the classes
    in a few epochs (tests and smoke runs of the image path). The parameters are written into the index
    (meta.adapter_params), so the system side can rebuild the same adapter.
    """

    def __init__(self, num_classes: int = 8, n_train: int = 20, n_val: int = 0, n_test: int = 6,
                 num_domains: int = 0, image_size: int = 32, grayscale: bool = False, seed: int = 0,
                 pattern: str = "noise"):
        if pattern not in SYNTHETIC_PATTERNS:
            raise ValueError(f"pattern must be one of {SYNTHETIC_PATTERNS}, got {pattern!r}")
        self.num_classes, self.n_train, self.n_val, self.n_test = num_classes, n_train, n_val, n_test
        self.num_domains, self.image_size, self.grayscale, self.seed = num_domains, image_size, grayscale, seed
        self.pattern = pattern

    def params(self) -> dict:
        """The constructor arguments: SyntheticAdapter(**params()) builds the same dataset."""
        return {"num_classes": self.num_classes, "n_train": self.n_train, "n_val": self.n_val, "n_test": self.n_test,
                "num_domains": self.num_domains, "image_size": self.image_size, "grayscale": self.grayscale,
                "seed": self.seed, "pattern": self.pattern}

    def build_index(self) -> RawIndex:
        labels, split, domain, ids = [], [], [], []
        for code, n in ((SPLIT_TRAIN, self.n_train), (SPLIT_VAL, self.n_val), (SPLIT_TEST, self.n_test)):
            for c in range(self.num_classes):
                for i in range(n):
                    labels.append(c)
                    split.append(code)
                    domain.append(i % self.num_domains if self.num_domains else -1)
                    ids.append(f"{SPLIT_NAMES[code]}/c{c}/{i}")
        return RawIndex(
            name="Synthetic", labels=np.array(labels), split=np.array(split), domain=np.array(domain), ids=ids,
            class_names=[f"class_{c}" for c in range(self.num_classes)],
            domain_names=[f"domain_{d}" for d in range(self.num_domains)],
            meta={"source": "synthetic", "licence": "none (generated)", "adapter_params": self.params()})

    def preload(self) -> None:
        pass

    def label_and_domain(self, row: int) -> tuple[int, int]:
        """Class and domain of a row, in the order of build_index (split, then class, then sample)."""
        r = int(row)
        for n in (self.n_train, self.n_val, self.n_test):
            size = self.num_classes * n
            if 0 <= r < size:
                c, i = divmod(r, n)
                return c, (i % self.num_domains if self.num_domains else -1)
            r -= size
        raise IndexError(f"row {row} is outside the synthetic index")

    def load_image(self, row: int):
        s = self.image_size
        if self.pattern == "noise":
            rng = np.random.default_rng([self.seed, int(row)])
            shape = (s, s) if self.grayscale else (s, s, 3)
            return to_rgb_image(rng.integers(0, 256, size=shape, dtype=np.uint8))
        c, d = self.label_and_domain(row)
        rng = np.random.default_rng([self.seed, 0x434C53, int(row)])
        img = rng.integers(0, 64, size=(s, s, 3)).astype(np.int32)
        cell = max(s // 4, 1)
        pos = c % 16
        top = (pos // 4) * cell + int(rng.integers(-(cell // 4), cell // 4 + 1))
        left = (pos % 4) * cell + int(rng.integers(-(cell // 4), cell // 4 + 1))
        top, left = min(max(top, 0), s - cell), min(max(left, 0), s - cell)
        colour = np.array([(c * 97) % 160 + 96, (c * 53 + 40) % 160 + 96, (c * 29 + 80) % 160 + 96])
        img[top:top + cell, left:left + cell] += colour
        if d >= 0:
            img[..., d % 3] += 48
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.grayscale:
            img = img.mean(axis=2).round().astype(np.uint8)
        return to_rgb_image(img)


class MedMNISTAdapter:
    """MedMNIST+ 2D datasets through the medmnist package (size=224 by default).

    Rows are train, then val, then test, each in the package's own order.
    Files are stored in dataset/<Name>/rawdata/ (the HtFLlib layout).
    """

    def __init__(self, name: str, rawdata_dir: str, size: int = 224, download: bool = False):
        if name not in MEDMNIST_FLAGS:
            raise ValueError(f"{name} is not a supported MedMNIST dataset: {sorted(MEDMNIST_FLAGS)}")
        import medmnist  # imported here so the rest of the module works without it

        self.name, self.flag, self.size, self.download = name, MEDMNIST_FLAGS[name], size, download
        self.rawdata_dir = rawdata_dir
        self.info = medmnist.INFO[self.flag]
        if "multi-class" not in str(self.info.get("task", "")):
            raise ValueError(f"{name}: only multi-class MedMNIST datasets are supported (task={self.info.get('task')})")
        self._cls = getattr(medmnist, self.info["python_class"])
        self._version = getattr(medmnist, "__version__", "unknown")
        self._images: dict[int, np.ndarray] = {}
        self._offsets: dict[int, int] = {}

    def _open(self, split_name: str):
        os.makedirs(self.rawdata_dir, exist_ok=True)
        ds = self._cls(split=split_name, download=self.download, size=self.size, root=self.rawdata_dir)
        for attr in ("imgs", "labels"):
            if not hasattr(ds, attr):
                raise RuntimeError(f"medmnist {self._version}: dataset object has no attribute {attr!r}; "
                                   f"check the installed version (expected 3.0.x)")
        return ds

    def expected_counts(self) -> dict:
        return {k: int(v) for k, v in self.info["n_samples"].items()}

    def build_index(self) -> RawIndex:
        labels, split, ids = [], [], []
        offset = 0
        for code in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            sname = SPLIT_NAMES[code]
            ds = self._open(sname)
            y = np.asarray(ds.labels).reshape(-1).astype(np.int64)
            labels.append(y)
            split.append(np.full(len(y), code, dtype=np.int8))
            ids.extend(f"{sname}/{i}" for i in range(len(y)))
            self._offsets[code] = offset
            offset += len(y)
            del ds
        n_classes = len(self.info["label"])
        return RawIndex(
            name=self.name, labels=np.concatenate(labels), split=np.concatenate(split),
            domain=np.full(offset, -1, dtype=np.int8), ids=ids,
            class_names=[self.info["label"][str(i)] for i in range(n_classes)],
            meta={"source": "medmnist", "medmnist_version": self._version, "flag": self.flag, "size": self.size,
                  "licence": self.info.get("license", "see medmnist.INFO"),
                  "expected_counts": self.expected_counts()})

    def preload(self) -> None:
        """Load all images into memory once (call before starting DataLoader workers)."""
        offset = 0
        for code in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            if code not in self._images:
                ds = self._open(SPLIT_NAMES[code])
                self._images[code] = np.asarray(ds.imgs)
                del ds
            self._offsets[code] = offset
            offset += len(self._images[code])

    def load_image(self, row: int):
        if len(self._images) < 3:
            self.preload()
        row = int(row)
        for code in (SPLIT_TEST, SPLIT_VAL, SPLIT_TRAIN):
            if row >= self._offsets[code]:
                return to_rgb_image(self._images[code][row - self._offsets[code]])
        raise IndexError(row)


DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FICL-prepare_data"}


def _download(url: str, path: str, retries: int = 5, wait: float = 5.0) -> None:
    """Download url to path through path + ".part", resuming after a cut connection (HTTP Range) and trying
    up to `retries` times. The file appears at `path` only when it is complete (its Content-Length).

    A browser-like User-Agent is sent: some hosts refuse urllib's default one with HTTP 403."""
    import http.client
    import shutil
    import time
    import urllib.error
    import urllib.request

    tmp = path + ".part"
    print(f"downloading {url}", flush=True)
    last = None
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = dict(DOWNLOAD_HEADERS)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                if have and getattr(resp, "status", None) != 206:
                    have = 0                                  # the server ignored the range: start again
                length = resp.headers.get("Content-Length")
                expected = have + int(length) if length is not None else None
                with open(tmp, "ab" if have else "wb") as f:
                    shutil.copyfileobj(resp, f, 1 << 20)
            size = os.path.getsize(tmp)
            if expected is not None and size < expected:
                raise OSError(f"connection closed after {size} of {expected} bytes")
            os.replace(tmp, path)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:                    # the range starts at the end: the file is complete
                os.replace(tmp, path)
                return
            last = exc
            if exc.code in (401, 403, 404):
                break                                         # trying again does not help
        except (OSError, http.client.HTTPException) as exc:
            last = exc
        if attempt < retries:
            print(f"  attempt {attempt} failed ({last}); resuming in {wait:g} s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"could not download {url}: {last}. Download it yourself, e.g.  wget -c -O {path} \"{url}\"  "
                     f"(or with a browser), then run the same prepare_data.py command again: an archive that is "
                     f"already in rawdata/ is not downloaded again")


def download_domainnet(rawdata_dir: str) -> None:
    """Download and unpack DomainNet into rawdata_dir: six domain folders and splits/<domain>_{train,test}.txt.

    Large download. Finished parts are kept, so an interrupted download can simply be started again;
    each zip is unpacked into a temporary folder and moved into place only when complete.
    """
    import shutil
    import zipfile

    os.makedirs(os.path.join(rawdata_dir, "splits"), exist_ok=True)
    for d in DOMAINNET_DOMAINS:
        target = os.path.join(rawdata_dir, d)
        if not os.path.isdir(target):
            zpath = os.path.join(rawdata_dir, f"{d}.zip")
            if not os.path.exists(zpath):
                _download(DOMAINNET_ZIPS[d], zpath)
            tmp_dir = os.path.join(rawdata_dir, f".unpack_{d}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"unpacking {zpath}", flush=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp_dir)
            if not os.path.isdir(os.path.join(tmp_dir, d)):
                raise RuntimeError(f"{zpath} has no top-level {d}/ folder; check the download")
            os.replace(os.path.join(tmp_dir, d), target)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        for s in ("train", "test"):
            path = os.path.join(rawdata_dir, "splits", f"{d}_{s}.txt")
            if not os.path.exists(path):
                _download(DOMAINNET_SPLITS.format(domain=d, split=s), path)


class DomainNetAdapter:
    """DomainNet from rawdata/: the six domain folders and the official split files
    rawdata/splits/<domain>_{train,test}.txt (python prepare_data.py DomainNet --download gets both).

    Rows are ordered by domain (clipart, infograph, painting, quickdraw, real, sketch), then train
    lines, then test lines, in file order.
    """

    def __init__(self, rawdata_dir: str, download: bool = False):
        self.rawdata_dir = rawdata_dir
        self._paths: list[str] = []
        if download:
            download_domainnet(rawdata_dir)

    def build_index(self) -> RawIndex:
        labels, split, domain, ids, names, paths = [], [], [], [], {}, []
        for d, dname in enumerate(DOMAINNET_DOMAINS):
            for code, sname in ((SPLIT_TRAIN, "train"), (SPLIT_TEST, "test")):
                split_file = os.path.join(self.rawdata_dir, "splits", f"{dname}_{sname}.txt")
                if not os.path.exists(split_file):
                    raise FileNotFoundError(f"{split_file} not found: run python prepare_data.py DomainNet "
                                            f"--download, or put the official split files there")
                with open(split_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rel, lab = line.rsplit(" ", 1)
                        lab = int(lab)
                        cname = rel.split("/")[1] if rel.count("/") >= 2 else str(lab)
                        if names.setdefault(lab, cname) != cname:
                            raise ValueError(f"label {lab} maps to two class folders: {names[lab]} and {cname}")
                        labels.append(lab)
                        split.append(code)
                        domain.append(d)
                        ids.append(f"{sname}:{rel}")
                        paths.append(os.path.join(self.rawdata_dir, rel))
        n_classes = max(names) + 1
        missing = [c for c in range(n_classes) if c not in names]
        if missing:
            raise ValueError(f"DomainNet: no samples for labels {missing[:10]}")
        self._paths = paths                  # replaced, never appended: building twice gives the same rows
        return RawIndex(name="DomainNet", labels=np.array(labels), split=np.array(split), domain=np.array(domain),
                        ids=ids, class_names=[names[c] for c in range(n_classes)],
                        domain_names=list(DOMAINNET_DOMAINS),
                        meta={"source": "DomainNet (M3SDA), official split files", "licence": "see the dataset site"})

    def preload(self) -> None:
        if not self._paths:
            self.build_index()

    def load_image(self, row: int):
        from PIL import Image

        if not self._paths:
            self.build_index()
        with Image.open(self._paths[int(row)]) as img:
            return img.convert("RGB")


# ---------------------------------------------------------------------------
# Standard benchmarks read from image files (or the CIFAR pickles) in rawdata/
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 0

# One entry per benchmark. "images" and "classes" are the published totals, checked after the index
# is built. "download": (url, archive name, md5 or None); None means a manual download.
# "split": "official" (the dataset's own train/test lists) or "seeded" (no official split: a seeded,
# stratified train/test split, see seeded_stratified_split; a protocol choice recorded in the index).
BENCHMARKS = {
    "CIFAR100": {
        "classes": 100, "images": 60000, "split": "official", "split_note": "50,000 train / 10,000 test",
        "download": [("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "cifar-100-python.tar.gz",
                      "eb9058c3a382ffc7106e4002c42a8d85")],
        "homepage": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "licence": "no licence stated on the dataset page; cite Krizhevsky (2009)",
        "overlap": "not checked",
    },
    "TinyImageNet": {
        "classes": 200, "images": 110000, "split": "official",
        "split_note": "100,000 train / 10,000 val (the labelled val set is the test set; the unlabelled test set is not used)",
        "download": [("http://cs231n.stanford.edu/tiny-imagenet-200.zip", "tiny-imagenet-200.zip", None)],
        "homepage": "http://cs231n.stanford.edu/",
        "licence": "a subset of ImageNet: ImageNet terms of access (non-commercial research and education)",
        "overlap": "the images come from ImageNet, so ImageNet-pretrained backbones (ViT IN-21K) may have seen them",
    },
    "ImageNetR": {
        "classes": 200, "images": 30000, "split": "seeded",
        "split_note": "no official split; seeded stratified 80/20 (the 24,000 / 6,000 sizes of DualPrompt and "
                      "CODA-Prompt, not their exact images)",
        "download": [("https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar", "imagenet-r.tar",
                      "a61312130a589d0ca1a8fca1f2bd3337")],
        "homepage": "https://github.com/hendrycks/imagenet-r",
        "licence": "no licence stated; renditions collected from the web, research use",
        "overlap": "200 ImageNet-1k classes; the renditions themselves were collected separately",
    },
    "ImageNetA": {
        "classes": 200, "images": 7500, "split": "seeded", "split_note": "no official split; seeded stratified 80/20",
        "download": [("https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar", "imagenet-a.tar",
                      "c3e55429088dc681f30d81f4726b6595")],
        "homepage": "https://github.com/hendrycks/natural-adv-examples",
        "licence": "no licence stated; images collected from the web, research use",
        "overlap": "200 ImageNet-1k classes; the images were collected separately (not from ImageNet)",
    },
    "CUB200": {
        "classes": 200, "images": 11788, "split": "official", "split_note": "train_test_split.txt: 5,994 / 5,794",
        "download": [("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
                      "CUB_200_2011.tgz", None)],
        "homepage": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
        "licence": "non-commercial research and educational purposes only (dataset page)",
        "overlap": "the dataset page warns that its images overlap with ImageNet",
    },
    "EuroSAT": {
        "classes": 10, "images": 27000, "split": "seeded", "split_note": "no official split; seeded stratified 80/20",
        "download": [("https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1", "EuroSAT_RGB.zip", None)],
        "homepage": "https://github.com/phelber/EuroSAT",
        "licence": "MIT; Copernicus Sentinel data terms also apply",
        "overlap": "not checked",
    },
    "RESISC45": {
        "classes": 45, "images": 31500, "split": "seeded", "split_note": "no official split; seeded stratified 80/20",
        "download": None,
        "manual": "download NWPU-RESISC45.rar (links on the homepage and in TensorFlow Datasets' resisc45 page), "
                  "extract it into dataset/RESISC45/rawdata/ (any layout that contains the 45 class folders)",
        "homepage": "https://arxiv.org/abs/1703.00121 (Cheng et al., 2017)",
        "licence": "no licence stated; check the authors' terms",
        "overlap": "not checked",
    },
}


def seeded_stratified_split(labels: np.ndarray, test_fraction: float = DEFAULT_TEST_FRACTION,
                            seed: int = DEFAULT_SPLIT_SEED) -> np.ndarray:
    """Train/test codes for a dataset without an official split.

    Per class: floor(n * test_fraction + 0.5) samples go to test (at least 1 and at most n - 1 when the
    class has 2 or more samples), chosen by a generator seeded with (seed, class). The result depends
    only on the labels in row order, the fraction and the seed.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1), got {test_fraction}")
    labels = np.asarray(labels, dtype=np.int64)
    split = np.full(len(labels), SPLIT_TRAIN, dtype=np.int8)
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        n = len(rows)
        n_test = int(np.floor(n * test_fraction + 0.5))
        n_test = min(max(n_test, 1), n - 1) if n >= 2 else 0
        perm = np.random.default_rng([int(seed), int(c)]).permutation(n)
        split[rows[perm[:n_test]]] = SPLIT_TEST
    return split


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_extract(archive: str, dest: str) -> None:
    """Unpack a .tar(.gz) or .zip archive into dest; refuse members that would land outside dest or are links."""
    import tarfile
    import zipfile

    root = os.path.realpath(dest)

    def inside(name: str) -> bool:
        target = os.path.realpath(os.path.join(root, name))
        return target == root or target.startswith(root + os.sep)

    os.makedirs(dest, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            bad = [m for m in z.namelist() if not inside(m)]
            if bad:
                raise RuntimeError(f"{archive}: refusing members outside the target folder, e.g. {bad[0]!r}")
            z.extractall(dest)
        return
    with tarfile.open(archive) as t:
        members = t.getmembers()
        bad = [m.name for m in members if not inside(m.name) or m.issym() or m.islnk() or m.isdev()]
        if bad:
            raise RuntimeError(f"{archive}: refusing links or members outside the target folder, e.g. {bad[0]!r}")
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, members=members, filter="data")
        else:
            t.extractall(dest, members=members)


def download_benchmark(name: str, rawdata_dir: str) -> dict:
    """Download and unpack the archives of benchmark `name` into rawdata_dir. Returns {archive: md5}.

    An archive already present is not downloaded again, and one already unpacked (marker file
    .unpacked_<archive>) is not unpacked again. A different md5 than the published one is reported.
    """
    spec = BENCHMARKS[name]
    if spec["download"] is None:
        raise SystemExit(f"{name} has no automatic download: {spec['manual']}")
    os.makedirs(rawdata_dir, exist_ok=True)
    md5s = {}
    for url, fname, expected in spec["download"]:
        archive = os.path.join(rawdata_dir, fname)
        marker = os.path.join(rawdata_dir, f".unpacked_{fname}")
        if os.path.exists(marker):
            continue
        if not os.path.exists(archive):
            _download(url, archive)
        md5s[fname] = _md5(archive)
        if expected and md5s[fname] != expected:
            print(f"WARNING: {fname} has md5 {md5s[fname]}, the published value is {expected}. If the download "
                  f"was cut off, delete the file and run again.", flush=True)
        tmp = os.path.join(rawdata_dir, f".unpack_{fname}")
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        print(f"unpacking {archive}", flush=True)
        _safe_extract(archive, tmp)
        for entry in sorted(os.listdir(tmp)):
            target = os.path.join(rawdata_dir, entry)
            if os.path.exists(target):
                raise RuntimeError(f"{target} already exists; move it away or delete it, then run again")
            os.replace(os.path.join(tmp, entry), target)
        shutil.rmtree(tmp, ignore_errors=True)
        with open(marker, "w") as f:
            f.write(md5s[fname] + "\n")
    return md5s


def _is_image(fname: str) -> bool:
    return not fname.startswith(".") and fname.lower().endswith(IMAGE_EXTENSIONS)


def _class_dirs(path: str) -> list[str]:
    """Sub-folders of path that directly contain at least one image."""
    out = []
    for entry in sorted(os.listdir(path)):
        sub = os.path.join(path, entry)
        if not entry.startswith(".") and os.path.isdir(sub) and any(_is_image(f) for f in os.listdir(sub)):
            out.append(entry)
    return out


def find_class_root(rawdata_dir: str, num_classes: int, max_depth: int = 3) -> str:
    """The folder under rawdata_dir (searched up to max_depth levels) whose sub-folders are the num_classes
    class folders. Archives differ in their top-level folder names, so the layout is found, not assumed."""
    level = [rawdata_dir]
    for _ in range(max_depth + 1):
        nxt = []
        for path in level:
            if len(_class_dirs(path)) == num_classes:
                return path
            nxt += [os.path.join(path, e) for e in sorted(os.listdir(path))
                    if not e.startswith(".") and os.path.isdir(os.path.join(path, e))]
        level = nxt
    raise FileNotFoundError(f"no folder with {num_classes} class folders of images under {rawdata_dir}")


def _missing_data_message(name: str, rawdata_dir: str) -> str:
    spec = BENCHMARKS[name]
    how = (f"run  python prepare_data.py {name} --download" if spec["download"] else spec["manual"])
    return f"{name}: the raw files are not in {rawdata_dir}. To get them, {how}."


def _benchmark_meta(name: str, split_rule: dict) -> dict:
    spec = BENCHMARKS[name]
    return {"source": spec["homepage"], "licence": spec["licence"], "pretraining_overlap": spec["overlap"],
            "split": split_rule, "published_counts": {"images": spec["images"], "classes": spec["classes"]}}


def _seeded_rule(test_fraction: float, seed: int) -> dict:
    return {"rule": "seeded stratified train/test split (seeded_stratified_split)", "test_fraction": test_fraction,
            "seed": seed}


class _FileImages:
    """Shared image loading for adapters that keep one file path per row."""

    _paths: list[str]

    def preload(self) -> None:
        if not self._paths:
            self.build_index()

    def load_image(self, row: int):
        from PIL import Image

        if not self._paths:
            self.build_index()
        with Image.open(self._paths[int(row)]) as img:
            return img.convert("RGB")


class ImageFolderAdapter(_FileImages):
    """A benchmark stored as one folder per class (ImageNet-R, ImageNet-A, EuroSAT, RESISC45).

    Classes are the class folder names in sorted order; rows are classes in that order, then the image
    files of each class in sorted order. There is no official split: seeded_stratified_split decides
    train and test, and the rule (fraction, seed) is written into the index.
    """

    def __init__(self, name: str, rawdata_dir: str, download: bool = False,
                 test_fraction: float = DEFAULT_TEST_FRACTION, split_seed: int = DEFAULT_SPLIT_SEED):
        if BENCHMARKS[name]["split"] != "seeded":
            raise ValueError(f"{name} has an official split; ImageFolderAdapter is for datasets without one")
        self.name, self.rawdata_dir = name, rawdata_dir
        self.test_fraction, self.split_seed = float(test_fraction), int(split_seed)
        self._paths: list[str] = []
        self.archive_md5 = download_benchmark(name, rawdata_dir) if download else {}

    def build_index(self) -> RawIndex:
        spec = BENCHMARKS[self.name]
        if not os.path.isdir(self.rawdata_dir):
            raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))
        try:
            root = find_class_root(self.rawdata_dir, spec["classes"])
        except FileNotFoundError:
            raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir)) from None
        classes = _class_dirs(root)
        labels, ids, paths = [], [], []
        for c, cname in enumerate(classes):
            for fname in sorted(f for f in os.listdir(os.path.join(root, cname)) if _is_image(f)):
                labels.append(c)
                ids.append(f"{cname}/{fname}")
                paths.append(os.path.join(root, cname, fname))
        labels = np.array(labels, dtype=np.int64)
        split = seeded_stratified_split(labels, self.test_fraction, self.split_seed)
        self._paths = paths
        meta = _benchmark_meta(self.name, _seeded_rule(self.test_fraction, self.split_seed))
        meta["class_root"] = os.path.relpath(root, self.rawdata_dir)
        return RawIndex(name=self.name, labels=labels, split=split, domain=np.full(len(labels), -1, np.int8),
                        ids=ids, class_names=classes, meta=meta)


class TinyImageNetAdapter(_FileImages):
    """Tiny-ImageNet: train/<wnid>/images/*.JPEG as train, val/images/* with val/val_annotations.txt as test.

    Classes are the wnids of wnids.txt in sorted order. Rows: train (class order, sorted files), then val
    (sorted file names).
    """

    def __init__(self, rawdata_dir: str, download: bool = False):
        self.name, self.rawdata_dir = "TinyImageNet", rawdata_dir
        self._paths: list[str] = []
        self.archive_md5 = download_benchmark(self.name, rawdata_dir) if download else {}

    def _root(self) -> str:
        for dirpath, dirnames, filenames in os.walk(self.rawdata_dir):
            if "wnids.txt" in filenames and {"train", "val"} <= set(dirnames):
                return dirpath
            if dirpath.count(os.sep) - self.rawdata_dir.count(os.sep) >= 2:
                dirnames[:] = []
        raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))

    def build_index(self) -> RawIndex:
        if not os.path.isdir(self.rawdata_dir):
            raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))
        root = self._root()
        with open(os.path.join(root, "wnids.txt")) as f:
            classes = sorted(line.strip() for line in f if line.strip())
        cid = {w: i for i, w in enumerate(classes)}
        labels, split, ids, paths = [], [], [], []
        for w in classes:
            img_dir = os.path.join(root, "train", w, "images")
            for fname in sorted(f for f in os.listdir(img_dir) if _is_image(f)):
                labels.append(cid[w])
                split.append(SPLIT_TRAIN)
                ids.append(f"train/{w}/{fname}")
                paths.append(os.path.join(img_dir, fname))
        with open(os.path.join(root, "val", "val_annotations.txt")) as f:
            rows = sorted(line.rstrip("\n").split("\t")[:2] for line in f if line.strip())
        for fname, w in rows:
            if w not in cid:
                raise ValueError(f"val_annotations.txt names an unknown class {w!r}")
            labels.append(cid[w])
            split.append(SPLIT_TEST)
            ids.append(f"val/{fname}")
            paths.append(os.path.join(root, "val", "images", fname))
        self._paths = paths
        return RawIndex(name=self.name, labels=np.array(labels), split=np.array(split),
                        domain=np.full(len(labels), -1, np.int8), ids=ids, class_names=classes,
                        meta=_benchmark_meta(self.name, {"rule": "official: train / labelled val as test"}))


class CUB200Adapter(_FileImages):
    """CUB-200-2011 with its official split (images.txt, image_class_labels.txt, train_test_split.txt).

    Rows follow the image ids of images.txt; class names come from classes.txt.
    """

    def __init__(self, rawdata_dir: str, download: bool = False):
        self.name, self.rawdata_dir = "CUB200", rawdata_dir
        self._paths: list[str] = []
        self.archive_md5 = download_benchmark(self.name, rawdata_dir) if download else {}

    def _root(self) -> str:
        need = {"images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"}
        for dirpath, dirnames, filenames in os.walk(self.rawdata_dir):
            if need <= set(filenames):
                return dirpath
            if dirpath.count(os.sep) - self.rawdata_dir.count(os.sep) >= 2:
                dirnames[:] = []
        raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))

    @staticmethod
    def _pairs(path: str) -> dict:
        out = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    key, value = line.strip().split(" ", 1)
                    out[int(key)] = value
        return out

    def build_index(self) -> RawIndex:
        if not os.path.isdir(self.rawdata_dir):
            raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))
        root = self._root()
        images = self._pairs(os.path.join(root, "images.txt"))
        label_of = self._pairs(os.path.join(root, "image_class_labels.txt"))
        is_train = self._pairs(os.path.join(root, "train_test_split.txt"))
        classes = self._pairs(os.path.join(root, "classes.txt"))
        n_classes = len(classes)
        if sorted(classes) != list(range(1, n_classes + 1)):
            raise ValueError("classes.txt must number the classes 1..C")
        labels, split, ids, paths = [], [], [], []
        for image_id in sorted(images):
            rel = images[image_id]
            labels.append(int(label_of[image_id]) - 1)
            split.append(SPLIT_TRAIN if is_train[image_id].strip() == "1" else SPLIT_TEST)
            ids.append(rel)
            paths.append(os.path.join(root, "images", rel))
        self._paths = paths
        return RawIndex(name=self.name, labels=np.array(labels), split=np.array(split),
                        domain=np.full(len(labels), -1, np.int8), ids=ids,
                        class_names=[classes[c] for c in range(1, n_classes + 1)],
                        meta=_benchmark_meta(self.name, {"rule": "official: train_test_split.txt"}))


class CIFAR100Adapter:
    """CIFAR-100 from the official python archive (cifar-100-python/{train,test,meta}), 100 fine labels.

    Rows: the 50,000 training images, then the 10,000 test images, in file order. The files are
    Python pickles, read the same way torchvision reads them; use only the archive from the official
    page (its md5 is checked on download).
    """

    def __init__(self, rawdata_dir: str, download: bool = False):
        self.name, self.rawdata_dir = "CIFAR100", rawdata_dir
        self._images: np.ndarray | None = None
        self.archive_md5 = download_benchmark(self.name, rawdata_dir) if download else {}

    def _root(self) -> str:
        for dirpath, dirnames, filenames in os.walk(self.rawdata_dir):
            if {"train", "test", "meta"} <= set(filenames):
                return dirpath
            if dirpath.count(os.sep) - self.rawdata_dir.count(os.sep) >= 2:
                dirnames[:] = []
        raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))

    def _read(self) -> tuple[np.ndarray, np.ndarray, list[str], int]:
        import pickle

        root = self._root()
        with open(os.path.join(root, "meta"), "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        names = [n.decode() for n in meta[b"fine_label_names"]]
        imgs, labels, n_train = [], [], 0
        for part in ("train", "test"):
            with open(os.path.join(root, part), "rb") as f:
                d = pickle.load(f, encoding="bytes")
            x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            imgs.append(x)
            labels.append(np.asarray(d[b"fine_labels"], dtype=np.int64))
            if part == "train":
                n_train = len(x)
        return np.concatenate(imgs), np.concatenate(labels), names, n_train

    def build_index(self) -> RawIndex:
        if not os.path.isdir(self.rawdata_dir):
            raise FileNotFoundError(_missing_data_message(self.name, self.rawdata_dir))
        imgs, labels, names, n_train = self._read()
        n = len(labels)
        split = np.where(np.arange(n) < n_train, SPLIT_TRAIN, SPLIT_TEST).astype(np.int8)
        ids = [f"train/{i}" for i in range(n_train)] + [f"test/{i}" for i in range(n - n_train)]
        return RawIndex(name=self.name, labels=labels, split=split, domain=np.full(n, -1, np.int8), ids=ids,
                        class_names=names, meta=_benchmark_meta(self.name, {"rule": "official: train / test"}))

    def preload(self) -> None:
        if self._images is None:
            self._images = self._read()[0]

    def load_image(self, row: int):
        self.preload()
        return to_rgb_image(self._images[int(row)])


def available_datasets() -> list[str]:
    return sorted(list(MEDMNIST_FLAGS) + list(BENCHMARKS) + ["DomainNet", "Synthetic"])


def has_seeded_split(name: str) -> bool:
    return name in BENCHMARKS and BENCHMARKS[name]["split"] == "seeded"


def get_adapter(name: str, dataset_root: str = ".", download: bool = False, size: int = 224,
                test_fraction: float = DEFAULT_TEST_FRACTION, split_seed: int = DEFAULT_SPLIT_SEED, **kw):
    """Adapter for dataset `name`, reading from dataset_root/<name>/rawdata/."""
    rawdata_dir = os.path.join(dataset_root, name, "rawdata")
    if name in MEDMNIST_FLAGS:
        return MedMNISTAdapter(name, rawdata_dir, size=size, download=download)
    if name == "DomainNet":
        return DomainNetAdapter(rawdata_dir, download=download)
    if name == "CIFAR100":
        return CIFAR100Adapter(rawdata_dir, download=download)
    if name == "TinyImageNet":
        return TinyImageNetAdapter(rawdata_dir, download=download)
    if name == "CUB200":
        return CUB200Adapter(rawdata_dir, download=download)
    if has_seeded_split(name):
        return ImageFolderAdapter(name, rawdata_dir, download=download, test_fraction=test_fraction,
                                  split_seed=split_seed)
    if name == "Synthetic":
        return SyntheticAdapter(**kw)
    raise ValueError(f"unknown dataset {name!r}; available: {available_datasets()}")


def _read_index_json(dataset_root: str, name: str) -> dict:
    path = os.path.join(index_dir(dataset_root, name), "index.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run  python prepare_data.py {name}  in dataset/ first")
    with open(path) as f:
        meta = json.load(f)
    if meta.get("format") != INDEX_FORMAT:
        raise ValueError(f"{path}: unknown index format {meta.get('format')!r}")
    if meta.get("name") != name:
        raise ValueError(f"{path} is the index of {meta.get('name')!r}, not of {name!r}")
    return meta


def adapter_for_index(dataset_root: str, name: str, download: bool = False):
    """The adapter of dataset_root/<name>/cl_index, with the settings its index.json records: the MedMNIST
    image size, the seeded split rule, the synthetic parameters. Use verified_adapter() to also check that
    the raw files still give the same rows."""
    path = os.path.join(index_dir(dataset_root, name), "index.json")
    meta = _read_index_json(dataset_root, name).get("meta", {})
    kw: dict = {}
    if name in MEDMNIST_FLAGS:
        if "size" not in meta:
            raise ValueError(f"{path} does not record the MedMNIST image size")
        kw["size"] = int(meta["size"])
    elif has_seeded_split(name):
        rule = meta.get("split") or {}
        kw["test_fraction"] = float(rule.get("test_fraction", DEFAULT_TEST_FRACTION))
        kw["split_seed"] = int(rule.get("seed", DEFAULT_SPLIT_SEED))
    elif name == "Synthetic":
        params = meta.get("adapter_params")
        if params is None:
            raise ValueError(f"{path} was written before the synthetic image settings were recorded. Run  "
                             f"python make_synthetic.py  in dataset/ once more: it keeps the index (same hash), the "
                             f"caches and the scenarios, and adds the settings")
        kw.update(params)
    return get_adapter(name, dataset_root, download=download, **kw)


def verified_adapter(dataset_root: str, name: str):
    """adapter_for_index(), checked against cl_index, so row i reads the image every scenario and cache means.

    The index is rebuilt from the raw files and its sha256 compared with index.json. MedMNIST is the
    exception: the package's files fix the rows (train, val, test in file order), and rebuilding would load
    every image twice, so the sample counts per split are compared instead."""
    stored = _read_index_json(dataset_root, name)
    adapter = adapter_for_index(dataset_root, name)
    if name in MEDMNIST_FLAGS:
        expected = adapter.expected_counts()
        got = {s: int(stored["counts"][s]["total"]) for s in ("train", "val", "test")}
        if got != {s: int(expected[s]) for s in ("train", "val", "test")}:
            raise ValueError(f"{name}: cl_index has {got} samples per split, the installed medmnist files "
                             f"{expected}; the rows would not match")
        return adapter
    rebuilt = adapter.build_index()
    if rebuilt.hash() != stored["hash"]:
        raise ValueError(
            f"the raw files of {name} no longer give the rows of {index_dir(dataset_root, name)} (sha256 "
            f"{rebuilt.hash()[:12]}, index {stored['hash'][:12]}): images were added, removed or renamed in "
            f"rawdata/. Restore them, or rebuild the index (prepare_data.py --force) and everything built on it")
    return adapter
