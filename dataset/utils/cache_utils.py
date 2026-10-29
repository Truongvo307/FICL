"""fp16 feature caches for FICL.

One extraction run writes one or more "groups" for a (dataset, backbone) pair:

    dataset/<Name>/features/<backbone>/<group>/features.npy    float16 [N, L, T, D]
                                               manifest.json   how it was made, sha256

N = samples in cl_index order, L = recorded blocks, T = token variants (cls, mean),
D = embedding size. The file is written as features.npy.partial and renamed only when
finalize_cache has checked its shape and stored its sha256, so an interrupted run never
leaves a file that looks finished. Readers open it read-only with mmap_mode="r".

This module must stay standalone (numpy only): the system-side tests load it by path.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone

import numpy as np

FEATURE_FILE = "features.npy"
MANIFEST_FILE = "manifest.json"
PARTIAL_SUFFIX = ".partial"
CACHE_FORMAT = "fcl-feature-cache/1"


@dataclass
class CacheManifest:
    dataset: str
    backbone: str
    group: str
    blocks: list[int]                  # block ids (0-based) along axis 1
    tokens: list[str]                  # token variants along axis 2, e.g. ["cls", "mean"]
    final_ln: bool                     # True: the backbone's final norm was applied per token before pooling
    embed_dim: int
    num_samples: int
    shape: list[int]
    raw_index_hash: str
    dtype: str = "float16"
    sha256: str = ""
    preprocess: str = ""
    weights: dict = field(default_factory=dict)
    forward_precision: str = "fp32"
    tf32: bool = False
    batch_size: int = 0
    num_workers: int = 0
    versions: dict = field(default_factory=dict)
    seconds: float = 0.0
    images_per_s: float = 0.0
    reference_check_max_abs_diff: float | None = None
    created: str = ""
    git_commit: str | None = None
    notes: str = ""
    format: str = CACHE_FORMAT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheManifest":
        if d.get("format") != CACHE_FORMAT:
            raise ValueError(f"unknown cache format {d.get('format')!r}")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def group_dir(dataset_root: str, dataset: str, backbone: str, group: str) -> str:
    return os.path.join(dataset_root, dataset, "features", backbone, group)


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def create_cache(group_path: str, shape: tuple[int, ...]) -> np.memmap:
    """Create features.npy.partial (float16, zero-filled) and return it as a writable memmap."""
    os.makedirs(group_path, exist_ok=True)
    final = os.path.join(group_path, FEATURE_FILE)
    if os.path.exists(final):
        raise FileExistsError(f"{final} already exists; delete the group folder to rebuild it")
    return np.lib.format.open_memmap(final + PARTIAL_SUFFIX, mode="w+", dtype=np.float16,
                                     shape=tuple(int(s) for s in shape))


def _write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def finalize_cache(group_path: str, manifest: CacheManifest) -> CacheManifest:
    """Check the partial file against the manifest, store its sha256, rename it, write manifest.json.

    Flush and drop your memmap (del arr) before calling this.
    """
    final = os.path.join(group_path, FEATURE_FILE)
    partial = final + PARTIAL_SUFFIX
    if not os.path.exists(partial):
        raise FileNotFoundError(f"{partial} not found (nothing to finalize)")
    arr = np.load(partial, mmap_mode="r", allow_pickle=False)
    shape, dtype = list(arr.shape), arr.dtype
    del arr
    if dtype != np.float16:
        raise ValueError(f"{partial}: dtype {dtype}, expected float16")
    if shape != list(manifest.shape):
        raise ValueError(f"{partial}: shape {shape} does not match manifest {manifest.shape}")
    if len(shape) != 4 or shape[0] != manifest.num_samples or shape[1] != len(manifest.blocks) \
            or shape[2] != len(manifest.tokens) or shape[3] != manifest.embed_dim:
        raise ValueError(f"{partial}: shape {shape} does not match num_samples/blocks/tokens/embed_dim")
    manifest.sha256 = sha256_file(partial)
    if not manifest.created:
        manifest.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.replace(partial, final)
    _write_json(os.path.join(group_path, MANIFEST_FILE), manifest.to_dict())
    return manifest


def open_cache(group_path: str, expected_raw_index_hash: str | None = None,
               verify_checksum: bool = False) -> tuple[np.ndarray, CacheManifest]:
    """Open a finished group read-only. Raises if it was built for another raw index."""
    final = os.path.join(group_path, FEATURE_FILE)
    man_path = os.path.join(group_path, MANIFEST_FILE)
    if not (os.path.exists(final) and os.path.exists(man_path)):
        hint = " (extraction did not finish)" if os.path.exists(final + PARTIAL_SUFFIX) else ""
        raise FileNotFoundError(f"no finished cache in {group_path}{hint}")
    with open(man_path) as f:
        manifest = CacheManifest.from_dict(json.load(f))
    if expected_raw_index_hash is not None and manifest.raw_index_hash != expected_raw_index_hash:
        raise ValueError(f"{group_path} was built for another raw index "
                         f"({manifest.raw_index_hash[:12]} != {expected_raw_index_hash[:12]})")
    if verify_checksum and sha256_file(final) != manifest.sha256:
        raise ValueError(f"{final}: sha256 does not match manifest.json (edited or corrupted)")
    arr = np.load(final, mmap_mode="r", allow_pickle=False)
    if arr.dtype != np.float16 or list(arr.shape) != list(manifest.shape):
        raise ValueError(f"{final}: dtype/shape {arr.dtype}/{list(arr.shape)} do not match manifest.json")
    return arr, manifest


def list_cache_groups(dataset_root: str, dataset: str) -> list[tuple[str, str]]:
    """[(backbone, group)] with a finished cache."""
    base = os.path.join(dataset_root, dataset, "features")
    out = []
    if os.path.isdir(base):
        for bb in sorted(os.listdir(base)):
            bdir = os.path.join(base, bb)
            if not os.path.isdir(bdir):
                continue
            for g in sorted(os.listdir(bdir)):
                if os.path.exists(os.path.join(bdir, g, MANIFEST_FILE)):
                    out.append((bb, g))
    return out
