"""Synthetic dataset for FICL smoke runs and tests (numpy only, standalone).

write_synthetic_dataset() writes <dataset_root>/Synthetic/ with the same files as the real
pipeline, so the whole system side (main.py, the loop, the result files) can be run and
tested in seconds on a CPU:

    cl_index/                       SyntheticAdapter index: 8 classes, 3 domains, train/test splits; its
                                    images (load_image) are class-structured 32 x 32 squares for the
                                    image path (pattern "class"), and their settings are in index.json
    features/synthetic/pre/         fp16 [N, 4 blocks, 2 tokens (cls, mean), 16]: class means plus
                                    noise; later blocks separate the classes better (nearest class
                                    mean on 8 classes: about 0.29 at block 0, 0.92 at block 3) and
                                    every domain shifts the features a little
    cl_5clients_dir_0.5_cpt2_seed{0,1,2}/   Dirichlet scenarios, 4 tasks of 2 classes
    cl_natural_cpt2_seed{0,1,2}/            one client per domain

The three data modules (raw_utils, scenario_utils, cache_utils) are passed in as
arguments, so this file works from dataset/ (normal imports, make_synthetic.py) and from
system/tests (modules loaded by file path). Results on these features are smoke tests only.
"""
from __future__ import annotations

import os

import numpy as np

TOKENS = ("cls", "mean")
IMAGE_SIZE = 32         # the class-structured images of SyntheticAdapter(pattern="class")
SYNTHETIC_DEFAULTS = {
    "num_classes": 8,
    "n_train": 60,
    "n_val": 0,
    "n_test": 20,
    "num_domains": 3,
    "num_blocks": 4,
    "dim": 16,
    "class_sep": 1.0,
    "domain_shift": 1.0,
    "seed": 0,
}


def synthetic_features(labels: np.ndarray, domain: np.ndarray, num_classes: int, num_domains: int,
                       num_blocks: int = 4, dim: int = 16, class_sep: float = 1.0, domain_shift: float = 1.0,
                       seed: int = 0) -> np.ndarray:
    """float16 [N, num_blocks, 2, dim]. Block b separates classes with scale class_sep * (b + 1) / num_blocks."""
    labels = np.asarray(labels, dtype=np.int64)
    domain = np.asarray(domain, dtype=np.int64)
    n = len(labels)
    rng = np.random.default_rng([int(seed), 0x53594E])
    out = np.empty((n, num_blocks, len(TOKENS), dim), dtype=np.float16)
    for b in range(num_blocks):
        means = rng.standard_normal((num_classes, dim)) * (class_sep * (b + 1) / num_blocks)
        base = means[labels]
        if num_domains > 0:
            shifts = rng.standard_normal((num_domains, dim)) * domain_shift
            base = base + shifts[np.clip(domain, 0, None)]
        for k in range(len(TOKENS)):
            noise = rng.standard_normal((n, dim)) * (1.0 + 0.5 * k)
            out[:, b, k] = (base + noise).astype(np.float16)
    return out


def default_scenario_specs(scen, dataset: str = "Synthetic") -> list:
    specs = []
    for seed in (0, 1, 2):
        specs.append(scen.ScenarioSpec(dataset=dataset, partition="dir", num_clients=5, alpha=0.5,
                                       class_per_task=2, seed=seed))
        specs.append(scen.ScenarioSpec(dataset=dataset, partition="natural", class_per_task=2, seed=seed))
    return specs


def write_synthetic_dataset(dataset_root: str, raw, scen, cache, backbone: str = "synthetic", group: str = "pre",
                            scenario_specs=None, **params) -> dict:
    """Write (or check) the synthetic dataset. Identical existing files are kept; different ones are refused."""
    p = dict(SYNTHETIC_DEFAULTS)
    unknown = sorted(set(params) - set(p))
    if unknown:
        raise ValueError(f"unknown synthetic parameters {unknown}; known: {sorted(p)}")
    p.update(params)

    index = raw.SyntheticAdapter(num_classes=p["num_classes"], n_train=p["n_train"], n_val=p["n_val"],
                                 n_test=p["n_test"], num_domains=p["num_domains"], seed=p["seed"],
                                 image_size=IMAGE_SIZE, pattern="class").build_index()
    idx_dir = raw.index_dir(dataset_root, index.name)
    if os.path.exists(os.path.join(idx_dir, "index.json")):
        old = raw.load_raw_index(idx_dir)
        if old.hash() != index.hash():
            raise FileExistsError(f"{idx_dir} holds a different index; delete {os.path.dirname(idx_dir)} first")
        if old.meta != index.meta:
            # same rows (same sha256): only the metadata changes, e.g. an index written before the image
            # settings were recorded. Caches and scenarios refer to the hash, so they stay valid.
            raw.save_raw_index(index, idx_dir)
    else:
        raw.save_raw_index(index, idx_dir)

    gdir = cache.group_dir(dataset_root, index.name, backbone, group)
    if os.path.exists(os.path.join(gdir, cache.MANIFEST_FILE)):
        _feats, manifest = cache.open_cache(gdir, expected_raw_index_hash=index.hash())
        if manifest.weights.get("synthetic_params") != p:
            raise FileExistsError(f"{gdir} was written with other synthetic parameters; delete it first")
    else:
        feats = synthetic_features(index.labels, index.domain, index.num_classes, index.num_domains,
                                   p["num_blocks"], p["dim"], p["class_sep"], p["domain_shift"], p["seed"])
        arr = cache.create_cache(gdir, feats.shape)
        arr[:] = feats
        arr.flush()
        del arr
        cache.finalize_cache(gdir, cache.CacheManifest(
            dataset=index.name, backbone=backbone, group=group, blocks=list(range(p["num_blocks"])),
            tokens=list(TOKENS), final_ln=False, embed_dim=p["dim"], num_samples=index.num_samples,
            shape=list(feats.shape), raw_index_hash=index.hash(), weights={"synthetic_params": p},
            notes="synthetic features for FICL smoke runs and tests; not a real backbone"))

    dataset_dir = os.path.join(dataset_root, index.name)
    specs = default_scenario_specs(scen, index.name) if scenario_specs is None else scenario_specs
    names = [os.path.basename(scen.save_scenario(scen.build_scenario(index, s), dataset_dir)) for s in specs]
    return {"index": index, "cache_dir": gdir, "cache_file": os.path.join(gdir, cache.FEATURE_FILE),
            "scenarios": names, "params": p}
