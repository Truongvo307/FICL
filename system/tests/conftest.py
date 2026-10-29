"""Shared test setup for the system side of FICL.

Run from the system folder:  cd system && python -m pytest tests -q
Only system/ goes on the path. Dataset-side modules that are needed to build
fixtures are loaded by file path, so dataset/utils never shadows system/utils.
"""
import importlib.util
import os
import sys

# deterministic cuBLAS for the GPU tests of the image path; must be set before CUDA starts (main.py does the same)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(SYSTEM_DIR), "dataset")
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU")
    config.addinivalue_line("markers", "data: needs a real dataset prepared under dataset/<Name>/")


def load_dataset_module(filename, name):
    """Import dataset/utils/<filename> under a private name, without touching sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(DATASET_DIR, "utils", filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_dataset_root(tmp_path):
    """dataset_root/Synthetic with a raw index, two scenarios and a fake fp16 cache group 'pre'."""
    raw = load_dataset_module("raw_utils.py", "_ds_raw_utils")
    scen = load_dataset_module("scenario_utils.py", "_ds_scenario_utils")
    cache = load_dataset_module("cache_utils.py", "_ds_cache_utils")

    root = str(tmp_path)
    index = raw.SyntheticAdapter(num_classes=7, n_train=20, n_val=4, n_test=6, num_domains=3, seed=0).build_index()
    raw.save_raw_index(index, os.path.join(root, "Synthetic", "cl_index"))
    dir_spec = scen.ScenarioSpec(dataset="Synthetic", partition="dir", num_clients=5, alpha=0.3,
                                 class_per_task=2, seed=0)
    nat_spec = scen.ScenarioSpec(dataset="Synthetic", partition="natural", class_per_task=2, seed=0)
    scen.save_scenario(scen.build_scenario(index, dir_spec), os.path.join(root, "Synthetic"))
    scen.save_scenario(scen.build_scenario(index, nat_spec), os.path.join(root, "Synthetic"))

    shape = (len(index.labels), 4, 2, 8)
    group = os.path.join(root, "Synthetic", "features", "fake", "pre")
    arr = cache.create_cache(group, shape)
    arr[:] = np.random.default_rng(0).standard_normal(shape).astype(np.float16)
    arr.flush()
    del arr
    cache.finalize_cache(group, cache.CacheManifest(
        dataset="Synthetic", backbone="fake", group="pre", blocks=[0, 1, 2, 3], tokens=["cls", "mean"],
        final_ln=False, embed_dim=8, num_samples=shape[0], shape=list(shape), raw_index_hash=index.hash()))
    return {"root": root, "index": index, "cache_file": os.path.join(group, cache.FEATURE_FILE),
            "raw": raw, "scen": scen, "cache": cache}


@pytest.fixture
def synthetic_root(tmp_path):
    """dataset/Synthetic exactly as dataset/make_synthetic.py writes it, in a temporary folder.

    8 classes, 3 domains, fp16 features [N, 4 blocks, 2 tokens, 16] for backbone "synthetic",
    group "pre"; scenarios cl_5clients_dir_0.5_cpt2_seed{0,1,2} and cl_natural_cpt2_seed{0,1,2}.
    """
    raw = load_dataset_module("raw_utils.py", "_ds_raw_utils")
    scen = load_dataset_module("scenario_utils.py", "_ds_scenario_utils")
    cache = load_dataset_module("cache_utils.py", "_ds_cache_utils")
    synth = load_dataset_module("synthetic_utils.py", "_ds_synthetic_utils")
    root = str(tmp_path / "dataset")
    info = synth.write_synthetic_dataset(root, raw, scen, cache)
    return {"root": root, "results": str(tmp_path / "results"), **info}
