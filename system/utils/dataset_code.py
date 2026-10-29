"""Dataset-side modules that the system side runs, loaded by file path under private names.

system/ and dataset/ both have a package called utils, so dataset/ is never put on sys.path (its
utils would shadow this one). The two modules loaded here are standalone by design (no utils.* imports):
    raw_utils.py       the dataset adapters that built cl_index (numpy and Pillow; medmnist for MedMNIST)
    backbone_utils.py  the frozen backbones of feature extraction (torch; timm, open_clip, torch.hub lazily)
Image algorithms read raw images through the same adapters that built the index, and pretrained
networks are loaded by the same code that writes the feature caches. The trained and the frozen
families therefore see the same rows, the same images and the same evaluation preprocessing.
"""
from __future__ import annotations

import importlib.util
import os
import sys

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CODE_DIR = os.path.join(os.path.dirname(SYSTEM_DIR), "dataset", "utils")


def load(stem: str):
    """dataset/utils/<stem>.py as module ficl_dataset_<stem> (imported once per process)."""
    name = f"ficl_dataset_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(DATASET_CODE_DIR, f"{stem}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: system/ needs the dataset/ folder of this repository next to it")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod             # registered before running it (dataclasses look the module up)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod
