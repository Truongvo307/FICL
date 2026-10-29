"""Shared test setup for the dataset side of FICL.

Run from the dataset folder:  cd dataset && python -m pytest tests -q
Only dataset/ goes on the path here. Never add system/ (or the reverse):
dataset/utils is a regular package and would shadow system/utils.
"""
import os
import sys

import pytest

DATASET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and downloaded backbone weights")
    config.addinivalue_line("markers", "data: needs a real dataset prepared under dataset/<Name>/")


@pytest.fixture
def synthetic_index():
    """7 classes, 3 domains, train/val/test rows: small enough to check every row by hand."""
    from utils.raw_utils import SyntheticAdapter
    return SyntheticAdapter(num_classes=7, n_train=20, n_val=4, n_test=6, num_domains=3, seed=0).build_index()
