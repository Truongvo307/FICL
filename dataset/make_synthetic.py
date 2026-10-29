#!/usr/bin/env python
"""
Write the synthetic dataset used by the FICL smoke runs (CPU, a few seconds).

    cd dataset
    python make_synthetic.py

Creates Synthetic/cl_index, Synthetic/features/synthetic/pre (class-structured fp16 features) and
the scenarios cl_5clients_dir_0.5_cpt2_seed{0,1,2} and cl_natural_cpt2_seed{0,1,2}. Running it
again keeps identical files and refuses to overwrite different ones. Numbers produced on this
dataset only show that the pipeline works; they are not results.
"""
import argparse
import os
import sys

from utils import cache_utils, raw_utils, scenario_utils
from utils.synthetic_utils import write_synthetic_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser(description="Write dataset/Synthetic for FICL smoke runs")
    p.add_argument("--dataset_root", default=HERE)
    args = p.parse_args(argv)
    info = write_synthetic_dataset(args.dataset_root, raw_utils, scenario_utils, cache_utils)
    idx = info["index"]
    print(f"Synthetic dataset in {os.path.join(args.dataset_root, idx.name)}: {idx.num_samples} samples, "
          f"{idx.num_classes} classes, {idx.num_domains} domains")
    print(f"  features: {info['cache_dir']}")
    for name in info["scenarios"]:
        print(f"  scenario: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
