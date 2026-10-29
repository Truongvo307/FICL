#!/usr/bin/env python
"""
Build the raw sample index of one dataset for FICL.

    cd dataset
    python prepare_data.py TissueMNIST --download   # MedMNIST+ at 224 px into TissueMNIST/rawdata/
    python prepare_data.py DomainNet --download      # six domains + official split files into DomainNet/rawdata/ (large)
    python prepare_data.py ImageNetR --download      # a standard benchmark (python prepare_data.py --info lists them)
    python prepare_data.py --info                    # every supported dataset: size, split, source, licence
    python prepare_data.py --list                    # datasets that already have an index

Writes <Name>/cl_index/ (labels, official split, domain, sample ids, sha256). Every feature
cache and every scenario refers to samples by their row in this index, so build it once and
do not rebuild it unless you also rebuild the caches and scenarios.

Datasets without an official split (ImageNetR, ImageNetA, EuroSAT, RESISC45) get a seeded,
stratified train/test split (default 80/20, seed 0). The rule is written into the index and is
part of its hash; --test_fraction and --split_seed change it (refused for datasets that have an
official split).
"""
import argparse
import os
import sys

from utils.raw_utils import (
    BENCHMARKS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    MEDMNIST_FLAGS,
    SPLIT_NAMES,
    available_datasets,
    get_adapter,
    has_seeded_split,
    index_dir,
    load_raw_index,
    save_raw_index,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build dataset/<Name>/cl_index for FICL")
    p.add_argument("dataset", nargs="?", choices=available_datasets())
    p.add_argument("--download", action="store_true",
                   help="download missing files (every dataset except RESISC45, which is a manual download)")
    p.add_argument("--size", type=int, default=224, help="MedMNIST image size (64, 128 or 224)")
    p.add_argument("--dataset_root", default=HERE, help="folder that holds the <Name>/ dataset folders")
    p.add_argument("--force", action="store_true",
                   help="replace an existing index that has a different hash (invalidates its caches and scenarios)")
    p.add_argument("--test_fraction", type=float, default=None,
                   help=f"datasets without an official split: share of each class used as test "
                        f"(default {DEFAULT_TEST_FRACTION})")
    p.add_argument("--split_seed", type=int, default=None,
                   help=f"datasets without an official split: seed of the split (default {DEFAULT_SPLIT_SEED})")
    p.add_argument("--list", action="store_true", help="list datasets that already have an index")
    p.add_argument("--info", action="store_true", help="list every supported dataset: size, split, source, licence")
    return p.parse_args(argv)


def print_info():
    print(f"{'name':13s} {'classes':>7s} {'images':>8s}  {'split':9s} download")
    for name in MEDMNIST_FLAGS:
        print(f"{name:13s} {'':>7s} {'':>8s}  {'official':9s} --download (medmnist package)")
    print(f"{'DomainNet':13s} {345:7d} {'':>8s}  {'official':9s} --download (six domains, large)")
    for name, b in BENCHMARKS.items():
        how = "--download" if b["download"] else "manual"
        print(f"{name:13s} {b['classes']:7d} {b['images']:8d}  {b['split']:9s} {how}")
    print("\nPer benchmark: split, licence and overlap with the backbones' pretraining data")
    for name, b in BENCHMARKS.items():
        print(f"  {name}: {b['split_note']}\n      licence: {b['licence']}\n      overlap: {b['overlap']}")
        if not b["download"]:
            print(f"      manual download: {b['manual']}")


def print_summary(idx, h):
    counts = idx.counts()
    print(f"\n{'=' * 60}\n{idx.name}: {idx.num_samples} samples, {idx.num_classes} classes"
          + (f", {idx.num_domains} domains" if idx.num_domains else ""))
    for sname in SPLIT_NAMES.values():
        per = counts[sname]["per_class"]
        print(f"  {sname:5s}: {counts[sname]['total']:8d} samples | per class min {min(per)} max {max(per)}"
              + (f" | per domain {counts[sname]['per_domain']}" if idx.num_domains else ""))
    print(f"  hash : {h}\n{'=' * 60}")


def list_indexes(root):
    found = False
    for name in available_datasets():
        path = index_dir(root, name)
        if os.path.exists(os.path.join(path, "index.json")):
            idx = load_raw_index(path, verify=False)
            print(f"  {name:12s} {idx.num_samples:8d} samples  {idx.num_classes:4d} classes  hash {idx.hash()[:12]}")
            found = True
    if not found:
        print("  (no cl_index yet)")


def main(argv=None):
    args = parse_args(argv)
    if args.info:
        print_info()
        return 0
    if args.list:
        list_indexes(args.dataset_root)
        return 0
    if not args.dataset:
        print("give a dataset name, --info or --list")
        return 1
    split_args = {"test_fraction": args.test_fraction, "split_seed": args.split_seed}
    given = [f"--{k}" for k, v in split_args.items() if v is not None]
    if given and not has_seeded_split(args.dataset):
        raise SystemExit(f"{' and '.join(given)}: {args.dataset} has an official split, so there is no split to "
                         f"choose; remove the option")
    if args.test_fraction is None:
        split_args["test_fraction"] = DEFAULT_TEST_FRACTION
    if args.split_seed is None:
        split_args["split_seed"] = DEFAULT_SPLIT_SEED

    out = index_dir(args.dataset_root, args.dataset)
    try:
        adapter = get_adapter(args.dataset, args.dataset_root, download=args.download, size=args.size,
                              **(split_args if has_seeded_split(args.dataset) else {}))
        idx = adapter.build_index()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    if args.dataset in BENCHMARKS:
        b = BENCHMARKS[args.dataset]
        if idx.num_samples != b["images"] or idx.num_classes != b["classes"]:
            print(f"WARNING: {args.dataset} has {idx.num_samples} images in {idx.num_classes} classes; the published "
                  f"totals are {b['images']} in {b['classes']}. Check the raw files before using this index.")
        print(f"licence: {b['licence']}\noverlap with pretraining data: {b['overlap']}")

    if args.dataset in MEDMNIST_FLAGS:
        expected = adapter.expected_counts()
        got = idx.counts()
        for sname in ("train", "val", "test"):
            if got[sname]["total"] != expected[sname]:
                raise SystemExit(f"{args.dataset} {sname}: {got[sname]['total']} samples, but medmnist.INFO "
                                 f"says {expected[sname]}; stopping")

    if os.path.exists(os.path.join(out, "index.json")):
        old = load_raw_index(out, verify=False)
        if old.hash() == idx.hash():
            print(f"{out} already holds this exact index; nothing to do.")
            print_summary(idx, idx.hash())
            return 0
        if not args.force:
            raise SystemExit(f"{out} holds a different index (hash {old.hash()[:12]}, new {idx.hash()[:12]}). "
                             f"Caches and scenarios built on it would no longer match; use --force only if "
                             f"you will rebuild them.")

    h = save_raw_index(idx, out)
    print(f"Saved {out}")
    print_summary(idx, h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
