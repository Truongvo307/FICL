#!/usr/bin/env python
"""
Extract frozen-backbone features into fp16 caches for FICL.

    cd dataset
    python extract_features.py TissueMNIST --backbone dinov2_vitb14 --limit 512   # smoke test first
    python extract_features.py TissueMNIST --backbone dinov2_vitb14
    python extract_features.py TissueMNIST --list

Reads <Name>/cl_index and the raw images (through the adapter with the settings the index records,
checked against the index). For every group in --config (default
configs/extract_default.yaml) writes <Name>/features/<backbone>/<group>/features.npy
(float16 [N, blocks, tokens, 768], cl_index row order) and manifest.json. One forward pass
serves all groups. The forward runs in fp32 with TF32 off. With --limit N only the first N
samples are extracted, into <group>_limitN, so a smoke test never mixes with a full cache.

Every batch also checks that the hooked last-block CLS with the final norm equals the
model's own output; the largest difference goes into the manifest and the run stops before
writing anything final if it exceeds --max_check.
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from utils.backbone_utils import (
    BACKBONES,
    ExtractionPlan,
    GroupSpec,
    RowImages,
    check_torch_numpy,
    hub_weights_sha256,
    iter_extracted_batches,
    load_backbone,
    resolve_blocks,
)
from utils.cache_utils import CacheManifest, create_cache, finalize_cache, group_dir, list_cache_groups
from utils.raw_utils import index_dir, load_raw_index, verified_adapter

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract fp16 feature caches (FICL)")
    p.add_argument("dataset")
    p.add_argument("--backbone", choices=sorted(BACKBONES))
    p.add_argument("--config", default=os.path.join(HERE, "configs", "extract_default.yaml"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device_id", default="0")
    p.add_argument("--dataset_root", default=HERE)
    p.add_argument("--limit", type=int, default=0, help="extract only the first N samples, into <group>_limitN")
    p.add_argument("--model_id", default=None, help="override the timm / open_clip model id")
    p.add_argument("--hub_ref", default=None, help="git ref of facebookresearch/dinov2 for torch.hub")
    p.add_argument("--max_check", type=float, default=1e-3,
                   help="largest allowed |hooked final CLS - model output| before refusing to finalize")
    p.add_argument("--list", action="store_true", help="list finished caches of this dataset")
    return p.parse_args(argv)


def load_plan(path, num_blocks):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    groups = {}
    for name, g in cfg["groups"].items():
        groups[name] = GroupSpec(blocks=resolve_blocks(g["blocks"], num_blocks), tokens=list(g["tokens"]),
                                 final_ln=bool(g["final_ln"]))
    return ExtractionPlan(groups)


def library_versions():
    out = {"python": platform.python_version()}
    for pkg in ("torch", "torchvision", "numpy", "timm", "open_clip_torch", "medmnist", "Pillow", "PyYAML"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance only
        return None


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for bb, g in list_cache_groups(args.dataset_root, args.dataset):
            print(f"  {bb}/{g}")
        return 0
    if not args.backbone:
        print("give --backbone, or --list")
        return 1
    try:
        check_torch_numpy()          # before any download: every batch crosses torch -> numpy
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA is not available; extraction on the CPU will be very slow")

    idx = load_raw_index(index_dir(args.dataset_root, args.dataset))
    # the adapter with the settings recorded in the index (e.g. the MedMNIST image size), checked against it
    raw = verified_adapter(args.dataset_root, args.dataset)
    model, preprocess, adapter, info = load_backbone(args.backbone, device, model_id=args.model_id,
                                                     hub_ref=args.hub_ref)
    if BACKBONES[args.backbone]["source"] == "torchhub":
        info["hub_checkpoints"] = hub_weights_sha256(BACKBONES[args.backbone]["entry"])
    plan = load_plan(args.config, adapter.num_blocks)

    n = idx.num_samples if args.limit <= 0 else min(args.limit, idx.num_samples)
    suffix = f"_limit{n}" if args.limit > 0 else ""
    rows = np.arange(n, dtype=np.int64)
    paths = {g: group_dir(args.dataset_root, args.dataset, args.backbone, g + suffix) for g in plan.groups}
    arrays = {g: create_cache(paths[g], (n, len(s.blocks), len(s.tokens), adapter.embed_dim))
              for g, s in plan.groups.items()}

    print(f"{args.dataset}: {n} samples, backbone {args.backbone} on {device}, groups "
          + ", ".join(f"{g}{suffix} {list(arrays[g].shape)}" for g in plan.groups))
    raw.preload()       # load images once, before DataLoader workers start
    loader = DataLoader(RowImages(raw, preprocess, rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"))

    t0, done, max_check, next_report = time.time(), 0, 0.0, 0
    for rows_b, out, check in iter_extracted_batches(loader, adapter, plan, device):
        start, end = int(rows_b[0]), int(rows_b[-1]) + 1
        if start != done or end - start != len(rows_b):
            raise RuntimeError(f"rows arrived out of order: expected {done}, got {start}..{end - 1}")
        for g in plan.groups:
            arrays[g][start:end] = out[g]
        done = end
        max_check = max(max_check, check)
        if done >= next_report or done == n:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{n} images  {rate:.0f} img/s  reference check {max_check:.2e}", flush=True)
            next_report = done + max(n // 20, 1)
    seconds = time.time() - t0
    for g in list(arrays):
        arrays[g].flush()
    del arrays

    if max_check > args.max_check:
        raise SystemExit(f"reference check {max_check:.3e} > {args.max_check}: the hooked features do not match "
                         f"the model output. Nothing was finalized (the .partial files stay for inspection).")

    for g, s in plan.groups.items():
        manifest = CacheManifest(
            dataset=args.dataset, backbone=args.backbone, group=g + suffix, blocks=s.blocks, tokens=s.tokens,
            final_ln=s.final_ln, embed_dim=adapter.embed_dim, num_samples=n,
            shape=[n, len(s.blocks), len(s.tokens), adapter.embed_dim], raw_index_hash=idx.hash(),
            preprocess=info.get("preprocess", ""),
            weights={k: v for k, v in info.items() if k != "preprocess"},
            forward_precision="fp32", tf32=False, batch_size=args.batch_size, num_workers=args.num_workers,
            versions=library_versions(), seconds=round(seconds, 2), images_per_s=round(n / max(seconds, 1e-9), 2),
            reference_check_max_abs_diff=max_check, git_commit=git_commit(),
            notes=(f"smoke test, first {n} samples; " if args.limit > 0 else "") + f"config {args.config}")
        finalize_cache(paths[g], manifest)
        size_gb = os.path.getsize(os.path.join(paths[g], "features.npy")) / 1e9
        print(f"{paths[g]}: {size_gb:.2f} GB, sha256 {manifest.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
