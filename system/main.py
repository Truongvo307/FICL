#!/usr/bin/env python
"""
Entry point of FICL: federated class-incremental learning on cached features or on images.

    cd system
    python main.py -cfg configs/smoke_single.yaml --dry_run    # check settings and data, run nothing
    python main.py -cfg configs/smoke_single.yaml               # closed-form: one exchange per task
    python main.py -cfg configs/smoke_multi.yaml -rpt 30         # a flag overrides the YAML value
    python main.py -cfg configs/smoke_multi_images.yaml          # a network trained on images (CPU, tiny CNN)
    python main.py --list_algorithms

Every invocation prints the protocol first (algorithm and regime, tasks, rounds or no rounds,
the scenario folder of every seed), then runs seeds seed .. seed+times-1 into
    ../results/<dataset>/<scenario>/<algorithm>[_FLAGS]_<backbone>-<group>_<goal>_<timestamp>/times_<i>/
(image algorithms: <algorithm>[_FLAGS]_<network>-<trainable>_<goal>_<timestamp>)
averages the complete runs into avg/, and checks the files (check_run.py).
Exit codes: 0 all runs complete and all checks pass, 1 a run or a check failed, 2 the settings or
the data cannot be used (nothing is written in that case).
Settings and hyperparameters
    python main.py --help                     every general setting (defined in utils/config_utils.py, OPTIONS)
    python main.py -algo NAME --help          the same plus the hyperparameters of algorithm NAME (--NAME VALUE)
    python main.py --list_settings            the general settings as a table: YAML key, flag, default, meaning
    python main.py --list_algorithms          every algorithm with its regime and hyperparameters
An algorithm declares its hyperparameters in HPARAMS in its server class; see flcore/servers/serverbase.py.

To add an algorithm: write its server and client classes and register it in ALGORITHMS below.
"""
from __future__ import annotations

import importlib
import os
from importlib import metadata
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np

from check_run import check_experiment
from utils.comm_utils import WIRE_RULE
from utils.config_utils import (
    FEATURES,
    REGIME_TEXT,
    ConfigError,
    describe_protocol,
    finalize_config,
    find_algorithm,
    merge_sources,
    model_tag,
    parse_cli,
    read_yaml,
    settings_record,
    settings_table,
)
from utils.data_utils import DataConfig, FeatureStore, FeatureView, check_compatible, load_scenario, set_data_config
from utils.result_utils import compute_averages, write_json

# name -> "module:Class". Modules are imported only when used, after CUDA_VISIBLE_DEVICES is set.
ALGORITHMS = {
    "SmokeClassMean": "flcore.servers.serverclassmean:SmokeClassMean",          # single-round smoke test
    "SmokeFedAvgLinear": "flcore.servers.serverfedavglin:SmokeFedAvgLinear",    # multi-round smoke test
    "FedAvg": "flcore.servers.serverfedavg:FedAvg",                              # images: sequential fine-tuning
    "FedAvgJoint": "flcore.servers.serverfedavg:FedAvgJoint",                    # images: joint upper bound
}


def get_server_class(name: str):
    if name not in ALGORITHMS:
        raise ConfigError(f"unknown algorithm {name!r}; registered: {sorted(ALGORITHMS)}")
    module, cls_name = ALGORITHMS[name].split(":")
    server_cls = getattr(importlib.import_module(module), cls_name)
    server_cls.check_contract()
    return server_cls


def algorithms_table() -> list[str]:
    """Every registered algorithm: regime, flags, the first line of its description, its hyperparameters."""
    lines = []
    for name in sorted(ALGORITHMS):
        cls = get_server_class(name)
        flags = f"   flags: {' '.join(cls.RUN_FLAGS)}" if cls.RUN_FLAGS else ""
        reads = "cached features" if cls.INPUT == FEATURES else "images (model section)"
        lines.append(f"{name}   [{REGIME_TEXT[cls.REGIME]}, reads {reads}]{flags}")
        doc = (cls.__dict__.get("__doc__") or sys.modules[cls.__module__].__doc__ or "").strip().splitlines()
        if doc:
            first = doc[0][len(name) + 1:].strip() if doc[0].startswith(name + ":") else doc[0]
            lines.append(f"    {first}")
        if cls.HPARAMS:
            for hp_name, spec in cls.HPARAMS.items():
                lines.append(f"    {'--' + hp_name + ' ' + spec.kind.upper():26s} {spec.describe()}")
        else:
            lines.append("    (no hyperparameters)")
    return lines


def resolve_args(argv):
    """Defaults < YAML (-cfg) < flags, checked against the algorithm. Returns (args, server class, control).

    The algorithm is looked up first (from -algo or the YAML file): its hyperparameters become flags of
    the command line, and CUDA_VISIBLE_DEVICES is set before its module (and torch) is imported."""
    algo, device_id = find_algorithm(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id if device_id is not None else "0")
    # deterministic cuBLAS (torch.use_deterministic_algorithms); must be set before CUDA starts
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    server_cls = get_server_class(algo) if algo else None
    cli_values, cli_hp, control = parse_cli(argv, server_cls.HPARAMS if server_cls else None, algo)
    file_values, file_hp, origin = {}, {}, "yaml"
    if control["config"]:
        origin = f"yaml:{control['config']}"
        file_values, file_hp = read_yaml(control["config"])
    values, sources, hp_raw, hp_src = merge_sources(cli_values, cli_hp, file_values, file_hp, origin)
    if values.get("algorithm") is None or server_cls is None:
        raise ConfigError(f"missing algorithm.name (-algo); registered: {sorted(ALGORITHMS)}")
    args = finalize_config(values, sources, hp_raw, hp_src, server_cls)
    args.config_file = os.path.abspath(control["config"]) if control["config"] else None
    if server_cls.USES_TORCH:
        if args.device == "cuda":
            try:
                import torch
                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False
            if not cuda:
                args.notes.append("CUDA is not available: running on the CPU")
                args.device = "cpu"
        args.device_used = args.device
    else:
        args.device_used = "cpu (numpy algorithm; -dev is not used)"
    return args, server_cls, control


def _data_config(args, seed: int) -> DataConfig:
    return DataConfig(dataset=args.dataset, partition=args.partition, num_clients=args.num_clients or 0,
                        alpha=args.data_alpha if args.data_alpha is not None else 0.0,
                        class_per_task=args.class_per_task, seed=seed, use_val=args.use_val,
                        min_train_per_client=args.min_train_per_client, dataset_root=args.dataset_root)


def preflight(args, verify_cache: bool = False) -> dict:
    """Load the scenario of every seed and open the feature cache (or check the images), so a run cannot
    fail on missing data."""
    runs, first = [], None
    for i in range(args.times):
        cfg = _data_config(args, args.seed + i)
        set_data_config(cfg)
        scn = load_scenario(cfg)
        runs.append({"run": i, "seed": args.seed + i, "scenario": scn.config_name,
                     "config_dir": os.path.abspath(cfg.config_dir)})
        if first is None:
            first = scn
        elif (scn.num_tasks, scn.num_clients, scn.num_classes, scn.raw_index_hash) != \
                (first.num_tasks, first.num_clients, first.num_classes, first.raw_index_hash):
            raise ConfigError(f"{scn.config_name} does not match {first.config_name} (tasks, clients or raw index)")
    plan = {"runs": runs, "num_tasks": first.num_tasks, "num_classes": first.num_classes,
            "num_clients": first.num_clients, "task_sizes": [len(c) for c in first.tasks]}
    if args.input_kind != FEATURES:
        if verify_cache:
            raise ConfigError(f"--verify_cache: {args.algorithm} reads images, there is no feature cache to verify")
        from utils.image_utils import ImageStore

        store = ImageStore(args.dataset, dataset_root=args.dataset_root, num_workers=0)
        check_compatible(first, store)
        row = int(first.train_rows[0][int(np.argmax([len(r) for r in first.train_rows[0]]))][0])
        img = store.load_image(row)                     # one training image decodes
        plan.update({"images": store.num_samples, "image_check": f"row {row}: {img.size[0]} x {img.size[1]} "
                                                                 f"{img.mode}", "index_sha256": store.raw_index_hash})
        return plan
    view = FeatureView(group=args.feature_group, blocks=tuple(args.blocks) if args.blocks is not None else None,
                       tokens=tuple(args.tokens) if args.tokens is not None else None, proj_rank=args.proj_rank,
                       proj_seed=args.proj_seed if args.proj_seed is not None else (0 if args.proj_rank else None))
    store = FeatureStore(args.dataset, args.backbone, view, dataset_root=args.dataset_root)
    check_compatible(first, store)
    if verify_cache:
        store.verify_checksum()
    plan.update({"feature_dim": store.dim, "depth_groups": store.depth_groups, "blocks": store.blocks,
                 "tokens": store.tokens, "cache_sha256": store.manifest.get("sha256"), "cache_verified": verify_cache})
    return plan


def make_exp_dir(args, plan: dict) -> str:
    if args.exp_dir:
        path = args.exp_dir
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; every experiment gets a new folder")
    else:
        scenario_tag = re.sub(r"_seed\d+", "", plan["runs"][0]["scenario"])
        flags = "".join(f"_{f}" for f in args.run_flags)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        reads = f"{args.backbone}-{args.feature_group}" if args.input_kind == FEATURES else model_tag(args)
        base = os.path.join(args.results_root, args.dataset, scenario_tag,
                            f"{args.algorithm}{flags}_{reads}_{args.goal}_{stamp}")
        path, n = base, 2
        while os.path.exists(path):
            path, n = f"{base}-{n}", n + 1
    os.makedirs(path)
    return path


def environment_record() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    # pip distribution names: version numbers come from the installed metadata (open_clip has no __version__)
    for mod, dist in (("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"),
                      ("open_clip", "open_clip_torch"), ("PIL", "Pillow"), ("yaml", "PyYAML")):
        if mod in sys.modules:
            try:
                versions[mod] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                versions[mod] = getattr(sys.modules[mod], "__version__", "unknown")
    torch = sys.modules.get("torch")
    if torch is not None:
        versions["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    here = os.path.dirname(os.path.abspath(__file__))
    git = {"commit": None, "dirty": None}
    try:
        git["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                                       timeout=10).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True,
                                timeout=10).stdout
        git["dirty"] = bool(status.strip()) if git["commit"] else None
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        pass
    return {"versions": versions, "git": git}


def run_experiment(args, server_cls, plan: dict, protocol: list, argv: list) -> tuple[str, bool]:
    exp_dir = make_exp_dir(args, plan)
    args.exp_dir = exp_dir
    write_json(os.path.join(exp_dir, "experiment.json"), {
        "format": "fcl-experiment/1", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": ["main.py", *argv], "cwd": os.getcwd(), "algorithm": args.algorithm,
        "algorithm_class": args.algorithm_class, "regime": args.regime, "run_flags": list(args.run_flags),
        "settings": settings_record(args), "sources": args.sources, "hparam_sources": args.hparam_sources,
        "notes": args.notes, "device_used": args.device_used, "config_file": args.config_file,
        "protocol": protocol, "plan": plan, "wire_rule": WIRE_RULE, **environment_record()})
    failures = {}
    for i in range(args.times):
        try:
            server = server_cls(args, i)
            server.train()
        except KeyboardInterrupt:
            print(f"\n[main] interrupted during run {i}; {exp_dir} stays incomplete", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001 - recorded in the run's log and in avg/summary.json
            traceback.print_exc()
            failures[i] = f"{type(exc).__name__}: {exc}"
            print(f"[main] run {i} failed; the remaining runs are not started", file=sys.stderr)
            break
    ok = not failures
    summary = compute_averages(exp_dir, failures)
    checks = check_experiment(exp_dir)
    failed = [c for c in checks if not c[1]]
    print("\n" + "=" * 78)
    for name, m in summary["metrics"].items():
        if name in ("acc_final", "acc_avg_incremental", "client_acc_final_mean", "client_acc_final_min"):
            sd = f" +- {m['sd']:.4f}" if m["sd"] is not None else ""
            print(f"{name:24s} {m['mean']:.4f}{sd}  (n = {m['n']}, SD with ddof=1)")
    if args.run_flags:
        print(f"flags: {' '.join(args.run_flags)}")
    print(f"checks: {len(checks) - len(failed)} passed, {len(failed)} failed")
    for name, _ok, detail in failed:
        print(f"  FAIL {name}" + (f" ({detail})" if detail else ""))
    print("=" * 78)
    return exp_dir, ok and not failed


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list_algorithms" in argv:
        print("\n".join(algorithms_table()))
        return 0
    if "--list_settings" in argv:
        print("\n".join(settings_table()))
        return 0
    try:
        args, server_cls, control = resolve_args(argv)
        plan = preflight(args, control["verify_cache"])
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"\n[main] cannot run: {exc}", file=sys.stderr)
        return 2
    protocol = describe_protocol(args, plan)
    print("\n" + "=" * 78 + "\n" + "\n".join(protocol) + "\n" + "=" * 78)
    if control["dry_run"]:
        print(f"[main] dry run: settings and data are consistent for all {args.times} seed(s); nothing was run")
        return 0
    try:
        exp_dir, ok = run_experiment(args, server_cls, plan, protocol, argv)
    except FileExistsError as exc:
        print(f"\n[main] cannot run: {exc}", file=sys.stderr)
        return 2
    print(f"EXPERIMENT_DIR={exp_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
