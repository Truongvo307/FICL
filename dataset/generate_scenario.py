#!/usr/bin/env python
"""
Generate class-incremental federated scenarios for FICL.

    cd dataset
    python generate_scenario.py TissueMNIST dir --num_clients 10 --alpha 0.1 --class_per_task 2 --seed 0
    python generate_scenario.py TissueMNIST dir --num_clients 10 --alpha 0.1 --class_per_task 2 --seed 0 --num_seeds 3
    python generate_scenario.py TissueMNIST iid --num_clients 10 --class_per_task 2 --seed 0
    python generate_scenario.py DomainNet natural --class_per_task 35 --seed 0
    python generate_scenario.py TissueMNIST --list

Needs <Name>/cl_index (prepare_data.py). Writes <Name>/<config_name>/, for example
cl_10clients_dir_0.1_cpt2_seed0. The name includes the seed, so each seed has its own split;
main.py -t 3 -seed 0 reads seeds 0, 1 and 2.
"""
import argparse
import os
import sys

import numpy as np

from utils.raw_utils import index_dir, load_raw_index
from utils.scenario_utils import PARTITIONS, ScenarioSpec, build_scenario, list_configs, save_scenario

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate a class-incremental FL scenario (FICL)")
    p.add_argument("dataset", help="dataset folder name, e.g. TissueMNIST")
    p.add_argument("partition", nargs="?", choices=PARTITIONS, help="dir | iid | natural")
    p.add_argument("--num_clients", type=int, default=10, help="ignored for natural (one client per domain)")
    p.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration (dir only)")
    p.add_argument("--class_per_task", type=int, default=2, help="classes per task; the last task may be smaller")
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--num_seeds", type=int, default=1, help="generate seeds seed .. seed+num_seeds-1")
    p.add_argument("--use_val", choices=["ignore", "merge"], default="ignore",
                   help="what to do with an official validation split")
    p.add_argument("--min_train_per_client", type=int, default=0,
                   help="redraw a task until every client has this many training samples (0 = off)")
    p.add_argument("--dataset_root", default=HERE)
    p.add_argument("--list", action="store_true", help="list existing scenarios of this dataset")
    return p.parse_args(argv)


def print_summary(scn, idx, path):
    labels = idx.labels
    print(f"\n{'=' * 70}\n{scn.config_name}  ->  {path}")
    print(f"{scn.num_clients} clients, {scn.num_tasks} tasks, {scn.num_classes} classes, "
          f"class order seed streams {scn.sub_seeds}")
    print(f"{'task':>4} | {'classes':<34} | {'train':>7} | {'test':>6}")
    for t, classes in enumerate(scn.tasks):
        names = ", ".join(idx.class_names[c] for c in classes)
        names = names if len(names) <= 34 else names[:31] + "..."
        n_tr = sum(len(r) for r in scn.train_rows[t])
        print(f"{t:>4} | {names:<34} | {n_tr:>7} | {len(scn.test_global_rows[t]):>6}")
    print("\ntraining samples per client (rows) and task (columns):")
    print("client " + " ".join(f"{t:>7}" for t in range(scn.num_tasks)))
    for k in range(scn.num_clients):
        row = [len(scn.train_rows[t][k]) for t in range(scn.num_tasks)]
        n_cls = len(np.unique(np.concatenate([labels[scn.train_rows[t][k]] for t in range(scn.num_tasks)])))
        print(f"{k:>6} " + " ".join(f"{n:>7}" for n in row) + f"   ({n_cls} classes overall)")
    if any(scn.redraws):
        print(f"redraws per task (min_train_per_client): {scn.redraws}")
    print("=" * 70)


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = os.path.join(args.dataset_root, args.dataset)
    if args.list:
        configs = list_configs(dataset_dir)
        print(f"\nScenarios for {args.dataset}:")
        for c in configs:
            print(f"  - {c}")
        if not configs:
            print("  (none)")
        return 0
    if not args.partition:
        print("give a partition (dir, iid or natural), or --list")
        return 1
    if args.partition == "natural" and args.num_clients != 10:
        print(f"note: --num_clients {args.num_clients} is ignored for natural (one client per domain)")

    idx = load_raw_index(index_dir(args.dataset_root, args.dataset))
    for seed in range(args.seed, args.seed + args.num_seeds):
        spec = ScenarioSpec(dataset=args.dataset, partition=args.partition, num_clients=args.num_clients,
                            alpha=args.alpha, class_per_task=args.class_per_task, seed=seed,
                            use_val=args.use_val, min_train_per_client=args.min_train_per_client)
        scn = build_scenario(idx, spec)
        path = save_scenario(scn, dataset_dir)
        print_summary(scn, idx, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
