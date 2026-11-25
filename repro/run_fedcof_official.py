#!/usr/bin/env python
"""FedCOF (NeurIPS 2025) paper setting with the official FedCOF repository, then a dump for run_ours.py.

    python repro/run_fedcof_official.py --data /data/cifar --seeds 0 1 2 3 4
    python repro/run_ours.py repro/work/dumps/fedcof_cifar100_vit_a0.1_K100_seed0.npz      (FICL environment)

Paper setting (Table 3, ViT-B/16 column): CIFAR-100, 100 clients, Dirichlet alpha 0.1, 5 seeds, gamma 1.
The repository's own scripts are run unchanged, as its README and bash_for_*/ scripts do:

 1. client_accumulate.py   splits CIFAR-100 to the clients (its Dirichlet sharder) and stores every client's
                           features of the timm ViT-B/16 (train_client_data.pt, test_data.pt)
 2. server_aggregate.py    for each approach fedncm, fed3r (the repository's Fed3R variant, no bias column)
                           and fedcof: the official accuracy (its summary.csv)

then, in this process, from the files step 1 wrote:
 3. the same official classes (federate_classes.py) give the classifiers and predictions behind step 2's
    accuracies (checked: they must reproduce step 2's numbers), and the official Fed3R functions of the Fed3R
    repository (fed3r_step, calc_ridge_params, normalize_W; with bias) run on the same features;
 4. everything goes into a dump: features in the official row order, client of every row, test features,
    official predictions and classifiers.

Our FedCOF is compared with the official FedCOF, our Fed3R with the official Fed3R functions.
Needs: torch, torchvision, timm, tqdm, pandas (FedCOF's imports); a GPU is used when available.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DUMP_FORMAT, accuracy, save_dump  # noqa: E402
from fetch_official import REPOS  # noqa: E402
from official_common import (batches, check_modules, commit_of, official_fed3r_functions,  # noqa: E402
                             require_commit, run_logged, FeatureBackbone)

GAMMA = {"vit": 1.0, "squeezenet": 1.0, "mobilenet": 0.1}       # paper Appendix: gamma per backbone
DIM = {"vit": 768, "squeezenet": 512, "mobilenet": 1280}
PAPER_KEYS = {"vit": ["fedcof_neurips25_t3_vit_fedncm", "fedcof_neurips25_t3_vit_fed3r",
                      "fedcof_neurips25_t3_vit_fedcof"]}
APPROACHES = {"fedncm": "FedNCM", "fed3r": "Fed3R_fedcof_repo", "fedcof": "FedCOF"}


def feature_dir(work, model, alpha, K, seed):
    return os.path.join(work, "federated_features", f"cifar100_alpha_{alpha}_K_{K}_feats", model, f"seed_{seed}")


def run_official_scripts(a, seed, py) -> dict:
    """Steps 1 and 2: the repository's scripts as its README runs them. Returns {approach: accuracy}."""
    fdir = feature_dir(a.work, a.model, a.alpha, a.num_clients, seed)
    if not (os.path.isfile(os.path.join(fdir, "train_client_data.pt")) and
            os.path.isfile(os.path.join(fdir, "test_data.pt"))):
        run_logged([py, "-u", os.path.join(a.official, "client_accumulate.py"), "--dataset", "cifar100",
                    "--model_type", a.model, "--seed", str(seed), "--dataset_path", a.data,
                    "--dst_path", os.path.join(a.work, "federated_features"), "--alpha", str(a.alpha),
                    "--num_clients", str(a.num_clients), "--num_workers", str(a.num_workers)],
                   cwd=a.work, log_path=os.path.join(a.work, "logs", f"client_accumulate_{a.model}_seed{seed}.log"))
    accs = {}
    for approach in APPROACHES:
        out = os.path.join(a.work, "outputs", f"{approach}_{a.model}_seed{seed}")
        os.makedirs(out, exist_ok=True)
        text = run_logged([py, "-u", os.path.join(a.official, "server_aggregate.py"), "--dataset", "cifar100",
                           "--model_type", a.model, "--seed", str(seed), "--seed_to_load", str(seed),
                           "--approach", approach, "--gamma_shrink", str(GAMMA[a.model]), "--lamb", str(a.lamb),
                           "--src_path", os.path.join(a.work, "federated_features"), "--alpha", str(a.alpha),
                           "--num_clients", str(a.num_clients), "-op", out],
                          cwd=a.work, log_path=os.path.join(a.work, "logs", f"server_{approach}_{a.model}_seed{seed}.log"))
        m = re.findall(r"Approach (\w+) Test Accuracy ([0-9.]+)", text)
        if not m:
            raise SystemExit(f"server_aggregate.py printed no accuracy for {approach}")
        accs[approach] = float(m[-1][1])
        rows = []
        for path in glob.glob(os.path.join(out, "*", "summary.csv")):
            with open(path) as f:
                rows += list(csv.DictReader(f))
        print(f"official {approach}: {accs[approach]:.2f} % (summary.csv rows: {len(rows)})")
    return accs


def build_dump(a, seed, printed: dict) -> str:
    """Steps 3 and 4."""
    import torch
    sys.path.insert(0, a.official)
    import federate_classes as fc                     # the official module, unchanged

    fdir = feature_dir(a.work, a.model, a.alpha, a.num_clients, seed)
    with open(os.path.join(fdir, "train_client_data.pt"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(fdir, "test_data.pt"), "rb") as f:
        test = pickle.load(f)
    C, d = 100, DIM[a.model]
    users = sorted(train, key=lambda u: int(u))       # client ids as written by client_accumulate.py
    if len(users) != a.num_clients:
        print(f"note     {len(users)} of {a.num_clients} clients hold data")

    # step 3a: the official classes, exactly as server_aggregate.py calls them (dict order of the file)
    official, arrays = {}, {}
    Zte = test["data"].float()
    yte = test["labels"].numpy().astype(np.int64)
    for approach, key in APPROACHES.items():
        client_cls = {"fedncm": fc.FedNCMClient, "fed3r": fc.Fed3RClient, "fedcof": fc.FedCOFClient}[approach]
        server_data = {}
        for user_id, dd in train.items():
            cl = client_cls(user_id, dd["data"], dd["labels"])
            cl.compute()
            server_data[user_id] = cl.send()
        if approach == "fedncm":
            srv = fc.FedNCMServer(server_data, C, d)
        elif approach == "fed3r":
            srv = fc.Fed3RServer(server_data, C, d, a.lamb)
        else:
            srv = fc.FedCOFServer(server_data, C, d, GAMMA[a.model])
        srv.aggregate()
        W = srv.get_classifier_weights().float()                     # [C, d], as model.head.weight
        pred = torch.max(torch.nn.functional.linear(Zte, W), dim=1)[1].numpy().astype(np.int64)
        acc = accuracy(pred, yte)
        if abs(acc - printed[approach]) > 0.02:
            print(f"WARNING  {approach}: the official classes give {acc:.2f} % here, server_aggregate.py printed "
                  f"{printed[approach]:.2f} % (device arithmetic); the dump keeps the predictions computed here")
        official[key] = {"acc": [acc], "printed_by_official_script": printed[approach],
                         "hparams": {"gamma_shrink": GAMMA[a.model]} if approach == "fedcof" else
                         ({"lamb": a.lamb} if approach == "fed3r" else {}),
                         "precision": "float32 (as released)", "notes": f"federate_classes.py {srv.__class__.__name__}"}
        arrays[f"pred__{key}__0"] = pred
        arrays[f"W__{key}__0"] = W.double().numpy()
        arrays[f"Wcls__{key}__0"] = np.arange(C, dtype=np.int64)

    # rows in client order
    xs, ys, ks = [], [], []
    for k, u in enumerate(users):
        xs.append(train[u]["data"].float().numpy())
        ys.append(train[u]["labels"].numpy().astype(np.int64))
        ks.append(np.full(len(ys[-1]), k, dtype=np.int64))
    Xtr, ytr, ktr = np.concatenate(xs), np.concatenate(ys), np.concatenate(ks)

    # step 3b: the official Fed3R functions (Fed3R repository) on the same features, float32 as released
    f3 = official_fed3r_functions(a.fed3r_official, torch)
    A = a.lamb * torch.eye(d + 1)
    b = torch.zeros((d + 1, C))
    for k in range(len(users)):
        rows = ktr == k
        A, b = f3["fed3r_step"](A, b, batches(torch, Xtr[rows], ytr[rows]), FeatureBackbone(), C, "cpu")
    W, bias = f3["normalize_W"](*f3["calc_ridge_params"](A, b), warnings=False)
    logits = Zte @ W + bias
    pred = torch.max(logits, dim=1)[1].numpy().astype(np.int64)
    official["Fed3R"] = {"acc": [accuracy(pred, yte)], "hparams": {"lambda_": a.lamb},
                         "precision": "float32 (as released)",
                         "notes": "Fed3R repository src/fed3r.py fed3r_step / calc_ridge_params / normalize_W"}
    arrays["pred__Fed3R__0"] = pred
    arrays["W__Fed3R__0"] = W.double().numpy().T
    arrays["b__Fed3R__0"] = bias.double().numpy()
    arrays["Wcls__Fed3R__0"] = np.arange(C, dtype=np.int64)

    arrays.update(train_x=Xtr.astype(np.float32), train_y=ytr, train_task=np.zeros(len(ytr), np.int64),
                  train_client=ktr, test_x=Zte.numpy().astype(np.float32), test_y=yte,
                  evalrows__0=np.arange(len(yte), dtype=np.int64))
    meta = {"format": DUMP_FORMAT,
            "source": {"repo": "dipamgoswami/FedCOF", "commit": commit_of(a.official),
                       "entry": "client_accumulate.py + server_aggregate.py (unchanged)",
                       "fed3r_repo": "Erosinho13/Fed3R", "fed3r_commit": commit_of(a.fed3r_official),
                       "notes": "features as written by client_accumulate.py (training features with its "
                                "RandomHorizontalFlip, drawn once)"},
            "setting": {"dataset": "CIFAR-100", "backbone": f"FedCOF repo '{a.model}'", "num_classes": C,
                        "num_clients": len(users), "partition_label": "dir", "alpha": a.alpha,
                        "tasks": [list(range(C))], "seed": seed},
            "official": official,
            "ours": [{"algorithm": "FedCOF", "label": "FedCOF", "compare_to": "FedCOF",
                      "hparams": {"gamma_shrink": GAMMA[a.model], "means_per_client": 1}},
                     {"algorithm": "Fed3R", "label": "Fed3R", "compare_to": "Fed3R",
                      "hparams": {"ridge_lambda": a.lamb, "rf_dim": 0}}],
            "paper": PAPER_KEYS.get(a.model, [])}
    path = os.path.join(a.work, "..", "dumps", f"fedcof_cifar100_{a.model}_a{a.alpha}_K{a.num_clients}_seed{seed}.npz")
    path = os.path.abspath(path)
    save_dump(path, meta, arrays)
    print(f"dump     {path}")
    for key, o in official.items():
        print(f"         official {key:<18} {o['acc'][0]:.2f} %")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--official", default=os.path.join(HERE, "official_src", "FedCOF"))
    p.add_argument("--fed3r_official", default=os.path.join(HERE, "official_src", "Fed3R"))
    p.add_argument("--data", required=True, help="folder for torchvision's CIFAR-100 (FedCOF --dataset_path)")
    p.add_argument("--work", default=os.path.join(HERE, "work", "fedcof"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--model", choices=sorted(DIM), default="vit")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--num_clients", type=int, default=100)
    p.add_argument("--lamb", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--python", default=sys.executable, help="interpreter of the official environment")
    a = p.parse_args(argv)
    a.official, a.fed3r_official, a.work = map(os.path.abspath, (a.official, a.fed3r_official, a.work))
    a.data = os.path.abspath(a.data)
    require_commit(a.official, REPOS["FedCOF"][1], "FedCOF")
    require_commit(a.fed3r_official, REPOS["Fed3R"][1], "Fed3R")
    check_modules(["torch", "torchvision", "timm", "tqdm", "pandas"])
    os.makedirs(a.work, exist_ok=True)
    for seed in a.seeds:
        printed = run_official_scripts(a, seed, a.python)
        build_dump(a, seed, printed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
