#!/usr/bin/env python
"""AFL (CVPR 2025) paper setting with the official AFL repository, then a dump for run_ours.py.

    python repro/run_afl_official.py --data /data/afl --seeds 1 2 3
    python repro/run_ours.py repro/work/dumps/afl_cifar100_resnet18_a0.1_K100_seed1.npz       (FICL environment)

Paper setting (Table 2, CIFAR-100): ResNet-18 pre-trained on ImageNet-1k, 100 clients, Dirichlet alpha 0.1,
gamma 1 with the Regularization Intermediary, 3 runs. The command line is the "CIFAR-100 (Original for
comparison)" line of the repository's CIFAR-100/run.sh, plus --seed (partition seed) and --data:

    main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset cifar100
            --num_classes 100 --modelseed 1 --pretrained --seed <s> --data <data>

main.py runs unchanged, started in-process with runpy, one process per seed. Around four of its functions a
thin wrapper records what they compute and then returns the original result untouched:
    LinearAnalytic.forward   the backbone features that local_update and validate pass through it
    F.one_hot (in afl.py)    the training labels local_update one-hot encodes (same order as the features)
    correct (in afl.py)      predictions and labels of the global model's test evaluation
    clean_regularization     the final classifier
The dump holds, per client (in the order main.py trains them), exactly the features and labels that entered
X^T X and X^T Y, the test features, and the official predictions after regularization cleaning.
Needs: torch, torchvision, scikit-learn, ujson (AFL's imports). numpy >= 2 gets numpy.mat = numpy.asmatrix.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DUMP_FORMAT, accuracy, save_dump  # noqa: E402
from fetch_official import REPOS  # noqa: E402
from official_common import check_modules, commit_of, require_commit, shims_applied, subprocess_env  # noqa: E402


def official_argv(a, seed) -> list:
    return ["main.py", "--niid", "--partition", "dir", "--rg", str(a.rg), "--num_clients", str(a.num_clients),
            "--alpha", str(a.alpha), "--clean_reg", "--dataset", "cifar100", "--num_classes", "100",
            "--modelseed", "1", "--pretrained", "--seed", str(seed), "--data", a.data, "--gpu", str(a.gpu)]


def hook(a) -> int:
    """Run main.py of the official repository in this process with recording wrappers; write the dump."""
    import torch
    code_dir = os.path.join(a.official, "CIFAR-100")
    sys.path.insert(0, code_dir)
    shims = shims_applied()
    import afl                                                  # official module, unchanged
    import dataset as afl_dataset                               # official CIFAR-100/dataset.py, unchanged

    # Bug at 2a5f7ea: in the non-continual setting (the paper's) prepare_data returns 4 values
    # (trainset, idx, testset, None) but main.py unpacks 3. Drop the 4th (task_info = None) for main.py.
    orig_prepare = afl_dataset.prepare_data

    def prepare_data(args):
        out = orig_prepare(args)
        if not getattr(args, "continual", False) and isinstance(out, tuple) and len(out) == 4 and out[3] is None:
            shims.append("dataset.prepare_data: 4th return value (task_info None) dropped; main.py unpacks 3 "
                         "in the non-continual setting (bug at 2a5f7ea)")
            return out[:3]
        return out
    afl_dataset.prepare_data = prepare_data

    st = {"mode": None, "x": [], "y": [], "global": None}
    clients, tests, cleaned = [], [], []

    orig_forward = afl.LinearAnalytic.forward

    def forward(self, x):
        if st["mode"] is not None and (st["mode"] == "train" or self is st["global"]):
            st["x"].append(x.detach().float().cpu())
        return orig_forward(self, x)

    class FProxy:
        def __getattr__(self, name):
            return getattr(torch.nn.functional, name)

        def one_hot(self, labels, *args, **kw):
            if st["mode"] == "train":
                st["y"].append(labels.detach().cpu())
            return torch.nn.functional.one_hot(labels, *args, **kw)

    orig_local, orig_validate, orig_correct = afl.local_update, afl.validate, afl.correct
    orig_clean = afl.clean_regularization

    def local_update(train_loader, model, global_model, args):
        st.update(mode="train", x=[], y=[], **{"global": global_model})
        try:
            out = orig_local(train_loader, model, global_model, args)
        finally:
            st["mode"] = None
        X, y = torch.cat(st["x"]).numpy(), torch.cat(st["y"]).numpy().astype(np.int64)
        if len(X) != len(y) or len(y) != len(train_loader.dataset):
            raise RuntimeError(f"recorded {len(X)} features / {len(y)} labels for a client of "
                               f"{len(train_loader.dataset)} samples")
        clients.append((X, y))
        return out

    def correct(output, target):
        if st["mode"] == "test":
            st["pred"].append(torch.max(output, dim=1)[1].detach().cpu())
            st["y"].append(target.detach().cpu())
        return orig_correct(output, target)

    def validate(val_loader, model, global_model, args):
        is_test = global_model is st["global"]
        if is_test:
            st.update(mode="test", x=[], y=[], pred=[])
        try:
            out = orig_validate(val_loader, model, global_model, args)
        finally:
            if is_test:
                st["mode"] = None
        if is_test:
            tests.append({"x": torch.cat(st["x"]).numpy(), "y": torch.cat(st["y"]).numpy().astype(np.int64),
                          "pred": torch.cat(st["pred"]).numpy().astype(np.int64)})
        return out

    def clean_regularization(W, C, args):
        out = orig_clean(W, C, args)
        cleaned.append(out.detach().double().cpu().numpy())
        return out

    afl.LinearAnalytic.forward = forward
    afl.F = FProxy()
    afl.local_update, afl.validate, afl.correct = local_update, validate, correct
    afl.clean_regularization = clean_regularization

    argv = official_argv(a, a.seed)
    sys.argv = argv
    os.makedirs(a.work, exist_ok=True)
    os.chdir(a.work)                                            # main.py writes its csv into the working folder
    runpy.run_path(os.path.join(code_dir, "main.py"), run_name="__main__")

    if len(clients) != a.num_clients or len(tests) != 2 or len(cleaned) != 1:
        raise RuntimeError(f"recorded {len(clients)} clients, {len(tests)} test evaluations, {len(cleaned)} "
                           "cleaned classifiers; expected num_clients, 2 (before / after cleaning), 1")
    final = tests[-1]
    W_clean = cleaned[0]                                         # [d, C] float64 (the model uses it as float32)
    xs = [c[0] for c in clients]
    ys = [c[1] for c in clients]
    meta = {"format": DUMP_FORMAT,
            "source": {"repo": "RyougiSh1k1/AFL", "commit": commit_of(a.official), "entry": "CIFAR-100/main.py",
                       "argv": argv, "shims": shims,
                       "notes": "main.py unchanged; features recorded at LinearAnalytic.forward"},
            "setting": {"dataset": "CIFAR-100", "backbone": "torchvision resnet18 (pretrained=True)",
                        "num_classes": 100, "num_clients": a.num_clients, "partition_label": "dir",
                        "alpha": a.alpha, "tasks": [list(range(100))], "seed": a.seed},
            "official": {"AFL": {"acc": [accuracy(final["pred"], final["y"])],
                                 "acc_before_cleaning": accuracy(tests[0]["pred"], tests[0]["y"]),
                                 "hparams": {"rg": a.rg, "clean_reg": True},
                                 "precision": "as released (X^T X in float32, inverses in float64, logits float32)"}},
            "ours": [{"algorithm": "AFL", "label": "AFL", "compare_to": "AFL",
                      "hparams": {"rg": a.rg, "clean_reg": True}}],
            "paper": ["afl_cvpr25_t2_cifar100"]}
    arrays = {"train_x": np.concatenate(xs).astype(np.float32), "train_y": np.concatenate(ys),
              "train_task": np.zeros(sum(len(y) for y in ys), np.int64),
              "train_client": np.concatenate([np.full(len(y), k, np.int64) for k, y in enumerate(ys)]),
              "test_x": final["x"].astype(np.float32), "test_y": final["y"],
              "evalrows__0": np.arange(len(final["y"]), dtype=np.int64), "pred__AFL__0": final["pred"],
              "W__AFL__0": W_clean.T, "Wcls__AFL__0": np.arange(100, dtype=np.int64)}
    save_dump(a.out, meta, arrays)
    print(f"dump     {a.out}\n         official AFL {meta['official']['AFL']['acc'][0]:.2f} % after cleaning "
          f"({meta['official']['AFL']['acc_before_cleaning']:.2f} % before)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--official", default=os.path.join(HERE, "official_src", "AFL"))
    p.add_argument("--data", required=True, help="AFL --data: CIFAR-100 goes to <data>/cifar100")
    p.add_argument("--work", default=os.path.join(HERE, "work", "afl"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--num_clients", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--rg", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.official, a.work, a.data = map(os.path.abspath, (a.official, a.work, a.data))
    if a.hook:
        return hook(a)
    require_commit(a.official, REPOS["AFL"][1], "AFL")
    check_modules(["torch", "torchvision", "sklearn", "ujson"])
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    for seed in a.seeds:
        out = os.path.abspath(os.path.join(HERE, "work", "dumps",
                                           f"afl_cifar100_resnet18_a{a.alpha}_K{a.num_clients}_seed{seed}.npz"))
        if os.path.exists(out):
            print(f"ok       {out} exists")
            continue
        cmd = [a.python, os.path.abspath(__file__), "--hook", "--seed", str(seed), "--out", out,
               "--official", a.official, "--data", a.data, "--work", a.work, "--num_clients", str(a.num_clients),
               "--alpha", str(a.alpha), "--rg", str(a.rg), "--gpu", str(a.gpu)]
        log = os.path.join(a.work, "logs", f"afl_seed{seed}.log")
        print("run      " + " ".join(cmd))
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env=subprocess_env(), bufsize=1)
            for line in proc.stdout:
                f.write(line)
                if re.search(r"accuracy|dump|Error|error|WARNING|Traceback", line):
                    sys.stdout.write("  | " + line)
            if proc.wait() != 0:
                raise SystemExit(f"AFL seed {seed} failed; see {log}")
    print(json.dumps({"dumps": sorted(os.listdir(os.path.join(HERE, "work", "dumps")))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
