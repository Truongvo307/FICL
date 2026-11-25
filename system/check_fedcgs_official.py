#!/usr/bin/env python
"""
Run the OFFICIAL FedCGS code on our cached features and compare it with our FedCGS, task by task.

    cd system
    git clone https://github.com/Yuqin-G/FedCGS.git ../../FedCGS        # once (no git? download the zip)
    python check_fedcgs_official.py --official ../../FedCGS -cfg configs/fedcgs/fedcgs_cifar100_vit_in21k_b0.5.yaml
    (any further option goes to main.py's settings, e.g. -droot <dataset root>)

What it does, for one scenario seed (--run, default 0):
  1. runs our FedCGS (the config as given) in a temporary folder and keeps the head after every task;
  2. for each task t (or only --task t), gives the official code the same data our server had after task t:
     client k's training features of every task up to t, in the same client split, labels renumbered
     0..m-1 in the order of the seen classes. The official client loop, server aggregation and covariance
     inverse are executed from the official main_fedcgs.py itself (the statements from `cs_l = []` to the
     end of the gamma loop, read with ast; nothing is copied into this repository), with a stand-in
     backbone whose features are our cached features. Its classifier comes from the official
     utils_fl.eval();
  3. compares the official W and b with ours, the predictions on the test images of the seen classes, and
     the accuracies.
Needs torch (and torchvision, which utils_fl imports); wandb is replaced by an empty module.

Two questions, kept apart:
  * Is the classifier the same? PASS when the official W and b (the float32 values its fc layer holds)
    equal ours to a relative 1e-6; float32 itself rounds at about 6e-8, so a real difference in the method
    (a wrong mean, covariance, prior or inverse) shows as a far larger number. Also printed: our W and b
    evaluated in float32 on the same device against the official predictions (the same classifier in the
    same arithmetic), which is ~100 % unless float32 is too coarse for these logits altogether.
  * How much does float32 change the answers? Reported, not judged: the official code computes the logits
    in float32, ours in float64. With an ill-conditioned covariance, S^-1 and the logits are large and the
    gap between the two best classes is small next to them; float32 keeps about 7 digits, so some
    predictions flip. The line shows the median top logit and top-2 gap of the float64 logits, in the
    official form and in the centred form our predict() uses (same argmax, much smaller numbers).
Exit code 0 on pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_official(official_dir: str):
    """(utils_fl module, compiled statistics block of main_fedcgs.py) from a clone of the official repository."""
    for name in ("main_fedcgs.py", "utils_fl.py"):
        if not os.path.exists(os.path.join(official_dir, name)):
            raise SystemExit(f"{official_dir}: no {name}; give the folder of a clone of github.com/Yuqin-G/FedCGS")
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))           # imported by utils_fl, never used here
    sys.modules.setdefault("registry", types.ModuleType("registry"))     # only used by its dataset helpers
    spec = importlib.util.spec_from_file_location("fedcgs_official_utils_fl", os.path.join(official_dir, "utils_fl.py"))
    utils_fl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_fl)
    for fn in ("eval", "test"):
        if not callable(getattr(utils_fl, fn, None)):
            raise SystemExit(f"utils_fl.py has no {fn}(); this check was written for commit e02cc1f")

    path = os.path.join(official_dir, "main_fedcgs.py")
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    block = None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cs_l" for t in stmt.targets):
                block = body[i:]
                break
        if block:
            break
    if not block or not any(isinstance(s, ast.For) and "gamma" in ast.get_source_segment(source, s) for s in block):
        raise SystemExit(f"{path}: cannot find the statistics block (cs_l = [] ... the gamma loop); this check "
                         f"was written for commit e02cc1f")
    segment = "\n".join(ast.get_source_segment(source, s) for s in block)
    return utils_fl, compile(ast.Module(body=block, type_ignores=[]), path, "exec"), segment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Official FedCGS vs ours on the same cached features")
    p.add_argument("--official", required=True, help="folder of a clone of github.com/Yuqin-G/FedCGS")
    p.add_argument("-cfg", "--config", required=True, help="a FedCGS config (configs/fedcgs/*.yaml)")
    p.add_argument("--run", type=int, default=0, help="which run of the config (scenario seed = seed + run)")
    p.add_argument("--task", type=int, default=None, help="only this task (default: every task)")
    p.add_argument("--device", default="cuda:0", help="device for the official code (cuda:0 or cpu)")
    p.add_argument("--show_block", action="store_true", help="print the official statements that are executed")
    a, extra = p.parse_known_args(argv)          # anything else goes to main.py's settings, e.g. -droot

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from main import resolve_args
    from flcore.servers.serverfedcgs import FedCGS

    utils_fl, block, segment = load_official(os.path.abspath(a.official))
    if a.show_block:
        print(segment)
    args, server_cls, _ = resolve_args(["-cfg", a.config, "-t", str(a.run + 1), *extra])
    if server_cls is not FedCGS:
        raise SystemExit(f"{a.config}: algorithm is {args.algorithm}, not FedCGS")
    gamma = float(args.hparams["gamma"])

    class Snapshots(FedCGS):
        def init_algorithm(self):
            super().init_algorithm()
            self.heads = []

        def on_end_task(self, t):
            super().on_end_task(t)
            self.heads.append((self.head_classes.copy(), self.W.copy(), self.b.copy(), dict(self.head_info)))

    tmp = tempfile.mkdtemp(prefix="fedcgs_check_")
    try:
        args.exp_dir = tmp
        ours = Snapshots(args, a.run)
        ours.train()
        scn, store = ours.scenario, ours.store
        tasks = range(scn.num_tasks) if a.task is None else [a.task]
        device = torch.device(a.device)

        class Backbone(torch.nn.Module):
            """Stand-in for the official ResNet-18: its 'features' are our cached features."""
            def __init__(self, d, m):
                super().__init__()
                self.fc = torch.nn.Linear(d, m)

            def forward(self, x, return_features=False):
                out = self.fc(x)
                return (out, x) if return_features else out

        captured = {}
        original_test = utils_fl.test

        def capture_test(model, test_loader, device_unused="cuda:0"):
            captured["W"] = model.fc.weight.detach().double().cpu().numpy()
            captured["b"] = model.fc.bias.detach().double().cpu().numpy()
            preds = []
            with torch.no_grad():
                for x, _ in test_loader:
                    preds.append(model.to(device)(x.to(device)).argmax(dim=1).cpu().numpy())
            captured["pred"] = np.concatenate(preds)
            captured["acc"] = original_test(model, test_loader, device=a.device)
            return captured["acc"]

        utils_fl.test = capture_test
        status = 0
        tf32 = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if str(device).startswith("cuda") else False
        print(f"official FedCGS ({os.path.abspath(a.official)}) vs ours, {args.dataset} {scn.config_name}, "
              f"features {store.describe()}, d = {store.dim}, gamma = {gamma:g}, official device {device}"
              f"{' (TF32 matmul ON)' if tf32 else ''}")
        print("  classifier (decides PASS): official W, b vs ours; ours in float32 on the same device vs official")
        print("  float32 effect (reported): official (float32) vs ours (float64, the numbers our runs report)")
        for t in tasks:
            classes, W, b, info = ours.heads[t]
            remap = {int(c): j for j, c in enumerate(classes)}
            loaders = []
            for k in range(scn.num_clients):
                rows = np.sort(np.concatenate([scn.train_rows[i][k] for i in range(t + 1)]))
                Z = store.features(rows).astype(np.float32)
                y = np.array([remap[int(c)] for c in store.labels(rows)], dtype=np.int64)
                loaders.append(DataLoader(TensorDataset(torch.from_numpy(Z), torch.from_numpy(y)),
                                          batch_size=1024, shuffle=False))
            with store.evaluation():
                test_rows = np.concatenate([scn.test_global_rows[i] for i in range(t + 1)])
                Zt = store.features(test_rows)
                yt = store.labels(test_rows)
            ours_pred = classes[np.argmax((Zt - info["mean"]) @ info["W_centered"].T + info["b_centered"],
                                          axis=1)]                    # our head after task t, as predict()
            plain_pred = classes[np.argmax(Zt @ W.T + b, axis=1)]       # the official form, in float64
            test_loader = DataLoader(TensorDataset(torch.from_numpy(Zt.astype(np.float32)),
                                                   torch.from_numpy(np.array([remap[int(c)] for c in yt]))),
                                     batch_size=256, shuffle=False)
            ns = {"torch": torch, "np": np, "math": __import__("math"), "train_loader_list": loaders,
                  "num_classes": len(classes), "num_d": int(store.dim), "device": device,
                  "model": Backbone(int(store.dim), len(classes)).eval(), "test_loader": test_loader,
                  "eval": utils_fl.eval}
            code = block if gamma == 0 else compile(
                segment.replace("gamma_list = [0]", f"gamma_list = [{gamma!r}]"), "main_fedcgs.py (gamma set)", "exec")
            captured.clear()
            exec(code, ns)                                       # the official statements, unchanged
            off_pred = classes[captured["pred"]]
            logits = Zt @ W.T + b                                       # official parametrization, float64
            # 1. the classifier: official W, b (float32 values) evaluated exactly, against ours
            off64_pred = classes[np.argmax(Zt @ captured["W"].T + captured["b"], axis=1)]
            agree64 = float(np.mean(off64_pred == ours_pred))
            dW = np.max(np.abs(captured["W"] - W)) / np.max(np.abs(W))
            db = np.max(np.abs(captured["b"] - b)) / np.max(np.abs(b))
            # 2. float32 evaluation: the official predictions, and ours in float32 on the same device
            with torch.no_grad():
                l32 = (torch.from_numpy(Zt.astype(np.float32)).to(device)
                       @ torch.from_numpy(W.astype(np.float32)).to(device).T
                       + torch.from_numpy(b.astype(np.float32)).to(device))
            ours32_pred = classes[l32.argmax(dim=1).cpu().numpy()]
            agree32 = float(np.mean(off_pred == ours_pred))
            ctrl = float(np.mean(ours32_pred == off_pred))
            cmax = ((Zt - info["mean"]) @ info["W_centered"].T + info["b_centered"]).max(axis=1)
            top2 = np.sort(logits, axis=1)[:, -2:]
            margin, scale = top2[:, 1] - top2[:, 0], np.abs(top2[:, 1])
            acc_ours = 100.0 * float(np.mean(ours_pred == yt))
            acc_ours32 = 100.0 * float(np.mean(ours32_pred == yt))
            acc_off = float(captured["acc"])
            ok = dW <= 1e-6 and db <= 1e-6
            status |= 0 if ok else 1
            print(f"task {t}: {len(classes)} classes, {len(yt)} test images, covariance condition number "
                  f"{info['condition_number']:.2e} | {'PASS' if ok else 'FAIL'}")
            print(f"    classifier: max rel. diff W {dW:.1e}, b {db:.1e}; ours in float32 = official on "
                  f"{100 * ctrl:.2f} % of predictions; accuracy official {acc_off:.2f}, ours float32 {acc_ours32:.2f}")
            print(f"    float32 effect: ours float64 {acc_ours:.2f} vs official {acc_off:.2f}, predictions agree "
                  f"{100 * agree32:.2f} % (official W, b evaluated in float64: {100 * agree64:.2f} %; our W, b in the "
                  f"official, uncentred form in float64: {100 * float(np.mean(plain_pred == ours_pred)):.2f} %); "
                  f"median |top logit| {np.median(scale):.3g} (centred: {np.median(np.abs(cmax)):.3g}), median "
                  f"top-2 gap {np.median(margin):.3g}")
        utils_fl.test = original_test
        return status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
