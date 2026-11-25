#!/usr/bin/env python
"""
Run the OFFICIAL code of Fed3R, FedCOF, STSA, STSA-E or AFL on our cached features and compare it with our
implementation, task by task.

    cd system
    python check_baselines_official.py --method FedCOF --official ../../FedCOF -cfg configs/baselines/fedcof_cifar100_vit_in21k_b0.5.yaml
    (any further option goes to main.py's settings, e.g. -droot <dataset root> or --gamma_shrink 0.5)

Clones needed (one per method; nothing of them is copied into this repository):
    Fed3R   github.com/Erosinho13/Fed3R        src/fed3r.py: fed3r_step, calc_ridge_params, normalize_W
    FedCOF  github.com/dipamgoswami/FedCOF     federate_classes.py: FedCOFClient, FedCOFServer
    STSA    github.com/Yuqin-G/STSA            methods/STSA.py: target2onehot, replace_fc1,
    STSA_E                                     replace_fc_ours3, compute_multiple_sum_by_class
    AFL     github.com/RyougiSh1k1/AFL         CIFAR-100/afl.py: LinearAnalytic, local_update, aggregation,
                                               clean_regularization
Functions are read from the clone with ast and executed unchanged, with stand-ins only for what they get
from outside (a backbone whose features are our cached features, data loaders over our rows, the server
object's fields). Fed3R's and STSA's modules import tensorflow, wandb, sklearn and their own packages,
which is why their functions are extracted rather than imported. Needs torch.

What is compared, for every task t of one run (--run, scenario seed = seed + run):
    Fed3R    every upload so far through fed3r_step in our order, then calc_ridge_params + normalize_W
    FedCOF   FedCOFClient on each client's data of every task so far, FedCOFServer.aggregate (one-shot; the
             classes of different tasks never mix, so this is FedCOF on the seen data)
    STSA(-E) the official methods called once per task on that task's client data, carrying the official
             self.G and self.P from task to task (the official temporal aggregation); our W_rand is given to
             it, and for STSA-E torch.randperm returns our client's permutation, so both use the same groups
    AFL      local_update per upload, aggregation over every upload so far in our order, clean_regularization
Class ids are renumbered 0..C-1 in the order classes arrive, as the official code expects.

Precision. The released code computes partly in float32 (Fed3R: statistics and solve; FedCOF: means and
pinv; STSA: the final W; STSA-E: the estimated G_t; AFL: X^T X and X^T Y), ours in float64. On realistic
features (d = 768, ill-conditioned) float32 alone moves W by 1e-4 to 1e-2 relative, so a tolerance cannot
tell rounding from a different method. By default the official statements therefore run unchanged but
with torch's float32 lifted to float64 (default dtype float64, Tensor.float() returning float64, features
given in float64): then the official classifier must equal ours to --tolerance (default 1e-6), and that
decides PASS / FAIL. --as_released runs the released precision and only reports the differences (no
verdict). Prediction agreement on the test images of the seen classes and both accuracies are printed in
both modes. Exit code 0 when every task passes (always 0 with --as_released).

--device cpu runs the official code without a GPU: .cuda() then returns the object unchanged.
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

OFFICIAL_FILES = {
    "Fed3R": ("src/fed3r.py", ("fed3r_step", "calc_ridge_params", "normalize_W")),
    "FedCOF": ("federate_classes.py", None),
    "STSA": ("methods/STSA.py", ("target2onehot", "replace_fc1", "replace_fc_ours3", "compute_multiple_sum_by_class")),
    "STSA_E": ("methods/STSA.py", ("target2onehot", "replace_fc1", "replace_fc_ours3", "compute_multiple_sum_by_class")),
    "AFL": ("CIFAR-100/afl.py", None),
}


def extract_functions(path: str, names) -> dict:
    """Source of the named functions and methods of a file (top level or inside classes), without importing it."""
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in found:
            found[node.name] = ast.get_source_segment(source, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{path}: functions {missing} not found; this check was written for the commits named in "
                         f"docs/BASELINES.md")
    return found


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cpu_shim(torch) -> None:
    """Make .cuda() a no-op so official code that calls it runs on the CPU."""
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self


class Feats:
    """Stand-in backbone: the 'images' are our cached features."""
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, x, return_feats=False):
        return (x, None) if return_feats else x

    def extract_vector(self, x):
        return x

    def eval(self):
        return self

    def cuda(self, *a, **k):
        return self

    def to(self, *a, **k):
        return self


class float64_lift:
    """Run official torch code in float64: default dtype float64 and Tensor.float() -> float64 (restored after)."""
    def __init__(self, torch, on: bool):
        self.torch, self.on = torch, on

    def __enter__(self):
        if self.on:
            self.default, self.float = self.torch.get_default_dtype(), self.torch.Tensor.float
            self.torch.set_default_dtype(self.torch.float64)
            self.torch.Tensor.float = lambda t, *a, **k: t.double()
        return self

    def __exit__(self, *exc):
        if self.on:
            self.torch.set_default_dtype(self.default)
            self.torch.Tensor.float = self.float
        return False


FEATURE_DTYPE = [np.float32]


def batches(torch, Z, y, size=1024):
    Zt = torch.from_numpy(np.ascontiguousarray(Z, dtype=FEATURE_DTYPE[0]))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return [(Zt[i:i + size], yt[i:i + size]) for i in range(0, len(y), size)]


def rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b)) / max(np.max(np.abs(b)), 1e-300))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Official code vs ours on the same cached features")
    p.add_argument("--method", required=True, choices=sorted(OFFICIAL_FILES))
    p.add_argument("--official", required=True, help="folder of a clone of the method's official repository")
    p.add_argument("-cfg", "--config", required=True, help="a config of that method (configs/baselines/*.yaml)")
    p.add_argument("--run", type=int, default=0, help="which run of the config (scenario seed = seed + run)")
    p.add_argument("--task", type=int, default=None, help="only this task (default: every task)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--tolerance", type=float, default=1e-6, help="relative tolerance on the classifier")
    p.add_argument("--as_released", action="store_true",
                   help="run the official code in its released precision (partly float32) and only report")
    a, extra = p.parse_known_args(argv)

    import torch

    from flcore.servers.serverbase import Server  # noqa: F401  (system/ on the path)
    from main import resolve_args

    method = a.method
    rel_path, names = OFFICIAL_FILES[method]
    path = os.path.join(os.path.abspath(a.official), rel_path)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: give the folder of a clone of the official {method} repository")
    if a.device == "cpu":
        cpu_shim(torch)
    device = torch.device(a.device)
    lifted = not a.as_released
    FEATURE_DTYPE[0] = np.float64 if lifted else np.float32

    args, server_cls, _ = resolve_args(["-cfg", a.config, "-t", str(a.run + 1), *extra])
    if server_cls.__name__ != method:
        raise SystemExit(f"{a.config}: algorithm is {args.algorithm}, not {method}")
    if method == "Fed3R" and args.hparams["rf_dim"] > 0:
        raise SystemExit("Fed3R-RF is not in the released code; only rf_dim = 0 can be checked")

    class Snap(server_cls):
        def init_algorithm(self):
            super().init_algorithm()
            self.heads, self.order_uploads = [], []

        def aggregate(self, t, r, uploads):
            self.order_uploads.append([cid for cid, _ in uploads])
            super().aggregate(t, r, uploads)

        def on_end_task(self, t):
            super().on_end_task(t)
            self.heads.append({"classes": self.head_classes.copy(), "W": np.array(self.W, copy=True),
                               "bias": np.array(getattr(self, "bias", None), copy=True)
                               if getattr(self, "bias", None) is not None else None})

    tmp = tempfile.mkdtemp(prefix="baseline_check_")
    try:
        args.exp_dir = tmp
        ours = Snap(args, a.run)
        ours.train()
        scn, store = ours.scenario, ours.store
        hp = ours.hp
        order = np.concatenate([np.asarray(c) for c in scn.tasks])       # learning order of the classes
        label_of = {int(c): i for i, c in enumerate(order)}
        C_all, d = len(order), int(store.dim)

        def labels(rows):
            return np.array([label_of[int(c)] for c in store.labels(rows)], dtype=np.int64)

        def client_rows(t, k):
            return np.asarray(scn.train_rows[t][k], dtype=np.int64)

        tasks = range(scn.num_tasks) if a.task is None else [a.task]
        print(f"official {method} ({os.path.abspath(a.official)}) vs ours: {args.dataset} {scn.config_name}, features "
              f"{store.describe()}, d = {d}, hyperparameters {hp}, official device {device}, precision "
              + ("float64 (lifted; decides PASS / FAIL)" if lifted else "as released (report only)"))
        status = 0
        lift = float64_lift(torch, lifted)
        lift.__enter__()

        # ------------------------------------------------------------------ the official computations
        if method == "Fed3R":
            src = extract_functions(path, names)
            ns = {"torch": torch}
            for n in names:
                exec(src[n], ns)
            lam = float(hp["ridge_lambda"])
            A = lam * torch.eye(d + 1).to(device)
            b = torch.zeros((d + 1, C_all)).to(device)
            official = []
            for t in range(scn.num_tasks):
                for k in ours.order_uploads[t]:
                    rows = client_rows(t, k)
                    A, b = ns["fed3r_step"](A, b, batches(torch, store.features(rows), labels(rows)),
                                            Feats(torch), C_all, device)
                W, bias = ns["normalize_W"](*ns["calc_ridge_params"](A, b), warnings=False)
                m = sum(len(scn.tasks[u]) for u in range(t + 1))
                official.append((W.double().cpu().numpy()[:, :m], bias.double().cpu().numpy()[:m]))

        elif method == "FedCOF":
            mod = load_module(path, "official_fedcof_classes")
            official = []
            for t in range(scn.num_tasks):
                m = sum(len(scn.tasks[u]) for u in range(t + 1))
                client_data = {}
                for k in range(scn.num_clients):
                    rows = np.concatenate([client_rows(u, k) for u in range(t + 1)])
                    if len(rows) == 0:
                        continue
                    cl = mod.FedCOFClient(k, torch.from_numpy(store.features(rows).astype(FEATURE_DTYPE[0])),
                                          torch.from_numpy(labels(rows)))
                    cl.compute()
                    client_data[k] = cl.send()
                srv = mod.FedCOFServer(client_data, m, d, float(hp["gamma_shrink"]))
                srv.aggregate()
                official.append((srv.get_classifier_weights().double().cpu().numpy(), None))

        elif method in ("STSA", "STSA_E"):
            src = extract_functions(path, names)
            import torch.nn.functional as F_

            groups_queue = []

            def randperm(n, *args_, **kw):
                perm = groups_queue.pop(0)
                if len(perm) != int(n):
                    raise RuntimeError("permutation queue out of step with the official calls")
                return torch.as_tensor(perm, dtype=torch.int64)

            torch_proxy = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
            torch_proxy.randperm = randperm
            ns = {"torch": torch_proxy, "F": F_, "np": np, "nn": torch.nn}

            class Split:
                def __init__(self, dataset, idxs):
                    self.items = [dataset[i] for i in idxs]

            def loader(ds, batch_size=128, shuffle=False, num_workers=0, **kw):
                X = torch.stack([x for _, x, _ in ds.items]) if ds.items else torch.zeros(0, d)
                Y = torch.as_tensor([int(y) for _, _, y in ds.items], dtype=torch.int64)
                return [(None, X[i:i + batch_size], Y[i:i + batch_size]) for i in range(0, len(Y), batch_size)]

            ns["DataLoader"], ns["DatasetSplit"] = loader, Split
            for n in names:
                exec(src[n], ns)
            obj = types.SimpleNamespace()
            obj.args = {"M": int(hp["rp_dim"]), "nb_classes": C_all, "frac": 1.0, "num_users": scn.num_clients,
                        "local_bs": 128, "ridge": float(hp["ridge"]), "dummy": int(hp.get("subsets", 1))}
            if int(hp["rp_dim"]) != d and int(hp["rp_dim"]) in (512, 768):
                raise SystemExit("the official code skips the projection whenever M is 512 or 768; choose another M")
            if int(hp["rp_dim"]) == d and d not in (512, 768):
                raise SystemExit("M equals the feature dimension (no projection here), which the official code only "
                                 "recognises for d = 512 or 768")
            W_rand = ours.W_rand if ours.W_rand is not None else np.zeros((d, d))
            obj.W_rand = torch.from_numpy(W_rand).float().to(device)
            obj.P = torch.zeros(int(hp["rp_dim"]), C_all)
            obj.G = torch.zeros(int(hp["rp_dim"]), int(hp["rp_dim"]))
            fc = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(0, int(hp["rp_dim"])), data=None),
                                       W_rand=None, args=None)
            net = types.SimpleNamespace(fc=fc, eval=lambda: net, cuda=lambda *x, **y: net,
                                        extract_vector=lambda x: x)
            obj._network = net
            obj.compute_multiple_sum_by_class = types.MethodType(ns["compute_multiple_sum_by_class"], obj)
            fn = ns["replace_fc1"] if method == "STSA" else ns["replace_fc_ours3"]
            official = []
            for t in range(scn.num_tasks):
                m = sum(len(scn.tasks[u]) for u in range(t + 1))
                ds_rows, groups = [], {}
                for k in ours.order_uploads[t]:          # clients with data in task t, in upload order; the
                    rows = client_rows(t, k)             # official loop assumes every client holds samples
                    groups[len(groups)] = list(range(len(ds_rows), len(ds_rows) + len(rows)))
                    ds_rows.extend(rows.tolist())
                    if method == "STSA_E":               # the permutation our client drew (dummy_groups)
                        groups_queue.append(ours.clients[k].rng(t, 0).permutation(len(rows)))
                obj.args["num_users"] = len(groups)
                feats = store.features(np.asarray(ds_rows, dtype=np.int64)) if ds_rows else np.zeros((0, d))
                labs = labels(np.asarray(ds_rows, dtype=np.int64)) if ds_rows else np.zeros(0, dtype=np.int64)
                dataset = [(None, torch.from_numpy(feats[i].astype(FEATURE_DTYPE[0])), int(labs[i]))
                           for i in range(len(labs))]
                obj.user_groups = groups
                fc.weight = types.SimpleNamespace(shape=(m, int(hp["rp_dim"])), data=None)
                fn(obj, None, net, obj.args, dataset)
                official.append((fc.weight.data.double().cpu().numpy(), None))

        else:                                                                    # AFL
            mod = load_module(path, "official_afl")
            afl_args = types.SimpleNamespace(feat_size=d, num_classes=C_all, rg=float(hp["rg"]), gpu=0,
                                             num_clients=0)
            global_model = mod.LinearAnalytic(d, C_all).to(device)
            Ws, Rs, Cs, official = [], [], [], []
            for t in range(scn.num_tasks):
                m = sum(len(scn.tasks[u]) for u in range(t + 1))
                for k in ours.order_uploads[t]:
                    rows = client_rows(t, k)
                    W, R, C = mod.local_update(batches(torch, store.features(rows), labels(rows)), Feats(torch),
                                               global_model, afl_args)
                    Ws.append(W.cpu())
                    Rs.append(R)
                    Cs.append(C)
                Wt, _Rt, Ct = mod.aggregation(list(Ws), list(Rs), list(Cs), afl_args)
                if hp["clean_reg"]:
                    afl_args.num_clients = len(Ws)
                    Wt = mod.clean_regularization(Wt, Ct, afl_args)
                official.append((Wt.double().cpu().numpy()[:, :m], None))

        lift.__exit__(None, None, None)

        # ------------------------------------------------------------------ compare, task by task
        for t in tasks:
            h = ours.heads[t]
            Wo, bo = official[t]
            Wours = h["W"]
            with store.evaluation():
                test_rows = np.concatenate([np.asarray(scn.test_global_rows[u]) for u in range(t + 1)])
                Zt, yt = store.features(test_rows), store.labels(test_rows)
            pred_ours = ours_predict(ours, h, Zt)
            pred_off = official_predict(method, ours, h, Wo, bo, Zt, np.float64 if lifted else np.float32)
            dW = rel(Wo, Wours)
            db = rel(bo, h["bias"]) if bo is not None else 0.0
            ok = dW <= a.tolerance and db <= a.tolerance
            status |= 0 if (ok or not lifted) else 1
            verdict = ("PASS" if ok else "FAIL") if lifted else "released precision, not judged"
            agree = float(np.mean(pred_ours == pred_off))
            print(f"task {t}: {len(h['classes'])} classes | max rel. diff W {dW:.2e}"
                  + (f", bias {db:.2e}" if bo is not None else "")
                  + f" | accuracy official {100 * np.mean(pred_off == yt):.2f}, ours {100 * np.mean(pred_ours == yt):.2f}, "
                  f"predictions agree {100 * agree:.2f} % | {verdict}")
        return status
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ours_predict(server, h, Z):
    """Our prediction with the head kept after task t (the same rule as the server's predict())."""
    cls = h["classes"]
    name = type(server).__mro__[1].__name__
    if name == "Fed3R":
        from flcore.clients.clientfed3r import apply_rf
        return cls[np.argmax(apply_rf(Z, server.rf) @ h["W"] + h["bias"], axis=1)]
    if name == "FedCOF":
        return cls[np.argmax(Z @ h["W"].T, axis=1)]
    if name in ("STSA", "STSA_E"):
        from flcore.clients.clientstsa import project
        Hf = project(Z, server.W_rand)
        return cls[np.argmax(Hf @ (h["W"] / np.linalg.norm(h["W"], axis=1, keepdims=True)).T, axis=1)]
    return cls[np.argmax(Z @ h["W"], axis=1)]                                  # AFL


def official_predict(method, server, h, Wo, bo, Z, dtype):
    """The official prediction rule with the official classifier (float32 as the official layers, or float64)."""
    cls = h["classes"]
    Zc, Wc = Z.astype(dtype), Wo.astype(dtype)
    if method == "Fed3R":
        return cls[np.argmax(Zc @ Wc + bo.astype(dtype), axis=1)]
    if method == "FedCOF":
        return cls[np.argmax(Zc @ Wc.T, axis=1)]
    if method in ("STSA", "STSA_E"):
        from flcore.clients.clientstsa import project
        Hf = project(Z, server.W_rand).astype(dtype)
        Hn = Hf / np.maximum(np.linalg.norm(Hf, axis=1, keepdims=True), 1e-12)
        Wn = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
        return cls[np.argmax(Hn @ Wn.T, axis=1)]
    return cls[np.argmax(Zc @ Wc, axis=1)]                                    # AFL


if __name__ == "__main__":
    sys.exit(main())
