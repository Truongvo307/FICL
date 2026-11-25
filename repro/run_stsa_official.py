#!/usr/bin/env python
"""STSA / STSA-E (WWW 2026) with the official STSA repository, then dumps for run_ours.py.

    python repro/run_stsa_official.py                                  # STSA and STSA-E, adapted and frozen
    python repro/run_stsa_official.py --variants STSA --stages frozen
    python repro/run_ours.py repro/work/dumps/stsa_cifar224_b0.5_K5_T10_adapted_STSA.npz   (FICL environment)

Paper setting (Table 2, CIFAR-100, ViT-B/16 IN-21K with adapters): the README command
    main.py --method STSA --tasks 10 --dataset cifar224 --beta 0.5 --com_round 10 --local_ep 2 --num_users 5
            --net vit_adapter --M 1250
(main.sh does not pass its seed argument, so the seed is main.py's default 0). Two stages:

    adapted   exactly that command: first-stage FedAvg of the adapters on task 1, then frozen  -> paper numbers
    frozen    the same with --com_round 0: main.py then loads a checkpoint instead of training; it is given
              the untouched initial network (pretrained ViT-B/16 IN-21K; adapters with zero up-projection, so they
              add nothing), i.e. STSA without its first stage, as FICL runs it (docs/BASELINES.md D1)

STSA-E: the README says to set args["type"] = 1 in main.py; here the same value is set on the learner's
args right after main.py creates it (main.py itself stays unchanged). K_D is main.py's args["dummy"] = 5.

main.py runs unchanged in-process (runpy), one process per (variant, stage). Recording wrappers, each calling
the original and returning its result untouched:
    STSA.replace_fc1 / replace_fc_ours3   before the original: every client's embeddings of the current task
                                          (model.extract_vector, eval mode, the same dataset objects), the
                                          client split, W_rand; after it: the official classifier.
                                          STSA-E: the order of every client's loader and each randperm of
                                          compute_multiple_sum_by_class -> the official dummy group of each sample
    STSA._eval_cnn                         the official predictions on the test set of the seen classes
The extra passes use their own torch.Generator, so the official random stream is not changed.
Needs: torch, torchvision, timm, easydict, kornia, sklearn, pandas, tensorboard (STSA's imports).
"""
from __future__ import annotations

import argparse
import copy
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

PAPER = {"STSA": ["stsa_www26_t2_b05_stsa_final", "stsa_www26_t2_b05_stsa_avg"],
         "STSA_E": ["stsa_www26_t2_b05_stsae_final", "stsa_www26_t2_b05_stsae_avg"]}


def official_argv(a) -> list:
    return ["main.py", "--method", "STSA", "--tasks", str(a.tasks), "--dataset", "cifar224", "--beta", str(a.beta),
            "--com_round", str(a.com_round if a.stage == "adapted" else 0), "--local_ep", str(a.local_ep),
            "--num_users", str(a.num_users), "--net", "vit_adapter", "--M", str(a.M), "--gpu", str(a.gpu)] + \
           (["--file", "<initial network, see run_stsa_official.py>"] if a.stage == "frozen" else [])


def hook(a) -> int:
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, a.official)
    os.makedirs(a.work, exist_ok=True)
    os.chdir(a.work)                       # main.py writes logs/ and checkpoints into the working folder
    shims = shims_applied()
    import methods.STSA as S               # official modules, unchanged
    from utils.data_manager import DatasetSplit

    rec = {"tasks": [], "evals": [], "W_rand": None, "adapter_zero": None}
    gen = torch.Generator()                # our extra passes never touch the global torch RNG

    def embed(model, dataset, idxs=None):
        ds = dataset if idxs is None else DatasetSplit(dataset, idxs)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, generator=gen)
        pos, xs, ys = [], [], []
        with torch.no_grad():
            for i, x, y in loader:
                xs.append(model.extract_vector(x.cuda()).float().cpu())
                pos.append(i.cpu())
                ys.append(y.cpu())
        return torch.cat(pos).numpy(), torch.cat(xs).numpy(), torch.cat(ys).numpy().astype(np.int64)

    orig_init = S.STSA.__init__

    def init(self, args):
        orig_init(self, args)
        if a.variant == "STSA_E":
            self.args["type"] = 1          # README: 'For STSA-E, set args["type"] = 1 in main.py'
    S.STSA.__init__ = init

    if a.stage == "frozen":
        orig_fl = S.STSA._fl_train

        def fl_train(self, train_dataset, test_loader):
            initial = copy.deepcopy(self._network.state_dict())
            ups = [v for k, v in initial.items() if "up_proj" in k]
            rec["adapter_zero"] = bool(ups) and all(float(v.abs().max()) == 0.0 for v in ups)
            real_load = torch.load
            torch.load = lambda *args_, **kw: copy.deepcopy(initial)   # the checkpoint main.py loads (com_round 0)
            try:
                return orig_fl(self, train_dataset, test_loader)
            finally:
                torch.load = real_load
        S.STSA._fl_train = fl_train

    def wrap_replace(name):
        orig = getattr(S.STSA, name)

        def replace(self, trainloader, model, args, train_dataset):
            model.eval()
            K = self.args["num_users"]
            clients = []
            for k in range(K):
                pos, x, y = embed(model, train_dataset, self.user_groups[k])
                clients.append({"pos": pos, "x": x, "y": y})
            if rec["W_rand"] is None:
                rec["W_rand"] = self.W_rand.detach().float().cpu().numpy()
                rec["args"] = {"M": int(self.args["M"]), "ridge": float(self.args["ridge"]),
                               "dummy": int(self.args["dummy"])}
            perms, orders = [], []
            if name == "replace_fc_ours3":
                real_randperm, RealLoader = torch.randperm, S.DataLoader

                def randperm(n, *args_, **kw):
                    out = real_randperm(n, *args_, **kw)
                    if "generator" not in kw:          # compute_multiple_sum_by_class (samplers pass a generator)
                        perms.append(out.clone())
                    return out

                class RecLoader(RealLoader):
                    def __iter__(self_):
                        order = []
                        orders.append(order)
                        for batch in super().__iter__():
                            order.append(batch[0].clone())
                            yield batch
                torch.randperm, S.DataLoader = randperm, RecLoader
                try:
                    out = orig(self, trainloader, model, args, train_dataset)
                finally:
                    torch.randperm, S.DataLoader = real_randperm, RealLoader
            else:
                out = orig(self, trainloader, model, args, train_dataset)
            W = self._network.fc.weight.detach().float().cpu().numpy().copy()
            groups = None
            if name == "replace_fc_ours3":
                if len(perms) != K or len(orders) != K:
                    raise RuntimeError(f"recorded {len(perms)} group permutations and {len(orders)} loaders for "
                                       f"{K} clients")
                KD = int(self.args["dummy"])
                groups = []
                for k in range(K):
                    order = torch.cat(orders[k]).numpy()
                    perm = perms[k].numpy()
                    n, step = len(order), len(order) // KD
                    gid = np.full(n, -1, dtype=np.int64)
                    for g in range(KD):
                        sl = perm[g * step:((g + 1) * step if g < KD - 1 else n)]
                        gid[sl] = g
                    by_pos = dict(zip(order.tolist(), gid.tolist(), strict=True))
                    groups.append(np.array([by_pos[int(p)] for p in clients[k]["pos"]], dtype=np.int64))
                # the recorded groups must reproduce the official group statistics
                Wr = torch.from_numpy(rec["W_rand"]).double()
                sums = [s.double().cpu() for s in self.mu_list]               # one [nb_classes, M] per group
                i = 0
                worst = 0.0
                for k in range(K):
                    H = torch.relu(torch.from_numpy(clients[k]["x"]).double() @ Wr)
                    for g in range(KD):
                        sel = groups[k] == g
                        mine = torch.zeros_like(sums[i])
                        mine.index_add_(0, torch.from_numpy(clients[k]["y"][sel]), H[sel])
                        worst = max(worst, float((mine - sums[i]).abs().max() / max(float(sums[i].abs().max()), 1e-12)))
                        i += 1
                rec.setdefault("group_check", []).append(worst)
                if worst > 1e-3:
                    raise RuntimeError(f"STSA-E group reconstruction differs from the official sums ({worst:.2e})")
            rec["tasks"].append({"clients": clients, "W": W, "groups": groups,
                                 "classes": np.arange(self._known_classes, self._total_classes)})
            return out
        setattr(S.STSA, name, replace)

    wrap_replace("replace_fc1")
    wrap_replace("replace_fc_ours3")

    orig_eval = S.STSA._eval_cnn

    def eval_cnn(self, loader):
        y_pred, y_true = orig_eval(self, loader)
        rec["evals"].append({"pred": y_pred[:, 0].astype(np.int64), "y": y_true.astype(np.int64),
                             "loader": loader, "network": self._network})
        return y_pred, y_true
    S.STSA._eval_cnn = eval_cnn

    sys.argv = official_argv(a)
    if a.stage == "frozen":
        sys.argv[sys.argv.index("--file") + 1] = "initial_network"
    runpy.run_path(os.path.join(a.official, "main.py"), run_name="__main__")

    T = len(rec["tasks"])
    evals = rec["evals"][-T:] if len(rec["evals"]) >= T else None
    if T != a.tasks or evals is None:
        raise RuntimeError(f"recorded {T} tasks and {len(rec['evals'])} evaluations, expected {a.tasks}")
    # test embeddings once, after the last task (the backbone is fixed after the first stage)
    last = evals[-1]
    _, Xte, yte = embed(last["network"], last["loader"].dataset)
    if not np.array_equal(yte, last["y"]):
        raise RuntimeError("test labels of the extra pass differ from the official evaluation order")
    arrays = {"test_x": Xte.astype(np.float32), "test_y": yte, "W_rand": rec["W_rand"]}
    official_acc, notes = [], []
    Wr = rec["W_rand"].astype(np.float32)
    for t, ev in enumerate(evals):
        n_t = len(ev["y"])
        if not np.array_equal(ev["y"], yte[:n_t]):
            raise RuntimeError(f"task {t}: the evaluated test set is not a prefix of the last one")
        arrays[f"evalrows__{t}"] = np.arange(n_t, dtype=np.int64)
        key = a.variant
        arrays[f"pred__{key}__{t}"] = ev["pred"]
        W = rec["tasks"][t]["W"]
        seen = np.arange(W.shape[0], dtype=np.int64)
        arrays[f"W__{key}__{t}"], arrays[f"Wcls__{key}__{t}"] = W.astype(np.float64), seen
        official_acc.append(accuracy(ev["pred"], ev["y"]))
        # the official prediction must follow from the recorded embeddings and classifier
        H = np.maximum(Xte[:n_t] @ Wr, 0)
        H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        agree = float(np.mean(np.argmax(H @ Wn.T, axis=1) == ev["pred"]))
        if agree < 0.999:
            raise RuntimeError(f"task {t}: recorded embeddings reproduce only {100 * agree:.2f} % of the official "
                               "predictions")
        notes.append(f"task {t}: recorded inputs reproduce {100 * agree:.2f} % of the official predictions")
    xs, ys, ts, ks, gs = [], [], [], [], {t: [] for t in range(T)}
    for t, task in enumerate(rec["tasks"]):
        for k, c in enumerate(task["clients"]):
            xs.append(c["x"])
            ys.append(c["y"])
            ts.append(np.full(len(c["y"]), t, np.int64))
            ks.append(np.full(len(c["y"]), k, np.int64))
            for u in range(T):
                gs[u].append(task["groups"][k] if (u == t and task["groups"] is not None)
                             else np.full(len(c["y"]), -1, np.int64))
    arrays.update(train_x=np.concatenate(xs).astype(np.float32), train_y=np.concatenate(ys),
                  train_task=np.concatenate(ts), train_client=np.concatenate(ks))
    if a.variant == "STSA_E":
        for t in range(T):
            arrays[f"group__{t}"] = np.concatenate(gs[t])
    tasks = [sorted(np.unique(arrays["train_y"][arrays["train_task"] == t]).tolist()) for t in range(T)]
    oa = rec["args"]                       # the values the official learner used (main.py: ridge 1e6, dummy 5)
    hp = {"rp_dim": oa["M"], "ridge": oa["ridge"]}
    ours = [{"algorithm": a.variant, "label": a.variant, "compare_to": a.variant, "compare_mode": "replay",
             "hparams": dict(hp, **({"subsets": oa["dummy"]} if a.variant == "STSA_E" else {}))}]
    if a.variant == "STSA_E":
        ours.append({"algorithm": "STSA_E", "label": "STSA_E-own-groups", "compare_to": "STSA_E",
                     "compare_mode": "own_groups", "hparams": dict(hp, subsets=oa["dummy"])})
    meta = {"format": DUMP_FORMAT,
            "source": {"repo": "Yuqin-G/STSA", "commit": commit_of(a.official), "entry": "main.py (unchanged)",
                       "argv": sys.argv, "shims": shims, "stage": a.stage, "variant": a.variant,
                       "type_set_on_learner": a.variant == "STSA_E",
                       "adapters_zero_at_start": rec["adapter_zero"], "group_check_max_rel": rec.get("group_check"),
                       "notes": notes},
            "setting": {"dataset": "CIFAR-100 (cifar224)", "backbone": "ViT-B/16 IN-21K + AdaptFormer adapters"
                        + (" trained in the first stage" if a.stage == "adapted" else " (untrained: frozen backbone)"),
                        "num_classes": 100, "num_clients": a.num_users, "partition_label": "dir", "alpha": a.beta,
                        "tasks": tasks, "seed": 0},
            "official": {a.variant: {"acc": official_acc, "hparams": oa,
                                     "precision": "as released (statistics float64, W and logits float32)"}},
            "ours": ours, "paper": PAPER[a.variant] if a.stage == "adapted" else []}
    save_dump(a.out, meta, arrays)
    print(f"dump     {a.out}\n         official {a.variant} ({a.stage}) per task: "
          + " ".join(f"{x:.2f}" for x in official_acc))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--official", default=os.path.join(HERE, "official_src", "STSA"))
    p.add_argument("--work", default=os.path.join(HERE, "work", "stsa"))
    p.add_argument("--variants", nargs="+", choices=["STSA", "STSA_E"], default=["STSA", "STSA_E"])
    p.add_argument("--stages", nargs="+", choices=["adapted", "frozen"], default=["adapted", "frozen"])
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--num_users", type=int, default=5)
    p.add_argument("--com_round", type=int, default=10)
    p.add_argument("--local_ep", type=int, default=2)
    p.add_argument("--M", type=int, default=1250)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", help=argparse.SUPPRESS)
    p.add_argument("--stage", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.official, a.work = os.path.abspath(a.official), os.path.abspath(a.work)
    if a.hook:
        return hook(a)
    require_commit(a.official, REPOS["STSA"][1], "STSA")
    check_modules(["torch", "torchvision", "timm", "easydict", "kornia", "sklearn", "pandas", "tensorboard"])
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    for stage in a.stages:
        for variant in a.variants:
            out = os.path.abspath(os.path.join(HERE, "work", "dumps", f"stsa_cifar224_b{a.beta}_K{a.num_users}_"
                                                                       f"T{a.tasks}_{stage}_{variant}.npz"))
            if os.path.exists(out):
                print(f"ok       {out} exists")
                continue
            cmd = [a.python, os.path.abspath(__file__), "--hook", "--variant", variant, "--stage", stage, "--out", out,
                   "--official", a.official, "--work", os.path.join(a.work, f"{stage}_{variant}"),
                   "--tasks", str(a.tasks), "--beta", str(a.beta), "--num_users", str(a.num_users),
                   "--com_round", str(a.com_round), "--local_ep", str(a.local_ep), "--M", str(a.M),
                   "--gpu", str(a.gpu)]
            log = os.path.join(a.work, "logs", f"{stage}_{variant}.log")
            print("run      " + " ".join(cmd))
            with open(log, "w") as f:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        env=subprocess_env(), bufsize=1)
                for line in proc.stdout:
                    f.write(line)
                    if re.search(r"CNN top1 curve|Learning on|dump|official|Error|error|WARNING|Traceback", line):
                        sys.stdout.write("  | " + line)
                if proc.wait() != 0:
                    raise SystemExit(f"STSA {stage} {variant} failed; see {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
