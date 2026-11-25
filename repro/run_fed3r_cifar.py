#!/usr/bin/env python
"""Fed3R (ICML 2024) CIFAR-100 setting of the paper's Table 6, then a dump for run_ours.py.

    python repro/run_fed3r_cifar.py --data /data/cifar --seed 0
    python repro/run_ours.py repro/work/dumps/fed3r_cifar100_mobilenetv2_alpha0_K100_seed0.npz   (FICL environment)

Paper setting (Appendix C, Table 4, Table 6): CIFAR-100, 100 clients, the most heterogeneous split (alpha = 0,
Hsu et al. 2019: every client holds the images of one class), MobileNetV2 pre-trained on ImageNet, images
scaled to 224 x 224, lambda = 0.01; Fed3R 63.2, Fed3R-RF 5k 66.2, Fed3R-RF 10k 67.5 [R].

The official repository has no CIFAR-100 pipeline (its loaders read Landmarks and iNaturalist TFRecords), so
this script RECONSTRUCTS the data side [H] and uses the repository for everything else:
    official   the MobileNetV2 of src/models/mobilenetv2.py with model_ckpts/pt_mobilenetv2_100-4065ukrp
               (GroupNorm, ImageNet pre-training), its feature output (return_features=True);
               fed3r_step, calc_ridge_params, normalize_W of src/fed3r.py; with TensorFlow installed, the
               image functions of src/datasets/tf_utils.py and the Landmarks map functions (224 px):
               train  random_crop_and_resize(224, crop_padding 32) + random flip, /255
               test   center_crop_and_resize(224, crop_padding 32), /255
               then (x - 0.5) / 0.5 as datasets/landmarks.py
    ours [H]   the CIFAR-100 reading (torchvision), the alpha = 0 split, one augmented view per training image;
               without TensorFlow a torch stand-in of the two crop functions (bicubic kernels differ slightly)
Fed3R is exact in its statistics, so neither the client split nor the client order changes its result;
the preprocessing does, so the paper comparison is indicative only.
Needs: torch, torchvision, timm, torchinfo (imported by the repository's mobilenetv2.py); tensorflow optional.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DUMP_FORMAT, accuracy, save_dump  # noqa: E402
from fetch_official import REPOS  # noqa: E402
from official_common import (FeatureBackbone, batches, check_modules, commit_of, extract_functions,  # noqa: E402
                             module_available, official_fed3r_functions, require_commit)

CKPT = os.path.join("model_ckpts", "pt_mobilenetv2_100-4065ukrp", "pretrained_imagenet.pth")
CROP_PADDING, SIZE = 32, 224


def tf_preprocess(fed3r):
    """The official tf_utils functions, executed from their source with TensorFlow."""
    import tensorflow as tf
    names = ("center_crop_and_resize", "_at_least_x_are_equal", "distorted_bounding_box_crop",
             "random_crop_and_resize")
    src = extract_functions(os.path.join(fed3r, "src", "datasets", "tf_utils.py"), names)
    ns = {"tf": tf}
    for n in names:
        exec(src[n], ns)

    def train(img):                    # datasets/landmarks.py train_map_fn (image size 224)
        x = ns["random_crop_and_resize"](tf.constant(img), SIZE, CROP_PADDING) / 255.0
        return tf.image.random_flip_left_right(x).numpy()

    def test(img):                     # datasets/landmarks.py test_map_fn
        return (ns["center_crop_and_resize"](tf.constant(img), SIZE, CROP_PADDING) / 255.0).numpy()
    return train, test, f"tensorflow {tf.__version__}: official tf_utils functions"


def torch_preprocess(seed):
    """Stand-in without TensorFlow: same geometry, torch bicubic (a = -0.75 instead of TF's -0.5)."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    crop = int(SIZE / (SIZE + CROP_PADDING) * 32)

    def resize(x):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(2, 0, 1)[None]
        return F.interpolate(t, size=(SIZE, SIZE), mode="bicubic", align_corners=False)[0].permute(1, 2, 0).numpy()

    def test(img):
        o = (32 - crop + 1) // 2
        return resize(img[o:o + crop, o:o + crop]) / 255.0

    def train(img):
        for _ in range(10):            # distorted_bounding_box_crop: area 0.08-1, aspect 3/4-4/3, 10 attempts
            area = 32 * 32 * float(torch.empty(1).uniform_(0.08, 1.0, generator=g))
            ar = float(torch.exp(torch.empty(1).uniform_(np.log(3 / 4), np.log(4 / 3), generator=g)))
            w, h = int(round((area * ar) ** 0.5)), int(round((area / ar) ** 0.5))
            if 0 < w <= 32 and 0 < h <= 32 and (w, h) != (32, 32):
                y0 = int(torch.randint(0, 32 - h + 1, (1,), generator=g))
                x0 = int(torch.randint(0, 32 - w + 1, (1,), generator=g))
                x = resize(img[y0:y0 + h, x0:x0 + w]) / 255.0
                break
        else:
            x = test(img)
        if float(torch.rand(1, generator=g)) < 0.5:
            x = x[:, ::-1]
        return x
    return train, test, "torch stand-in of tf_utils crop/resize [H]"


def load_backbone(fed3r, device):
    import torch
    spec = importlib.util.spec_from_file_location("fed3r_mobilenetv2", os.path.join(fed3r, "src", "models",
                                                                                   "mobilenetv2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.mobilenetv2(2028, return_features=True, pretrained_path=os.path.join(fed3r, CKPT))
    return model.to(device).eval(), torch


def features(model, torch, images, fn, device, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = np.stack([fn(im) for im in images[i:i + bs]]).astype(np.float32)
            x = (x - 0.5) / 0.5                                   # landmarks.py MEAN = STD = 0.5
            t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
            out.append(model(t).float().cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--official", default=os.path.join(HERE, "official_src", "Fed3R"))
    p.add_argument("--data", required=True, help="folder for torchvision's CIFAR-100")
    p.add_argument("--seed", type=int, default=0, help="augmentation seed")
    p.add_argument("--lamb", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preprocess", choices=["auto", "tensorflow", "torch"], default="auto")
    a = p.parse_args(argv)
    a.official, a.data = os.path.abspath(a.official), os.path.abspath(a.data)
    require_commit(a.official, REPOS["Fed3R"][1], "Fed3R")
    check_modules(["torch", "torchvision", "timm", "torchinfo"])
    import torch
    import torchvision
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        a.device = "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    use_tf = a.preprocess == "tensorflow" or (a.preprocess == "auto" and module_available("tensorflow"))
    if use_tf:
        import tensorflow as tf
        tf.random.set_seed(a.seed)
        train_fn, test_fn, how = tf_preprocess(a.official)
    else:
        train_fn, test_fn, how = torch_preprocess(a.seed)
    print(f"preprocessing: {how}")

    tr = torchvision.datasets.CIFAR100(a.data, train=True, download=True)
    te = torchvision.datasets.CIFAR100(a.data, train=False, download=True)
    ytr, yte = np.asarray(tr.targets, dtype=np.int64), np.asarray(te.targets, dtype=np.int64)
    model, torch = load_backbone(a.official, a.device)

    C = 100
    order = np.argsort(ytr, kind="stable")                        # alpha = 0: client c holds class c
    Xtr = features(model, torch, tr.data[order], train_fn, a.device)
    ytr, ktr = ytr[order], ytr[order].copy()
    Xte = features(model, torch, te.data, test_fn, a.device)
    d = Xtr.shape[1]

    f3 = official_fed3r_functions(a.official, torch)
    A = a.lamb * torch.eye(d + 1)
    b = torch.zeros((d + 1, C))
    for k in range(C):
        rows = ktr == k
        A, b = f3["fed3r_step"](A, b, batches(torch, Xtr[rows], ytr[rows]), FeatureBackbone(), C, "cpu")
    W, bias = f3["normalize_W"](*f3["calc_ridge_params"](A, b), warnings=False)
    pred = torch.max(torch.from_numpy(Xte) @ W + bias, dim=1)[1].numpy().astype(np.int64)
    acc = accuracy(pred, yte)
    print(f"official Fed3R functions on these features: {acc:.2f} %  (paper Table 6: 63.2 [R])")

    meta = {"format": DUMP_FORMAT,
            "source": {"repo": "Erosinho13/Fed3R", "commit": commit_of(a.official),
                       "entry": "src/fed3r.py functions + src/models/mobilenetv2.py + model_ckpts (official); "
                                "CIFAR-100 reading and alpha = 0 split reconstructed [H]",
                       "preprocessing": how, "notes": "one augmented view per training image, seed "
                                                      f"{a.seed}; clients in class order"},
            "setting": {"dataset": "CIFAR-100", "backbone": "Fed3R MobileNetV2-GN (pretrained_imagenet.pth)",
                        "num_classes": C, "num_clients": C, "partition_label": "single class per client (alpha 0)",
                        "alpha": 0.0, "tasks": [list(range(C))], "seed": a.seed},
            "official": {"Fed3R": {"acc": [acc], "hparams": {"lambda_": a.lamb},
                                   "precision": "float32 (as released)"}},
            "ours": [{"algorithm": "Fed3R", "label": "Fed3R", "compare_to": "Fed3R",
                      "hparams": {"ridge_lambda": a.lamb, "rf_dim": 0}},
                     {"algorithm": "Fed3R", "label": "Fed3R-RF5k", "compare_to": None,
                      "hparams": {"ridge_lambda": a.lamb, "rf_dim": 5000, "rf_sigma": 1000.0}},
                     {"algorithm": "Fed3R", "label": "Fed3R-RF10k", "compare_to": None,
                      "hparams": {"ridge_lambda": a.lamb, "rf_dim": 10000, "rf_sigma": 1000.0}}],
            "paper": ["fed3r_icml24_t6_fed3r", "fed3r_icml24_t6_fed3r_rf5k", "fed3r_icml24_t6_fed3r_rf10k"]}
    arrays = {"train_x": Xtr.astype(np.float32), "train_y": ytr, "train_task": np.zeros(len(ytr), np.int64),
              "train_client": ktr, "test_x": Xte.astype(np.float32), "test_y": yte,
              "evalrows__0": np.arange(len(yte), dtype=np.int64), "pred__Fed3R__0": pred,
              "W__Fed3R__0": W.double().numpy().T, "b__Fed3R__0": bias.double().numpy(),
              "Wcls__Fed3R__0": np.arange(C, dtype=np.int64)}
    out = os.path.join(HERE, "work", "dumps", f"fed3r_cifar100_mobilenetv2_alpha0_K100_seed{a.seed}.npz")
    save_dump(out, meta, arrays)
    print(f"dump     {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
