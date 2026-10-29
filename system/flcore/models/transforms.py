"""Image transforms of the image path: one deterministic evaluation transform per network, and training
augmentation whose random parameters come from a numpy generator (utils/image_utils.py), so a
training image's augmentation depends only on (key, row).

Evaluation
    pretrained ViTs   the backbone's own preprocessing, the object feature extraction uses
                      (dataset/utils/backbone_utils.py: DINOv2 resize 256 + centre crop 224; timm and
                      open_clip their own configs), so trained and frozen arms see identical test inputs
    CNNs              resize the shorter side to image_size and centre crop (skipped when the image already
                      has that size), then ImageNet mean and std

Training augmentation (model.augment)
    none        the evaluation transform
    flip        the evaluation transform, then a horizontal flip with probability 1/2
    crop_flip   CNNs: the evaluation geometry, zero padding of image_size // 8 pixels and a random crop of
                image_size (for 32 px: padding 4, the usual CIFAR recipe). ViTs: the evaluation resize, then
                a random crop of the crop size instead of the centre crop. Both then a random flip
    rrc_flip    a random resized crop of the original image (area 8% to 100%, aspect ratio 3/4 to 4/3, the
                torchvision 0.15 algorithm), resized to the input size, then a random flip
"""
from __future__ import annotations

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from utils.image_utils import flip_coin, random_crop_params, random_resized_crop_params

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def describe_preprocess(preprocess) -> dict:
    """resize, crop, interpolation, mean and std of a torchvision Compose of Resize, CenterCrop, ToTensor and
    Normalize (the evaluation transforms of the three pretrained backbones)."""
    found = {}
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, T.Resize):
            size = t.size
            if isinstance(size, (list, tuple)):
                if len(size) != 1:
                    raise RuntimeError(f"the evaluation Resize has size {size}; a shorter-side resize was expected")
                size = size[0]
            found["resize"], found["interpolation"] = int(size), t.interpolation
        elif isinstance(t, T.CenterCrop):
            h, w = (t.size, t.size) if isinstance(t.size, int) else tuple(t.size)
            if h != w:
                raise RuntimeError(f"the evaluation CenterCrop is {h} x {w}; a square crop was expected")
            found["crop"] = int(h)
        elif isinstance(t, T.Normalize):
            found["mean"] = [float(v) for v in t.mean]
            found["std"] = [float(v) for v in t.std]
    missing = [k for k in ("resize", "crop", "interpolation", "mean", "std") if k not in found]
    if missing:
        raise RuntimeError(f"cannot read {missing} from the backbone's preprocessing {preprocess!r}")
    return found


class ImagePipeline:
    """eval(img) and train(img, rng) for one network; images are RGB PIL images."""

    def __init__(self, kind: str, size: int, augment: str, mean, std, interpolation,
                 eval_transform=None, resize: int | None = None):
        if kind not in ("cnn", "vit"):
            raise ValueError(kind)
        self.kind, self.size, self.augment = kind, int(size), augment
        self.mean, self.std = [float(v) for v in mean], [float(v) for v in std]
        self.interpolation = interpolation
        self.resize = resize                          # ViTs: the shorter side of the evaluation resize
        self.pad = self.size // 8                     # CNNs: crop_flip padding
        self._eval = eval_transform

    @classmethod
    def for_cnn(cls, size: int, augment: str) -> "ImagePipeline":
        return cls("cnn", size, augment, IMAGENET_MEAN, IMAGENET_STD, InterpolationMode.BILINEAR)

    @classmethod
    def for_vit(cls, preprocess, augment: str) -> "ImagePipeline":
        d = describe_preprocess(preprocess)
        return cls("vit", d["crop"], augment, d["mean"], d["std"], d["interpolation"], eval_transform=preprocess,
                   resize=d["resize"])

    def describe(self) -> dict:
        return {"kind": self.kind, "size": self.size, "augment": self.augment, "mean": self.mean, "std": self.std,
                "interpolation": str(self.interpolation), "resize": self.resize,
                "pad": self.pad if self.kind == "cnn" else None,
                "eval": repr(self._eval) if self._eval is not None else "resize + centre crop, normalize"}

    # ---- pieces
    def _square(self, img):
        """CNNs: shorter side to size, centre crop (nothing to do when the image is already size x size)."""
        if img.size == (self.size, self.size):
            return img
        img = F.resize(img, self.size, interpolation=self.interpolation, antialias=True)
        return F.center_crop(img, [self.size, self.size])

    def _to_tensor(self, img) -> torch.Tensor:
        return F.normalize(F.to_tensor(img), self.mean, self.std)

    # ---- the two transforms
    def eval(self, img) -> torch.Tensor:
        if self._eval is not None:
            return self._eval(img)
        return self._to_tensor(self._square(img))

    def train(self, img, rng) -> torch.Tensor:
        a = self.augment
        if a == "none":
            return self.eval(img)
        if a == "flip":
            x = self.eval(img)
            return torch.flip(x, dims=[-1]) if flip_coin(rng) else x
        if a == "rrc_flip":
            top, left, h, w = random_resized_crop_params(img.height, img.width, rng)
            img = F.resized_crop(img, top, left, h, w, [self.size, self.size], interpolation=self.interpolation,
                                 antialias=True)
        elif a == "crop_flip":
            if self.kind == "cnn":
                img = F.pad(self._square(img), self.pad, fill=0)
            else:
                img = F.resize(img, self.resize, interpolation=self.interpolation, antialias=True)
            top, left = random_crop_params(img.height, img.width, self.size, rng)
            img = F.crop(img, top, left, self.size, self.size)
        else:
            raise ValueError(f"unknown augmentation {a!r}")
        if flip_coin(rng):
            img = F.hflip(img)
        return self._to_tensor(img)
