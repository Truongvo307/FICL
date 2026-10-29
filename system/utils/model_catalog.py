"""The networks image algorithms can train, described without importing torch.

Settings are checked (config_utils.finalize_config) before torch is imported, so the facts the
checks need live here. The networks themselves are built in flcore/models/.

Three families of federated class-incremental methods, and how a run selects them (model section):
    frozen pretrained   an algorithm on cached features (features section, INPUT = "features"), or an
                        image algorithm with a pretrained ViT and trainable: head
    pretrained, trained a pretrained ViT with trainable: last_k | lora | prompt | full
    from scratch        a CNN without weights (pretrained: false), trained in full
"""
from __future__ import annotations

from dataclasses import dataclass

CNN, VIT = "cnn", "vit"
TRAINABLE_MODES = ("full", "head", "last_k", "lora", "prompt")
AUGMENTATIONS = ("none", "flip", "crop_flip", "rrc_flip")
LORA_TARGETS = ("q", "k", "v", "proj", "fc1", "fc2")
DEFAULT_AUGMENT = {CNN: "crop_flip", VIT: "rrc_flip"}
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_TARGETS = ("q", "v")
DEFAULT_PROMPT_LENGTH = 10
DEFAULT_NUM_WORKERS = 4
DEFAULT_EVAL_BATCH_SIZE = 256
MIN_IMAGE_SIZE = 16

TRAINABLE_TEXT = {
    "full": "every parameter of the network and the head",
    "head": "the linear head only (the backbone stays frozen)",
    "last_k": "the last k transformer blocks, the final norm and the head",
    "lora": "LoRA factors on chosen weight matrices of every block, and the head",
    "prompt": "learnable prompt tokens (VPT), and the head",
}
AUGMENT_TEXT = {
    "none": "the evaluation transform only (no randomness)",
    "flip": "the evaluation transform, then a random horizontal flip",
    "crop_flip": "CNNs: 1/8-size zero padding and a random crop back to the input size; ViTs: the evaluation "
                 "resize, then a random crop instead of the centre crop; both then a random horizontal flip",
    "rrc_flip": "random resized crop (area 8% to 100%, aspect 3/4 to 4/3), then a random horizontal flip",
}


@dataclass(frozen=True)
class NetworkInfo:
    name: str
    kind: str                     # CNN or VIT
    weights: str | None           # where pretrained weights come from (None: trained from scratch only)
    image_size: int               # default input size in pixels
    resizable: bool               # False: the input size is fixed by the pretrained position embeddings
    num_blocks: int | None        # transformer blocks (ViTs)
    feature_dim: int              # size of the vector the head reads
    description: str
    smoke: bool = False           # pipeline checks only: runs get the SMOKE_TEST flag

    @property
    def scratch(self) -> bool:
        """Can be trained from a random initialisation here (the CNNs; the ViTs come with their weights)."""
        return self.kind == CNN


NETWORKS = {
    "tiny_cnn": NetworkInfo(
        "tiny_cnn", CNN, None, 32, True, None, 32,
        "two 3x3 conv layers with batch norm (32 features); for pipeline checks only, its runs carry SMOKE_TEST",
        smoke=True),
    "resnet18_cifar": NetworkInfo(
        "resnet18_cifar", CNN, None, 32, True, None, 512,
        "ResNet-18 with the CIFAR stem (3x3 conv, stride 1, no max pool), 512 features"),
    "resnet32": NetworkInfo(
        "resnet32", CNN, None, 32, True, None, 64,
        "CIFAR ResNet-32: 3 stages of 5 basic blocks (16, 32, 64 channels), parameter-free shortcuts, 64 features"),
    "vit_b16_in21k": NetworkInfo(
        "vit_b16_in21k", VIT, "timm vit_base_patch16_224.augreg_in21k", 224, False, 12, 768,
        "ViT-B/16 pretrained on ImageNet-21k (timm); head on the final-norm CLS token"),
    "dinov2_vitb14": NetworkInfo(
        "dinov2_vitb14", VIT, "torch.hub facebookresearch/dinov2 dinov2_vitb14", 224, False, 12, 768,
        "DINOv2 ViT-B/14 (torch.hub); head on the final-norm CLS token"),
    "clip_vitb16": NetworkInfo(
        "clip_vitb16", VIT, "open_clip ViT-B-16 pretrained=openai", 224, False, 12, 768,
        "CLIP ViT-B/16 image tower (open_clip); head on the final-norm CLS token before the projection"),
}


def describe_networks() -> list[str]:
    """One line per network, for help texts."""
    lines = []
    for n in NETWORKS.values():
        weights = (f"pretrained only: {n.weights}" if n.weights else "no pretrained weights (trained from scratch)")
        lines.append(f"{n.name:15s} {n.kind}  {n.image_size} px  {weights}. {n.description}")
    return lines
