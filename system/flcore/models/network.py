"""build_network(): the network an image algorithm trains, what is trainable, and its messages.

    bundle = build_network(settings, num_classes, device, seed)
    bundle.net               Network: encoder (CNN or PretrainedViT) + linear head over all classes
    bundle.pipeline          the image transforms (flcore/models/transforms.py)
    bundle.get_state(seen)   the trainable tensors (and batch-norm statistics) as CPU tensors; the head's rows
                             only for the classes in `seen` (rows of unseen classes never train)
    bundle.load_state(state, seen)
    average_states([(state, weight), ...])   the FedAvg mean, accumulated in float64

One network object lives on the device; clients are simulated one after another and load their
state into it (HtFLlib and PFLlib keep a copy per client instead; for a ViT-B that is 344 MB each).

What is trainable (settings.trainable)
    full     every parameter (CNNs from scratch; ViTs except parameters the pooled CLS never uses)
    head     the head only
    last_k   the last k blocks, the final norm and the head
    lora     the LoRA factors and the head
    prompt   the prompt tokens and the head
Batch-norm running statistics of trained modules travel with the state (FedAvg averages them; fed-mammoth
does not, see the implementation plan). Integer buffers (num_batches_tracked) stay where they are.

All random initial values (CNN weights, head, LoRA A, prompts) are drawn on the CPU from torch's
generator seeded with `seed` inside torch.random.fork_rng, so they do not depend on the device and
leave the global generator as it was.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from flcore.models.cnn import CNNS
from flcore.models.transforms import ImagePipeline
from utils.model_catalog import CNN, NETWORKS

HEAD_PARAMS = ("head.weight", "head.bias")


class Network(nn.Module):
    """encoder (images -> features [B, D]) followed by a linear head over all classes of the dataset."""

    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def configure_determinism() -> None:
    """Deterministic kernels where torch has them (a warning where it has none), cuDNN autotuning off, no TF32,
    and the math kernel of scaled_dot_product_attention (timm's fused attention and nn.MultiheadAttention
    call it): the flash and memory-efficient kernels are faster, but repeats of a run must agree exactly.
    cuBLAS also needs CUBLAS_WORKSPACE_CONFIG before CUDA starts; main.py sets it."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy as logsumexp - gather. torch 2.0 lists NLLLoss (behind F.cross_entropy) among the
    CUDA operations without a deterministic implementation; logsumexp and gather run deterministically."""
    return (torch.logsumexp(logits, dim=1) - logits.gather(1, y.unsqueeze(1)).squeeze(1)).mean()


def fork_devices(device: str) -> list:
    """Devices whose generator state torch.random.fork_rng saves and restores."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return [dev.index if dev.index is not None else torch.cuda.current_device()]
    return []


@dataclass
class Bundle:
    net: Network
    pipeline: ImagePipeline
    device: str
    trainable: list                       # names of trainable parameters, in named_parameters() order
    buffers: list                         # names of float buffers that travel with the state
    info: dict = field(default_factory=dict)
    holder: object = None                 # who loaded the current state (the server or a client)

    def __post_init__(self) -> None:
        self._params = dict(self.net.named_parameters())
        self._buffers = dict(self.net.named_buffers())

    # ---- the trainable parameters
    def trainable_parameters(self) -> list:
        return [self._params[n] for n in self.trainable]

    def num_trainable(self) -> int:
        return int(sum(self._params[n].numel() for n in self.trainable))

    def train_mode(self) -> None:
        """train() for trained parts; modules with running statistics whose parameters are all frozen stay
        in eval mode (their statistics must not move)."""
        self.net.train()
        for m in self.net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters(recurse=False))
                if params and not any(p.requires_grad for p in params):
                    m.eval()

    def eval_mode(self) -> None:
        self.net.eval()

    # ---- states (messages)
    def _rows(self, seen) -> torch.Tensor | None:
        return None if seen is None else torch.as_tensor(np.asarray(seen, dtype=np.int64))

    def get_state(self, seen=None) -> dict:
        """{name: CPU tensor} of the trainable parameters and travelling buffers; head rows of `seen` only."""
        rows = self._rows(seen)
        out = {}
        with torch.no_grad():
            for n in self.trainable:
                t = self._params[n].detach()
                if rows is not None and n in HEAD_PARAMS:
                    t = t.index_select(0, rows.to(t.device))
                out[n] = t.to("cpu", copy=True)
            for n in self.buffers:
                out[n] = self._buffers[n].detach().to("cpu", copy=True)
        return out

    def load_state(self, state: dict, seen=None, holder=None) -> None:
        """Write a state into the network (head rows of `seen` only when seen is given)."""
        rows = self._rows(seen)
        expected = set(self.trainable) | set(self.buffers)
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
            raise KeyError(f"state does not match the network: missing {missing[:5]}, unexpected {extra[:5]}")
        with torch.no_grad():
            for n, value in state.items():
                target = self._params[n] if n in self._params else self._buffers[n]
                value = value.to(device=target.device, dtype=target.dtype)
                if rows is not None and n in HEAD_PARAMS:
                    target.index_copy_(0, rows.to(target.device), value)
                else:
                    target.copy_(value)
        self.holder = holder


def average_states(weighted: list) -> dict:
    """sum_k w_k state_k / sum_k w_k for [(state, w), ...], accumulated in float64, returned in the dtype of
    the first state."""
    total = float(sum(w for _s, w in weighted))
    if not weighted or total <= 0:
        raise ValueError("nothing to average")
    first = weighted[0][0]
    out = {}
    for name, ref in first.items():
        acc = torch.zeros(ref.shape, dtype=torch.float64)
        for state, w in weighted:
            acc += state[name].to(torch.float64) * (float(w) / total)
        out[name] = acc.to(ref.dtype)
    return out


@contextlib.contextmanager
def seeded(seed: int, device: str = "cpu"):
    """torch's generators seeded with `seed` inside the block (torch.manual_seed seeds the CPU and every CUDA
    device), restored afterwards: the global generator state is the same before and after."""
    with torch.random.fork_rng(devices=fork_devices(device)):
        torch.manual_seed(int(seed))
        yield


def build_network(settings: dict, num_classes: int, device: str, seed: int) -> Bundle:
    """settings: the resolved model section (model, pretrained, trainable, last_k, lora_*, prompt_*,
    image_size, augment)."""
    name = settings["model"]
    info_cat = NETWORKS[name]
    mode = settings["trainable"]
    info: dict = {"network": name, "trainable": mode, "pretrained": bool(settings["pretrained"])}
    if info_cat.kind == CNN:
        if settings["pretrained"] or mode != "full":
            raise ValueError(f"{name} is trained from scratch, in full")
        with seeded(seed, device):                 # the CNN's weights, then the head
            encoder = CNNS[name]()
            net = Network(encoder, encoder.feature_dim, num_classes)
        pipeline = ImagePipeline.for_cnn(settings["image_size"], settings["augment"])
    else:
        from flcore.models.vit import PretrainedViT, load_pretrained

        # loaded outside the seeded block: what the libraries draw while building does not shift our draws
        root, preprocess, adapter, winfo, source = load_pretrained(name, device)
        encoder = PretrainedViT(root, adapter, source)
        encoder.freeze()
        with seeded(seed, device):                 # LoRA A or the prompts, then the head
            if mode == "full":
                encoder.unfreeze_all()
            elif mode == "last_k":
                encoder.unfreeze_last(int(settings["last_k"]))
            elif mode == "lora":
                info["lora_tensors"] = encoder.add_lora(settings["lora_targets"], int(settings["lora_rank"]),
                                                        float(settings["lora_alpha"]))
            elif mode == "prompt":
                encoder.add_prompts(int(settings["prompt_length"]), bool(settings["prompt_deep"]))
            elif mode != "head":
                raise ValueError(f"unknown trainable mode {mode!r}")
            net = Network(encoder, encoder.feature_dim, num_classes)
        pipeline = ImagePipeline.for_vit(preprocess, settings["augment"])
        info["weights"] = {k: v for k, v in winfo.items() if k != "preprocess"}
        info["source"] = source
    net = net.to(device)
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    trained_modules = {n.rsplit(".", 1)[0] for n in trainable if "." in n}
    buffers = [n for n, b in net.named_buffers()
               if b.is_floating_point() and n.rsplit(".", 1)[0] in trained_modules]
    bundle = Bundle(net=net, pipeline=pipeline, device=device, trainable=trainable, buffers=buffers)
    total = int(sum(p.numel() for p in net.parameters()))
    info.update({"parameters_total": total, "parameters_trainable": bundle.num_trainable(),
                 "buffers_in_messages": int(sum(bundle._buffers[n].numel() for n in buffers)),
                 "pipeline": pipeline.describe(), "trainable_tensors": len(trainable)})
    bundle.info = info
    return bundle


def make_optimizer(params: list, name: str, lr: float, momentum: float, weight_decay: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def lr_at(base: float, schedule: str, r: int, rounds: int) -> float:
    """Learning rate of round r (0-based) of a task with `rounds` rounds: constant, or cosine from base to near 0."""
    if schedule == "constant":
        return float(base)
    if schedule == "cosine":
        return float(base) * 0.5 * (1.0 + math.cos(math.pi * r / max(rounds, 1)))
    raise ValueError(f"unknown learning-rate schedule {schedule!r}")
