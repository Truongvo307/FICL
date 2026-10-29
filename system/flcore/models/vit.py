"""Pretrained ViTs as encoders of the image path, with LoRA and prompt tokens.

The three backbones are loaded by dataset/utils/backbone_utils.py, the code that writes the feature
caches (load_backbone: weights, evaluation preprocessing, TokenAdapter). PretrainedViT runs the
backbone's own forward pass and reads the output of the last block with a forward hook; its output is
the final-norm CLS token, the vector stored in a cache as (last block, cls, final_ln=True). With
nothing trained, the trained arms therefore start from exactly the features the frozen arms read.

LoRA (Hu et al. 2021) is added as a parametrization (torch.nn.utils.parametrize): the weight a layer
reads becomes W + (alpha / r) B A with B = 0 at the start, so the network starts as the pretrained one.
It works the same for nn.Linear (timm, DINOv2: fused qkv, proj, fc1, fc2) and for nn.MultiheadAttention
(open_clip: in_proj_weight, out_proj, c_fc, c_proj), whose forward reads the weights directly. q, k and v
are row blocks of the fused input projection; each chosen block gets its own pair (A, B).

Prompt tokens (VPT, Jia et al. 2022) are inserted after the prefix tokens (CLS, registers) by a forward
pre-hook on the first block (shallow), and replaced before every later block (deep). Position embeddings
are not added to them. The CLS token stays at position 0, so pooling is unchanged.

Library internals used (checked in the sources of timm 0.9.16, open_clip 2.24.0, the dinov2 hub code
and torch 2.0.1): blocks are called with the tokens as first positional argument; timm and DINOv2
attention modules have qkv (rows q, k, v) and proj, MLPs fc1 and fc2; open_clip blocks have attn
(nn.MultiheadAttention: in_proj_weight rows q, k, v; out_proj) and mlp.c_fc, mlp.c_proj, and see tokens
as [L, N, D]. Parameters the pooled CLS does not use (open_clip proj, DINOv2 mask_token) are never trained.
"""
from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from utils import dataset_code

UNUSED_PARAMETERS = {"open_clip": ("proj",), "torchhub": ("mask_token",), "timm": ()}
_LINEAR_SITES = {"q": ("attn.qkv", "weight", 0), "k": ("attn.qkv", "weight", 1), "v": ("attn.qkv", "weight", 2),
                 "proj": ("attn.proj", "weight", None), "fc1": ("mlp.fc1", "weight", None),
                 "fc2": ("mlp.fc2", "weight", None)}
LORA_SITES = {
    "timm": _LINEAR_SITES,
    "torchhub": _LINEAR_SITES,
    "open_clip": {"q": ("attn", "in_proj_weight", 0), "k": ("attn", "in_proj_weight", 1),
                  "v": ("attn", "in_proj_weight", 2), "proj": ("attn.out_proj", "weight", None),
                  "fc1": ("mlp.c_fc", "weight", None), "fc2": ("mlp.c_proj", "weight", None)},
}
LORA_ORDER = ("q", "k", "v", "proj", "fc1", "fc2")


def load_pretrained(name: str, device: str):
    """(root module, evaluation preprocess, TokenAdapter, provenance info, source) of a pretrained backbone,
    loaded exactly as feature extraction loads it (frozen, eval mode)."""
    bu = dataset_code.load("backbone_utils")
    root, preprocess, adapter, info = bu.load_backbone(name, device)
    return root, preprocess, adapter, info, bu.BACKBONES[name]["source"]


class LoRAUpdate(nn.Module):
    """Parametrization W -> W + (alpha / r) [B_1 A_1; ...] on chosen row blocks of W (other rows unchanged).

    A_i [r, in] start with kaiming_uniform_(a=sqrt(5)) and B_i [rows, r] with zeros, as in the reference
    implementation of LoRA, so the update is zero at the start."""

    def __init__(self, out_features: int, in_features: int, row_blocks, rank: int, alpha: float):
        super().__init__()
        self.out_features, self.in_features = int(out_features), int(in_features)
        self.row_blocks = [(int(a), int(b)) for a, b in row_blocks]
        if any(a >= b for a, b in self.row_blocks) or any(
                b1 > a2 for (_a1, b1), (a2, _b2) in zip(self.row_blocks, self.row_blocks[1:])):
            raise ValueError(f"row blocks must be sorted and disjoint, got {self.row_blocks}")
        self.rank, self.scale = int(rank), float(alpha) / float(rank)
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for a, b in self.row_blocks:
            A = torch.empty(self.rank, self.in_features)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A.append(nn.Parameter(A))
            self.lora_B.append(nn.Parameter(torch.zeros(b - a, self.rank)))

    def delta(self, like: torch.Tensor) -> torch.Tensor:
        pieces, pos = [], 0
        for (a, b), A, B in zip(self.row_blocks, self.lora_A, self.lora_B):
            if a > pos:
                pieces.append(like.new_zeros(a - pos, self.in_features))
            pieces.append((B @ A) * self.scale)
            pos = b
        if pos < self.out_features:
            pieces.append(like.new_zeros(self.out_features - pos, self.in_features))
        return torch.cat(pieces, dim=0)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.delta(W).to(W.dtype)


def _submodule(block: nn.Module, path: str) -> nn.Module:
    return functools.reduce(getattr, path.split("."), block)


class PretrainedViT(nn.Module):
    """x [B, 3, 224, 224] -> final-norm CLS token [B, D] of a pretrained ViT (see the module docstring)."""

    def __init__(self, root: nn.Module, adapter, source: str):
        super().__init__()
        if source not in LORA_SITES:
            raise ValueError(f"unknown backbone source {source!r}")
        self.root = root
        self.source = source
        self.adapter = adapter                     # a plain object: its blocks are submodules of root
        self.feature_dim = int(adapter.embed_dim)
        self.num_blocks = int(adapter.num_blocks)
        self.prompt = None
        self.prompt_length, self.prompt_deep = 0, False
        self._batch, self._captured = 0, None
        adapter.blocks[-1].register_forward_hook(self._capture)

    # ---- forward
    def _capture(self, _module, _inputs, output):
        self._captured = output[0] if isinstance(output, (tuple, list)) else output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._batch, self._captured = int(x.shape[0]), None
        self.adapter.reference_pooled(x)               # the backbone's own forward pass fills _captured
        out, self._captured = self._captured, None
        if out is None:
            raise RuntimeError(f"{self.adapter.name}: the last block did not run in the forward pass")
        tokens = self.adapter.to_nld(out, self._batch)
        # the final norm over every token, then the CLS token: the same operations as extraction's hook
        return self.adapter.final_norm(tokens)[:, 0]

    # ---- which parameters train
    def freeze(self) -> None:
        for p in self.root.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        unused = set(UNUSED_PARAMETERS[self.source])
        for name, p in self.root.named_parameters():
            p.requires_grad_(name not in unused)

    def unfreeze_last(self, k: int) -> None:
        if not 1 <= k <= self.num_blocks:
            raise ValueError(f"last_k must be 1..{self.num_blocks}, got {k}")
        for block in self.adapter.blocks[-k:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in self.adapter.final_norm.parameters():
            p.requires_grad_(True)

    def add_lora(self, targets, rank: int, alpha: float) -> int:
        """LoRA on the chosen weights of every block; returns the number of parametrized tensors. Call it
        under a seeded generator: A is drawn on the CPU, then moved to the weight's device."""
        sites = LORA_SITES[self.source]
        unknown = [t for t in targets if t not in sites]
        if unknown:
            raise ValueError(f"unknown LoRA targets {unknown}; known: {list(sites)}")
        count = 0
        for block in self.adapter.blocks:
            groups: dict = {}
            for target in [t for t in LORA_ORDER if t in targets]:
                path, tensor, chunk = sites[target]
                groups.setdefault((path, tensor), []).append(chunk)
            for (path, tensor), chunks in groups.items():
                module = _submodule(block, path)
                W = getattr(module, tensor)
                out_f, in_f = W.shape
                if None in chunks:
                    rows = [(0, out_f)]
                else:
                    if out_f % 3:
                        raise RuntimeError(f"{path}.{tensor} has {out_f} rows; a fused q, k, v projection was expected")
                    d = out_f // 3
                    rows = [(c * d, (c + 1) * d) for c in sorted(chunks)]
                update = LoRAUpdate(out_f, in_f, rows, rank, alpha).to(device=W.device, dtype=W.dtype)
                parametrize.register_parametrization(module, tensor, update)
                count += 1
        return count

    def add_prompts(self, length: int, deep: bool) -> None:
        """Learnable prompt tokens [blocks or 1, length, D], drawn from N(0, 0.02^2) on the CPU (seed the
        generator before calling), inserted by forward pre-hooks."""
        n = self.num_blocks if deep else 1
        values = torch.empty(n, int(length), self.feature_dim)
        nn.init.normal_(values, std=0.02)
        device = next(self.root.parameters()).device
        self.prompt = nn.Parameter(values.to(device))
        self.prompt_length, self.prompt_deep = int(length), bool(deep)
        blocks = self.adapter.blocks
        blocks[0].register_forward_pre_hook(self._insert_prompts)
        if deep:
            for i in range(1, len(blocks)):
                blocks[i].register_forward_pre_hook(functools.partial(self._replace_prompts, i))

    def _is_nld(self, x: torch.Tensor) -> bool:
        b = self._batch
        if x.dim() != 3:
            raise RuntimeError(f"{self.adapter.name}: expected tokens of 3 dimensions, got {tuple(x.shape)}")
        if x.shape[0] == b and x.shape[1] != b:
            return True
        if x.shape[1] == b and x.shape[0] != b:
            return False
        if self.adapter.layout in ("NLD", "LND"):
            return self.adapter.layout == "NLD"
        raise RuntimeError(f"{self.adapter.name}: cannot tell the token layout apart when the batch size equals "
                           f"the number of tokens ({b}); use another batch size")

    def _with_prompt(self, x: torch.Tensor, i: int, replace: bool) -> torch.Tensor:
        p, L = self.adapter.num_prefix_tokens, self.prompt_length
        prompt = self.prompt[i].to(x.dtype)
        rest = p + L if replace else p
        if self._is_nld(x):
            return torch.cat([x[:, :p], prompt.unsqueeze(0).expand(x.shape[0], -1, -1), x[:, rest:]], dim=1)
        return torch.cat([x[:p], prompt.unsqueeze(1).expand(-1, x.shape[1], -1), x[rest:]], dim=0)

    def _insert_prompts(self, _module, args):
        return (self._with_prompt(args[0], 0, replace=False), *args[1:])

    def _replace_prompts(self, i: int, _module, args):
        return (self._with_prompt(args[0], i, replace=True), *args[1:])
