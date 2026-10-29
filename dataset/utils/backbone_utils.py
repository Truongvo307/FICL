"""Frozen backbones, block hooks and token pooling for feature extraction (FICL).

Backbones (ViT-B, 12 blocks, 768-d tokens, 224 px):
    dinov2_vitb14   torch.hub facebookresearch/dinov2, entry dinov2_vitb14
    clip_vitb16     open_clip ViT-B-16, pretrained="openai" (image tower only)
    vit_b16_in21k   timm vit_base_patch16_224.augreg_in21k (override with --model_id)

For every recorded block we pool the block's output tokens into
    cls    the class token
    mean   the mean over patch tokens only (class and register tokens excluded)
with final_ln=False (the raw residual stream) or final_ln=True (the backbone's own final
norm applied to every token first, then pooled). With final_ln=True the last block's cls
equals the model's own pooled output; test_extraction checks this (checkpoint 3).

The library attribute names used here (dinov2 hub: blocks, norm, num_register_tokens;
timm VisionTransformer: blocks, norm, fc_norm, global_pool, num_prefix_tokens;
open_clip VisionTransformer: transformer.resblocks, ln_post, proj) are checked when the
model loads, and every failure says which one is missing. Pinned versions:
timm 0.9.16, open_clip_torch 2.24.0, torch 2.0.1.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn

TOKENS = ("cls", "mean")
EMBED_DIM = 768


def check_torch_numpy() -> None:
    """Fail early when torch cannot exchange arrays with NumPy.

    PyTorch 2.0.x is built against NumPy 1.x. Imported next to NumPy 2 it warns "Failed to initialize
    NumPy: _ARRAY_API not found", and every tensor <-> array conversion (each extracted batch) fails.
    """
    try:
        back = torch.from_numpy(np.arange(3, dtype=np.int64)).numpy()
        ok = back.dtype == np.int64 and back.tolist() == [0, 1, 2]
        error = "" if ok else f"round trip gave {back!r}"
    except Exception as exc:  # noqa: BLE001 - any failure of the bridge is reported the same way
        error = f"{type(exc).__name__}: {exc}"
    if error:
        raise RuntimeError(
            f"torch {torch.__version__} cannot exchange arrays with NumPy {np.__version__} ({error}); "
            f"numpy is imported from {os.path.dirname(np.__file__)}. PyTorch 2.0.x needs NumPy < 2: run  "
            f"python -m pip install \"numpy<2\"  then  python -m pip check. If python still imports NumPy 2 "
            f"from outside this environment, see README, Known problems")


BACKBONES = {
    "dinov2_vitb14": {"source": "torchhub", "repo": "facebookresearch/dinov2", "entry": "dinov2_vitb14"},
    "clip_vitb16": {"source": "open_clip", "model_id": "ViT-B-16", "pretrained": "openai"},
    "vit_b16_in21k": {"source": "timm", "model_id": "vit_base_patch16_224.augreg_in21k"},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# What to extract
# ---------------------------------------------------------------------------
def resolve_blocks(spec, num_blocks: int) -> list[int]:
    """'all' | 'last' | 'lastK' | list of ints (negative = from the end) -> sorted 0-based ids."""
    if isinstance(spec, str):
        if spec == "all":
            return list(range(num_blocks))
        if spec == "last":
            return [num_blocks - 1]
        if spec.startswith("last") and spec[4:].isdigit():
            k = int(spec[4:])
            if not 1 <= k <= num_blocks:
                raise ValueError(f"{spec}: need 1..{num_blocks} blocks")
            return list(range(num_blocks - k, num_blocks))
        raise ValueError(f"unknown block spec {spec!r} (use all, last, lastK or a list of ints)")
    out = []
    for b in spec:
        b = int(b)
        b = b + num_blocks if b < 0 else b
        if not 0 <= b < num_blocks:
            raise ValueError(f"block {b} out of range 0..{num_blocks - 1}")
        out.append(b)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate blocks in {spec}")
    return sorted(out)


@dataclass
class GroupSpec:
    blocks: list[int]          # resolved 0-based block ids
    tokens: list[str]          # subset of TOKENS, stored in this order
    final_ln: bool

    def __post_init__(self) -> None:
        bad = [t for t in self.tokens if t not in TOKENS]
        if bad or not self.tokens:
            raise ValueError(f"tokens must be a non-empty subset of {TOKENS}, got {self.tokens}")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError(f"duplicate tokens in {self.tokens}")
        if not self.blocks:
            raise ValueError("a group needs at least one block")


class ExtractionPlan:
    """Several groups extracted in one forward pass."""

    def __init__(self, groups: dict[str, GroupSpec]):
        if not groups:
            raise ValueError("an extraction plan needs at least one group")
        self.groups = dict(groups)

    def variants_by_block(self, last_block: int) -> dict[int, set[tuple[str, bool]]]:
        """Union of (token, final_ln) needed per block, plus the last block's (cls, True) for the reference check."""
        need: dict[int, set[tuple[str, bool]]] = {}
        for g in self.groups.values():
            for b in g.blocks:
                need.setdefault(b, set()).update((t, g.final_ln) for t in g.tokens)
        need.setdefault(last_block, set()).add(("cls", True))
        return need


# ---------------------------------------------------------------------------
# Token adapter: where the blocks are, how tokens are laid out, what the model outputs
# ---------------------------------------------------------------------------
class TokenAdapter:
    def __init__(self, name: str, blocks: list[nn.Module], final_norm: nn.Module, num_prefix_tokens: int,
                 embed_dim: int, layout: str, reference_fn: Callable[[torch.Tensor], torch.Tensor],
                 project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None):
        if layout not in ("NLD", "LND", "auto"):
            raise ValueError(f"layout must be NLD, LND or auto, got {layout}")
        self.name, self.blocks, self.final_norm = name, list(blocks), final_norm
        self.num_prefix_tokens, self.embed_dim, self.layout = int(num_prefix_tokens), int(embed_dim), layout
        self._reference_fn, self._project_fn = reference_fn, project_fn

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def to_nld(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Return block output as [batch, tokens, dim]. In auto mode the layout is decided on the
        first unambiguous batch and then kept fixed."""
        if x.dim() != 3:
            raise RuntimeError(f"{self.name}: expected a 3-D block output, got shape {tuple(x.shape)}")
        if self.layout == "auto":
            if x.shape[0] == batch_size and x.shape[1] != batch_size:
                self.layout = "NLD"
            elif x.shape[1] == batch_size and x.shape[0] != batch_size:
                self.layout = "LND"
            else:
                raise RuntimeError(f"{self.name}: cannot tell the token layout apart when the batch size "
                                   f"equals the number of tokens ({batch_size}); use another batch size")
        out = x if self.layout == "NLD" else x.transpose(0, 1)
        if out.shape[0] != batch_size or out.shape[2] != self.embed_dim:
            raise RuntimeError(f"{self.name}: block output {tuple(x.shape)} does not fit batch {batch_size}, "
                               f"dim {self.embed_dim} with layout {self.layout}")
        return out

    def reference_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full model and return its own pooled output (this forward also fills the hooks)."""
        return self._reference_fn(x)

    def head_of(self, pooled_ln_cls: torch.Tensor) -> torch.Tensor:
        """Map the final-norm CLS to the space of reference_pooled (CLIP: @ proj; others: identity)."""
        return self._project_fn(pooled_ln_cls) if self._project_fn is not None else pooled_ln_cls


class BlockRecorder:
    """Forward hooks that pool selected blocks' outputs. Use as a context manager."""

    def __init__(self, adapter: TokenAdapter, blocks: Iterable[int], variants):
        self.adapter = adapter
        if isinstance(variants, dict):
            self.variants = {int(b): set(v) for b, v in variants.items()}
        else:
            self.variants = {int(b): set(variants) for b in blocks}
        for b in self.variants:
            if not 0 <= b < adapter.num_blocks:
                raise ValueError(f"block {b} out of range 0..{adapter.num_blocks - 1}")
            for tok, _ln in self.variants[b]:
                if tok not in TOKENS:
                    raise ValueError(f"unknown token {tok}")
        self._handles = []
        self._store: dict[tuple[int, str, bool], torch.Tensor] = {}
        self._batch_size: int | None = None

    def __enter__(self) -> "BlockRecorder":
        for b in sorted(self.variants):
            self._handles.append(self.adapter.blocks[b].register_forward_hook(self._make_hook(b)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self, batch_size: int) -> None:
        self._store.clear()
        self._batch_size = int(batch_size)

    def _make_hook(self, block: int):
        def hook(_module, _inputs, output):
            if self._batch_size is None:
                raise RuntimeError("call reset(batch_size) before the forward pass")
            out = output[0] if isinstance(output, (tuple, list)) else output
            x = self.adapter.to_nld(out, self._batch_size).float()
            need = self.variants[block]
            xn = self.adapter.final_norm(x) if any(ln for _t, ln in need) else None
            p = self.adapter.num_prefix_tokens
            for tok, ln in need:
                src = xn if ln else x
                pooled = src[:, 0] if tok == "cls" else src[:, p:].mean(dim=1)
                self._store[(block, tok, ln)] = pooled
        return hook

    def get(self, block: int, token: str, final_ln: bool) -> torch.Tensor:
        key = (int(block), token, bool(final_ln))
        if key not in self._store:
            raise KeyError(f"{key} was not recorded in this forward pass")
        return self._store[key]


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
def _param_sha256(module: nn.Module) -> str:
    """Fingerprint of the weights actually loaded (names + fp32 bytes, sorted by name)."""
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def package_version(distribution: str) -> str:
    """Installed version of a pip distribution, e.g. "open_clip_torch" (open_clip 2.24 has no __version__)."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_timm_hub() -> None:
    """Fail early, with the fix, when timm cannot use huggingface_hub.

    timm 0.9.16 downloads vit_b16_in21k from the Hugging Face hub. It imports these names from
    huggingface_hub and, if any import fails, reports "Hugging Face hub model specified but package not
    installed" even when the package is installed: huggingface_hub 2.x no longer exports
    repo_type_and_id_from_hf_id (checked in its sources: present in 0.36.2 and 1.33.0, gone in 2.0.0)."""
    try:
        from huggingface_hub import (  # noqa: F401 - the imports of timm/models/_hub.py (timm 0.9.16)
            create_repo,
            get_hf_file_metadata,
            hf_hub_download,
            hf_hub_url,
            repo_type_and_id_from_hf_id,
            upload_folder,
        )
        from huggingface_hub.utils import EntryNotFoundError  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"timm 0.9.16 cannot use huggingface_hub {package_version('huggingface_hub')} ({exc}), so it cannot "
            f"download vit_b16_in21k. Run  python -m pip install \"huggingface_hub<1\"  (env_cuda.yaml pins it)"
        ) from None


def _require(obj, attr: str, where: str):
    if not hasattr(obj, attr):
        raise RuntimeError(f"{where}: expected attribute {attr!r} is missing; check the pinned library version")
    return getattr(obj, attr)


def make_token_adapter(source: str, model: nn.Module, name: str) -> TokenAdapter:
    """TokenAdapter of a loaded model: source "torchhub" (a DINOv2 model), "timm" (a timm VisionTransformer)
    or "open_clip" (the image tower, CLIP.visual). load_backbone() uses it for the three backbones; the
    training path and the tests use it on networks of the same classes."""
    if source == "torchhub":
        blocks = list(_require(model, "blocks", name))
        if getattr(model, "chunked_blocks", False):
            blocks = [b for chunk in blocks for b in chunk if not isinstance(b, nn.Identity)]
        norm = _require(model, "norm", name)
        num_prefix = 1 + int(getattr(model, "num_register_tokens", 0))
        embed_dim = int(getattr(model, "embed_dim", EMBED_DIM))
        return TokenAdapter(name, blocks, norm, num_prefix, embed_dim, "NLD", reference_fn=model.forward)
    if source == "open_clip":
        visual = model
        if getattr(visual, "attn_pool", None) is not None:
            raise NotImplementedError(f"{name}: attentional pooling is not supported")
        transformer = _require(visual, "transformer", name)
        blocks = list(_require(transformer, "resblocks", name))
        norm = _require(visual, "ln_post", name)
        proj = getattr(visual, "proj", None)
        batch_first = getattr(transformer, "batch_first", None)
        layout = "NLD" if batch_first is True else ("LND" if batch_first is False else "auto")
        embed_dim = int(getattr(visual, "width", None) or norm.normalized_shape[0])
        return TokenAdapter(name, blocks, norm, 1, embed_dim, layout, reference_fn=visual.forward,
                            project_fn=(lambda z, p=proj: z @ p) if proj is not None else None)
    if source == "timm":
        if getattr(model, "global_pool", None) != "token" or not isinstance(getattr(model, "fc_norm", None), nn.Identity):
            raise NotImplementedError(f"{name}: expected CLS-token pooling without fc_norm")
        blocks = list(_require(model, "blocks", name))
        norm = _require(model, "norm", name)
        num_prefix = int(_require(model, "num_prefix_tokens", name))
        embed_dim = int(getattr(model, "embed_dim", EMBED_DIM))
        return TokenAdapter(name, blocks, norm, num_prefix, embed_dim, "NLD", reference_fn=model.forward)
    raise ValueError(f"unknown backbone source {source!r}")


def load_backbone(name: str, device: str, model_id: str | None = None, hub_ref: str | None = None
                  ) -> tuple[nn.Module, Callable, TokenAdapter, dict]:
    """Return (model in eval mode on device, preprocess transform, TokenAdapter, provenance info)."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    cfg = BACKBONES[name]
    info: dict = {"backbone": name, "source": cfg["source"]}

    if cfg["source"] == "torchhub":
        from torchvision import transforms

        repo = cfg["repo"] + (f":{hub_ref}" if hub_ref else "")
        model = torch.hub.load(repo, cfg["entry"], trust_repo=True)
        info.update({"repo": repo, "entry": cfg["entry"]})
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        root = model
        adapter = make_token_adapter("torchhub", model, name)

    elif cfg["source"] == "open_clip":
        import open_clip

        mid = model_id or cfg["model_id"]
        clip_model, _, preprocess = open_clip.create_model_and_transforms(mid, pretrained=cfg["pretrained"])
        visual = _require(clip_model, "visual", name)
        adapter = make_token_adapter("open_clip", visual, name)
        info.update({"model_id": mid, "pretrained": cfg["pretrained"], "open_clip": package_version("open_clip_torch"),
                     "token_layout": adapter.layout})
        root = visual

    elif cfg["source"] == "timm":
        import timm
        from timm.data import create_transform, resolve_data_config

        mid = model_id or cfg["model_id"]
        check_timm_hub()
        model = timm.create_model(mid, pretrained=True, num_classes=0)
        adapter = make_token_adapter("timm", model, name)
        data_cfg = resolve_data_config({}, model=model)
        preprocess = create_transform(**data_cfg)
        info.update({"model_id": mid, "timm": package_version("timm"), "data_config": {k: str(v) for k, v in data_cfg.items()}})
        root = model
    else:
        raise ValueError(cfg["source"])

    root = root.to(device).eval()
    for p in root.parameters():
        p.requires_grad_(False)
    if len(adapter.blocks) != 12:
        raise RuntimeError(f"{name}: expected 12 transformer blocks, found {len(adapter.blocks)}")
    info["param_sha256"] = _param_sha256(root)
    info["preprocess"] = repr(preprocess)
    return root, preprocess, adapter, info


# ---------------------------------------------------------------------------
# Extraction loop shared by extract_features.py and the tests
# ---------------------------------------------------------------------------
def set_determinism() -> None:
    """fp32 forward without TF32, deterministic cuDNN, and the math kernel of scaled_dot_product_attention
    (DINOv2 and timm call it): with the flash / memory-efficient kernels two extractions of the same images
    differed (test_reextraction_identical on the GPU machine, 28 October 2026)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def iter_extracted_batches(loader, adapter: TokenAdapter, plan: ExtractionPlan, device: str
                           ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray], float]]:
    """For each (images, rows) batch yield (rows, {group: float16 [B, L, T, D]}, reference check).

    The reference check is max |head_of(last-block final-norm CLS) - model output| for the batch.
    Calls set_determinism() first (fp32, no TF32, deterministic cuDNN).
    """
    set_determinism()
    last = adapter.num_blocks - 1
    with BlockRecorder(adapter, [], plan.variants_by_block(last)) as rec, torch.inference_mode():
        for images, rows in loader:
            images = images.to(device, non_blocking=True)
            rec.reset(images.shape[0])
            ref = adapter.reference_pooled(images)
            check = (adapter.head_of(rec.get(last, "cls", True)) - ref.float()).abs().max().item()
            out = {}
            for gname, g in plan.groups.items():
                per_block = [torch.stack([rec.get(b, t, g.final_ln) for t in g.tokens], dim=1) for b in g.blocks]
                out[gname] = torch.stack(per_block, dim=1).to(torch.float16).cpu().numpy()
            rows_np = rows.numpy() if isinstance(rows, torch.Tensor) else np.asarray(rows)
            yield rows_np.astype(np.int64), out, check


class RowImages(torch.utils.data.Dataset):
    """(preprocessed image, row) for the given rows of a raw adapter."""

    def __init__(self, raw_adapter, preprocess: Callable, rows: np.ndarray):
        self.raw, self.preprocess = raw_adapter, preprocess
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        return self.preprocess(self.raw.load_image(r)), r


def hub_weights_sha256(prefix: str) -> dict:
    """sha256 of torch.hub checkpoint files starting with prefix (provenance for dinov2)."""
    ck = os.path.join(torch.hub.get_dir(), "checkpoints")
    out = {}
    if os.path.isdir(ck):
        for fn in sorted(os.listdir(ck)):
            if fn.startswith(prefix):
                h = hashlib.sha256()
                with open(os.path.join(ck, fn), "rb") as f:
                    for block in iter(lambda: f.read(1 << 24), b""):
                        h.update(block)
                out[fn] = h.hexdigest()
    return out
