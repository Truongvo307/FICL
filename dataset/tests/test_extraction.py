"""Tests for dataset/utils/backbone_utils.py (hooks, pooling, the three backbones).

The first tests need only PyTorch (CPU). Tests marked gpu need CUDA and download
the backbone weights; they are checkpoint 3 of reports/PROTOCOL.md.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from utils.backbone_utils import (  # noqa: E402
    BACKBONES,
    BlockRecorder,
    ExtractionPlan,
    GroupSpec,
    TokenAdapter,
    check_timm_hub,
    check_torch_numpy,
    iter_extracted_batches,
    load_backbone,
    package_version,
    resolve_blocks,
)

D = 8


def test_torch_and_numpy_exchange_arrays():
    """First, because every extracted batch crosses torch -> numpy (PyTorch 2.0.x needs NumPy < 2)."""
    check_torch_numpy()


def test_broken_torch_numpy_bridge_is_reported(monkeypatch):
    def broken(_array):
        raise RuntimeError("Numpy is not available")

    monkeypatch.setattr(torch, "from_numpy", broken)
    with pytest.raises(RuntimeError, match=r"numpy<2"):
        check_torch_numpy()


def test_timm_can_use_the_hugging_face_hub():
    """vit_b16_in21k downloads through timm's Hugging Face hub code (needs huggingface_hub < 2)."""
    pytest.importorskip("timm")
    check_timm_hub()


def test_library_versions_come_from_pip_metadata():
    """open_clip 2.24 has no __version__; versions are read from the installed metadata instead."""
    assert package_version("numpy") == np.__version__
    assert package_version("no-such-distribution-xyz") == "unknown"
    pytest.importorskip("open_clip")
    assert package_version("open_clip_torch") != "unknown"


class _FakeViT(nn.Module):
    """CLS = +100, two register tokens = -100, then the given patch tokens."""

    def __init__(self, num_blocks=3, layout="NLD"):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(D)
        self.layout = layout

    def forward(self, patches):
        b = patches.shape[0]
        cls = torch.full((b, 1, D), 100.0)
        reg = torch.full((b, 2, D), -100.0)
        x = torch.cat([cls, reg, patches], dim=1)
        if self.layout == "LND":
            x = x.transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        if self.layout == "LND":
            x = x.transpose(0, 1)
        return self.norm(x)[:, 0]


def _adapter(model, layout):
    return TokenAdapter(name="fake", blocks=list(model.blocks), final_norm=model.norm, num_prefix_tokens=3,
                        embed_dim=D, layout=layout, reference_fn=model.forward)


def test_meanpatch_excludes_prefix_tokens():
    model = _FakeViT()
    patches = torch.randn(4, 5, D)
    ad = _adapter(model, "NLD")
    with BlockRecorder(ad, [0, 2], {("cls", False), ("mean", False), ("mean", True)}) as rec:
        rec.reset(4)
        ad.reference_pooled(patches)          # one forward pass fills the recorder
        assert torch.allclose(rec.get(2, "mean", False), patches.mean(dim=1))
        assert torch.allclose(rec.get(0, "cls", False), torch.full((4, D), 100.0))
        assert torch.allclose(rec.get(2, "mean", True), model.norm(patches).mean(dim=1), atol=1e-6)
    assert all(len(b._forward_hooks) == 0 for b in model.blocks), "hooks must be removed on exit"


def test_final_ln_cls_matches_reference_on_fake_model():
    model = _FakeViT()
    patches = torch.randn(3, 4, D)
    ad = _adapter(model, "NLD")
    with BlockRecorder(ad, [2], {("cls", True)}) as rec:
        rec.reset(3)
        ref = ad.reference_pooled(patches)
        assert torch.allclose(ad.head_of(rec.get(2, "cls", True)), ref)


def test_lnd_layout_handled():
    model = _FakeViT(layout="LND")
    patches = torch.randn(4, 5, D)
    for layout in ("LND", "auto"):
        ad = _adapter(model, layout)
        with BlockRecorder(ad, [1], {("mean", False)}) as rec:
            rec.reset(4)
            ad.reference_pooled(patches)
            assert torch.allclose(rec.get(1, "mean", False), patches.mean(dim=1))


def test_auto_layout_refuses_ambiguous_first_batch():
    model = _FakeViT(layout="LND")
    ad = _adapter(model, "auto")
    patches = torch.randn(8, 5, D)        # 8 tokens and batch size 8: cannot tell the layout apart
    with BlockRecorder(ad, [0], {("cls", False)}) as rec:
        rec.reset(8)
        with pytest.raises(RuntimeError):
            ad.reference_pooled(patches)


def test_resolve_blocks():
    assert resolve_blocks("all", 12) == list(range(12))
    assert resolve_blocks("last", 12) == [11]
    assert resolve_blocks("last6", 12) == [6, 7, 8, 9, 10, 11]
    assert resolve_blocks([0, -1], 12) == [0, 11]
    with pytest.raises(ValueError):
        resolve_blocks([12], 12)
    with pytest.raises(ValueError):
        resolve_blocks([1, 1], 12)
    with pytest.raises(ValueError):
        resolve_blocks("first", 12)


def test_plan_validation():
    with pytest.raises(ValueError):
        GroupSpec(blocks=[0], tokens=["cls", "max"], final_ln=False)
    plan = ExtractionPlan({"pre": GroupSpec([0, 2], ["cls", "mean"], False), "ln": GroupSpec([2], ["cls"], True)})
    by_block = plan.variants_by_block(last_block=2)
    assert by_block[0] == {("cls", False), ("mean", False)}
    assert by_block[2] == {("cls", False), ("mean", False), ("cls", True)}


def test_iter_extracted_batches_on_fake_model():
    model = _FakeViT()
    ad = _adapter(model, "NLD")
    rows = np.arange(6)
    data = torch.randn(6, 5, D)
    loader = [(data[i:i + 4], torch.as_tensor(rows[i:i + 4])) for i in range(0, 6, 4)]
    plan = ExtractionPlan({"pre": GroupSpec([0, 2], ["cls", "mean"], False)})
    out = list(iter_extracted_batches(loader, ad, plan, device="cpu"))
    assert [len(r) for r, _, _ in out] == [4, 2]
    rows0, groups0, check0 = out[0]
    assert groups0["pre"].shape == (4, 2, 2, D) and groups0["pre"].dtype == np.float16
    assert np.allclose(groups0["pre"][:, 1, 1], data[:4].mean(dim=1).numpy().astype(np.float16))
    assert check0 < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_final_cls_matches_model_output(name):
    """Checkpoint 3: hooked last-block CLS + final norm equals the model's own pooled output."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from utils.raw_utils import SyntheticAdapter
    model, preprocess, ad, _info = load_backbone(name, "cuda")
    raw = SyntheticAdapter(num_classes=2, n_train=5, n_val=0, n_test=0, image_size=64, seed=0)
    x = torch.stack([preprocess(raw.load_image(i)) for i in range(10)]).cuda()
    last = ad.num_blocks - 1
    with BlockRecorder(ad, [last], {("cls", True)}) as rec, torch.inference_mode():
        rec.reset(x.shape[0])
        ref = ad.reference_pooled(x)
        ours = ad.head_of(rec.get(last, "cls", True))
    err = (ours - ref).abs().max().item()
    assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-5), f"{name}: max abs diff {err}"


@pytest.mark.gpu
def test_reextraction_identical():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from utils.raw_utils import SyntheticAdapter
    model, preprocess, ad, _info = load_backbone("dinov2_vitb14", "cuda")
    raw = SyntheticAdapter(num_classes=4, n_train=8, n_val=0, n_test=0, image_size=64, seed=0)
    x = torch.stack([preprocess(raw.load_image(i)) for i in range(32)])
    rows = torch.arange(32)
    plan = ExtractionPlan({"pre": GroupSpec(list(range(ad.num_blocks)), ["cls", "mean"], False)})
    runs = []
    for _ in range(2):
        loader = [(x[i:i + 16], rows[i:i + 16]) for i in range(0, 32, 16)]
        runs.append(np.concatenate([g["pre"] for _, g, _ in iter_extracted_batches(loader, ad, plan, "cuda")]))
    assert runs[0].tobytes() == runs[1].tobytes()
