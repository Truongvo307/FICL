"""The image path with torch, on the CPU: the networks, the transforms, LoRA and prompt tokens on small
timm and open_clip ViTs, and FedAvg / FedAvgJoint end to end on the synthetic images with the tiny CNN.

GPU tests (-m gpu) check what only the real backbones can show: the pretrained encoder gives exactly
the features the caches store, every trainable parameter of every mode receives a gradient, and GPU
runs are reproducible. They download the pretrained weights on first use.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import helpers as H  # noqa: E402
import main  # noqa: E402
from conftest import load_dataset_module  # noqa: E402

SCHEDULE = ["-rpt", "3", "-ls", "2", "-lbs", "16", "-lr", "0.1", "--momentum", "0.9"]
# The learning tests use two iid clients: every batch mixes the three domains of the synthetic images.
# With one domain per client (natural) or a few classes per client (dir 0.5), each client's batch-norm
# statistics differ from the averaged ones the evaluation uses, and the tiny CNN evaluated near chance
# (natural: final accuracy 0.125 with 8 classes, the training loss still went down).
LEARN = ["-rpt", "5", "-ls", "4", "-lbs", "16", "-lr", "0.05", "--momentum", "0.9"]
NO_MODE_KEYS = {"last_k": None, "lora_rank": None, "lora_alpha": None, "lora_targets": None,
                "prompt_length": None, "prompt_deep": None}


def _argv(sr, *extra, algo="FedAvg", tag="run", workers=0, device="cpu", schedule=SCHEDULE, iid=False):
    clients = ["-partition", "iid", "-nc", "2"] if iid else ["-partition", "dir", "-nc", "5", "-alpha", "0.5"]
    return ["-data", "Synthetic", *clients, "-cpt", "2",
            "-droot", sr["root"], "-algo", algo, "-m", "tiny_cnn", *schedule, "-nw", str(workers), "-dev", device,
            "-exp", os.path.join(sr["results"], tag), *extra]


def _csv(path):
    return H.read_csv(path)[1]


def _add_iid_scenario(sr, clients=2):
    """Write cl_<clients>clients_iid_cpt2_seed0 next to the synthetic dataset's other scenarios."""
    scen = load_dataset_module("scenario_utils.py", "_ds_scenario_utils")
    spec = scen.ScenarioSpec(dataset="Synthetic", partition="iid", num_clients=clients, class_per_task=2, seed=0)
    scen.save_scenario(scen.build_scenario(sr["index"], spec), os.path.join(sr["root"], "Synthetic"))


def _bu():
    return load_dataset_module("backbone_utils.py", "_ds_backbone_utils")


# ---------------------------------------------------------------------------
# networks and transforms
# ---------------------------------------------------------------------------
def test_cnns_have_the_published_sizes():
    from flcore.models.cnn import ResNet18Cifar, ResNet32, TinyCNN

    x = torch.randn(2, 3, 32, 32)
    for cls, n_params, dim in ((TinyCNN, 5136, 32), (ResNet18Cifar, 11168832, 512), (ResNet32, 463504, 64)):
        net = cls()
        assert sum(p.numel() for p in net.parameters()) == n_params, cls.__name__
        assert net(x).shape == (2, dim) and net.feature_dim == dim
    assert ResNet32()(torch.randn(1, 3, 64, 64)).shape == (1, 64)          # any input size (global mean pool)


def test_cnn_pipeline():
    from flcore.models.transforms import IMAGENET_MEAN, IMAGENET_STD, ImagePipeline
    from torchvision.transforms import functional as F

    raw = H.load_dataset_raw_utils()
    img = raw.SyntheticAdapter(image_size=32, pattern="class").load_image(3)
    p = ImagePipeline.for_cnn(32, "crop_flip")
    x = p.eval(img)
    assert x.shape == (3, 32, 32) and torch.allclose(x, F.normalize(F.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD))
    a = p.train(img, np.random.default_rng([1, 2]))
    assert torch.equal(a, p.train(img, np.random.default_rng([1, 2])))   # fixed by the generator
    assert any(not torch.equal(a, p.train(img, np.random.default_rng([1, s]))) for s in range(3, 8))
    assert torch.equal(ImagePipeline.for_cnn(32, "none").train(img, np.random.default_rng(0)), x)
    big = raw.SyntheticAdapter(image_size=50, pattern="class").load_image(3)
    for aug in ("flip", "crop_flip", "rrc_flip"):
        assert ImagePipeline.for_cnn(32, aug).train(big, np.random.default_rng(0)).shape == (3, 32, 32)


def test_vit_pipeline_reads_the_backbone_preprocessing():
    from flcore.models.transforms import ImagePipeline, describe_preprocess
    from torchvision import transforms as T

    pre = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    d = describe_preprocess(pre)
    assert (d["resize"], d["crop"], d["interpolation"]) == (256, 224, T.InterpolationMode.BICUBIC)
    raw = H.load_dataset_raw_utils()
    img = raw.SyntheticAdapter(image_size=64, pattern="class").load_image(0)
    p = ImagePipeline.for_vit(pre, "rrc_flip")
    assert torch.equal(p.eval(img), pre(img))                              # the very same evaluation transform
    for aug in ("none", "flip", "crop_flip", "rrc_flip"):
        assert ImagePipeline.for_vit(pre, aug).train(img, np.random.default_rng(5)).shape == (3, 224, 224)


def test_lora_parametrization_on_multihead_attention():
    from flcore.models.vit import LoRAUpdate
    from torch.nn.utils import parametrize

    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(16, 2)
    x = torch.randn(5, 3, 16)                                              # [tokens, batch, dim]
    with torch.no_grad():
        ref = mha(x, x, x, need_weights=False)[0]
    for p in mha.parameters():
        p.requires_grad_(False)
    update = LoRAUpdate(48, 16, [(0, 16), (32, 48)], rank=2, alpha=2.0)    # q and v rows of in_proj_weight
    parametrize.register_parametrization(mha, "in_proj_weight", update)
    with torch.no_grad():
        assert torch.allclose(mha(x, x, x, need_weights=False)[0], ref, atol=1e-6)   # B = 0: unchanged
    mha(x, x, x, need_weights=False)[0].pow(2).sum().backward()
    assert all(p.grad is not None for p in update.parameters())
    assert mha.parametrizations.in_proj_weight.original.grad is None


def _tiny_timm():
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(0)
    return VisionTransformer(img_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2, num_classes=0)


def test_timm_vit_encoder_lora_prompts_and_trainable_parts():
    from flcore.models.vit import PretrainedViT

    x = torch.randn(3, 3, 32, 32)
    vit = _tiny_timm()
    enc = PretrainedViT(vit, _bu().make_token_adapter("timm", vit, "tiny_timm"), "timm")
    with torch.no_grad():
        ref = vit(x)                                                       # timm's own pooled, normed CLS
        assert torch.allclose(enc(x), ref, atol=1e-6)
    enc.freeze()
    assert enc.add_lora(["q", "v"], rank=4, alpha=4.0) == 2                # one fused qkv per block
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert len(trainable) == 8 and all("lora_" in n for n in trainable)
    with torch.no_grad():
        assert torch.allclose(enc(x), ref, atol=1e-6)                     # B = 0 at the start
    qkv = vit.blocks[0].attn.qkv
    w0 = qkv.parametrizations.weight.original.clone()
    opt = torch.optim.SGD([p for p in enc.parameters() if p.requires_grad], lr=0.5)
    enc(x).pow(2).sum().backward()
    opt.step()
    assert torch.equal(qkv.parametrizations.weight.original, w0)          # the pretrained weight is untouched
    delta = (qkv.weight - w0).detach()
    assert delta[:32].abs().max() > 0 and delta[64:].abs().max() > 0 and delta[32:64].abs().max() == 0   # q, v

    for deep in (False, True):
        vit = _tiny_timm()
        enc = PretrainedViT(vit, _bu().make_token_adapter("timm", vit, "tiny_timm"), "timm")
        enc.freeze()
        tokens = []
        vit.blocks[1].register_forward_pre_hook(lambda _m, a: tokens.append(a[0].shape[1]))
        enc.add_prompts(3, deep=deep)
        out = enc(x)
        assert tokens == [1 + 16 + 3] and enc.prompt.shape == ((2 if deep else 1), 3, 32)
        out.sum().backward()
        assert enc.prompt.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in enc.named_parameters() if n != "prompt")

    vit = _tiny_timm()
    enc = PretrainedViT(vit, _bu().make_token_adapter("timm", vit, "tiny_timm"), "timm")
    enc.freeze()
    enc.unfreeze_last(1)
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(("root.blocks.1.", "root.norm.")) for n in names)
    enc.unfreeze_all()
    assert all(p.requires_grad for p in enc.parameters())


def test_open_clip_vit_encoder_lora_and_prompts():
    pytest.importorskip("open_clip")
    from flcore.models.vit import PretrainedViT
    from open_clip.transformer import VisionTransformer

    def tiny():
        torch.manual_seed(0)
        return VisionTransformer(image_size=32, patch_size=8, width=32, layers=2, heads=2, mlp_ratio=2.0,
                                 output_dim=16)

    x = torch.randn(3, 3, 32, 32)
    vt = tiny()
    enc = PretrainedViT(vt, _bu().make_token_adapter("open_clip", vt, "tiny_clip"), "open_clip")
    with torch.no_grad():
        ref = vt(x)                                                        # ln_post(CLS) @ proj
        assert torch.allclose(enc(x) @ vt.proj, ref, atol=1e-5)           # the encoder stops before proj
    enc.freeze()
    assert enc.add_lora(["q", "k", "v", "proj", "fc1", "fc2"], rank=2, alpha=2.0) == 8    # 4 weights x 2 blocks
    with torch.no_grad():
        assert torch.allclose(enc(x) @ vt.proj, ref, atol=1e-5)
    enc(x).sum().backward()
    lora = [p for n, p in enc.named_parameters() if "lora_B" in n]
    assert len(lora) == 2 * (3 + 3) and all(p.grad is not None for p in lora)
    enc.unfreeze_all()
    assert not vt.proj.requires_grad                                       # never used by the pooled CLS

    vt = tiny()
    enc = PretrainedViT(vt, _bu().make_token_adapter("open_clip", vt, "tiny_clip"), "open_clip")
    enc.freeze()
    tokens = []
    vt.transformer.resblocks[1].register_forward_pre_hook(lambda _m, a: tokens.append(tuple(a[0].shape)))
    enc.add_prompts(2, deep=True)
    assert enc(x).shape == (3, 32) and tokens == [(1 + 16 + 2, 3, 32)]     # open_clip blocks see [L, N, D]


# ---------------------------------------------------------------------------
# FedAvg and FedAvgJoint end to end (tiny CNN, CPU)
# ---------------------------------------------------------------------------
def test_cross_entropy_matches_torch_and_ignores_masked_classes():
    from flcore.models.network import cross_entropy

    logits = torch.randn(6, 5, generator=torch.Generator().manual_seed(0), requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    assert torch.allclose(cross_entropy(logits, y), torch.nn.functional.cross_entropy(logits, y), atol=1e-6)
    blocked = torch.tensor([False, False, True, True, True])
    masked = logits.masked_fill(blocked, float("-inf"))
    loss = cross_entropy(masked, torch.tensor([0, 1, 1, 0, 1, 0]))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[:, 2:].abs().max() == 0        # masked classes get no gradient
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits[:, :2], torch.tensor([0, 1, 1, 0, 1, 0])))


def test_fedavg_mean_is_weighted_by_samples():
    from flcore.models.network import average_states

    s1 = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    s2 = {"w": torch.tensor([5.0, 6.0]), "b": torch.tensor([4.0])}
    mean = average_states([(s1, 1), (s2, 3)])
    assert torch.allclose(mean["w"], torch.tensor([4.0, 5.0])) and torch.allclose(mean["b"], torch.tensor([3.0]))
    assert mean["w"].dtype == torch.float32


def test_fedavg_trains_a_cnn_from_scratch_end_to_end(synthetic_root):
    _add_iid_scenario(synthetic_root)
    argv = _argv(synthetic_root, "-go", "e2e", tag="fedavg", schedule=LEARN, iid=True)
    assert main.main(argv) == 0                                                 # every check of check_run passes
    run = os.path.join(synthetic_root["results"], "fedavg", "times_0")
    tasks = _csv(os.path.join(run, "tasks.csv"))
    rounds = _csv(os.path.join(run, "rounds.csv"))
    assert len(tasks) == 4 and len(rounds) == 20 and all(r["train_loss"] for r in rounds)
    loss = [float(r["train_loss"]) for r in rounds[:5]]
    accuracy = [float(t["acc_task_new"]) for t in tasks]
    assert loss[-1] < loss[0], loss                                             # task 0: the loss goes down
    # each new task has 2 classes (chance 0.5). The first task starts from random features and learns
    # slowest (on the GPU machine: 0.7, then 1.0, 0.825, 1.0, with the task-0 loss still falling)
    assert min(accuracy) > 0.6 and sum(accuracy) / len(accuracy) > 0.75, (accuracy, loss)
    run_info = H.read_json(os.path.join(run, "run.json"))
    assert run_info["input"] == "images" and run_info["model"]["model"] == "tiny_cnn"
    assert run_info["network"]["parameters_trainable"] == 5136 + 8 * 32 + 8
    assert "SMOKE_TEST" in run_info["run_flags"] and "REFERENCE" in run_info["run_flags"]


def test_messages_carry_the_trainable_state_and_the_seen_head_rows(synthetic_root):
    from flcore.servers.serverfedavg import FedAvg

    cfg = H.make_image_cfg(synthetic_root["root"], synthetic_root["results"], "FedAvg")
    server = H.run_server(FedAvg, cfg, os.path.join(synthetic_root["results"], "bytes"))
    b = server.bundle
    body = sum(b._params[n].numel() for n in b.trainable if not n.startswith("head."))
    buffers = sum(b._buffers[n].numel() for n in b.buffers)
    assert buffers == 2 * (16 + 32)                         # batch-norm running means and variances travel too
    for row in _csv(os.path.join(server.run_dir, "comm.csv")):
        seen = 2 * (int(row["task"]) + 1)
        expected = 4 * (body + buffers + seen * (32 + 1)) + (8 if row["direction"] == "up" else 0)
        assert int(row["nbytes"]) == expected, row


def test_batch_norm_statistics_travel_and_are_averaged(synthetic_root):
    from flcore.servers.serverfedavg import FedAvg
    from utils.log_utils import close_log_files

    cfg = H.make_image_cfg(synthetic_root["root"], synthetic_root["results"], "FedAvg")
    server = H.build_server(FedAvg, cfg, os.path.join(synthetic_root["results"], "bn"))
    before = {k: v.clone() for k, v in server.global_state.items()}
    uploads = []
    aggregate = server.aggregate
    server.aggregate = lambda t, r, ups: (uploads.extend(ups), aggregate(t, r, ups))[1]
    server._begin_task(0)
    server._run_round(0, 0)
    close_log_files(server.logger)
    for name in ("encoder.body.1.running_mean", "encoder.body.1.running_var", "encoder.body.5.running_var"):
        assert uploads and all(not torch.equal(p["state"][name], before[name]) for _c, p in uploads), name
        total = sum(int(p["n"]) for _c, p in uploads)
        expected = sum(p["state"][name].double() * int(p["n"]) for _c, p in uploads) / total
        assert torch.allclose(server.global_state[name].double(), expected, atol=1e-6), name


def test_training_never_reads_test_images(synthetic_root, monkeypatch):
    from flcore.servers.serverfedavg import FedAvg

    cfg = H.make_image_cfg(synthetic_root["root"], synthetic_root["results"], "FedAvg")
    server = H.build_server(FedAvg, cfg, os.path.join(synthetic_root["results"], "guard"))
    store, split = server.store, np.load(os.path.join(synthetic_root["root"], "Synthetic", "cl_index", "split.npy"))
    reads = []
    original = store.adapter.load_image
    monkeypatch.setattr(store.adapter, "load_image", lambda row: (reads.append((int(row), store.in_evaluation)),
                                                                   original(row))[1])
    server.train()
    test_reads = [(r, e) for r, e in reads if split[r] == 2]
    assert test_reads and all(e for _r, e in test_reads)                   # test images: evaluation only
    assert any(not e for r, e in reads if split[r] == 0)                   # training images: read in training


def test_runs_are_reproducible_and_do_not_depend_on_loader_workers(synthetic_root):
    for tag, workers in (("a", 0), ("b", 0), ("c", 2)):
        assert main.main(_argv(synthetic_root, tag=tag, workers=workers)) == 0
    files = {}
    for tag in "abc":
        run = os.path.join(synthetic_root["results"], tag, "times_0")
        with open(os.path.join(run, "acc_global.csv")) as f:
            acc = f.read()
        losses = [r["train_loss"] for r in _csv(os.path.join(run, "rounds.csv"))]
        files[tag] = (acc, losses)
    assert files["a"] == files["b"] == files["c"]


def test_joint_upper_bound_trains_on_every_task_and_forgets_less(synthetic_root):
    from flcore.servers.serverfedavg import FedAvg, FedAvgJoint

    for cls, name in ((FedAvg, "FedAvg"), (FedAvgJoint, "FedAvgJoint")):
        cfg = H.make_image_cfg(synthetic_root["root"], synthetic_root["results"], name)
        server = H.build_server(cls, cfg, os.path.join(synthetic_root["results"], f"rows_{name}"))
        scn = server.scenario
        for c in server.clients:
            c._begin_task(0)
            c._begin_task(1)
            expected = (np.concatenate([scn.train_rows[0][c.id], scn.train_rows[1][c.id]]) if cls is FedAvgJoint
                        else scn.train_rows[1][c.id])
            assert np.array_equal(c.training_rows(), expected)
    _add_iid_scenario(synthetic_root)
    finals = {}
    for algo in ("FedAvg", "FedAvgJoint"):
        argv = _argv(synthetic_root, algo=algo, tag=f"final_{algo}", schedule=LEARN, iid=True)
        assert main.main(argv) == 0
        finals[algo] = H.read_json(os.path.join(synthetic_root["results"], f"final_{algo}", "times_0",
                                                "final.json"))["metrics"]["acc_final"]
    # 8 classes at the end, chance 0.125; FedAvg forgets the earlier tasks, joint training does not
    assert finals["FedAvgJoint"] > 0.5 and finals["FedAvgJoint"] > finals["FedAvg"] + 0.1, finals


# ---------------------------------------------------------------------------
# GPU: the real backbones (downloads the weights on first use)
# ---------------------------------------------------------------------------
def _settings(name, mode, **kw):
    return {"model": name, "pretrained": True, "trainable": mode, **NO_MODE_KEYS, "image_size": 224,
            "augment": "none", **kw}


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["vit_b16_in21k", "dinov2_vitb14", "clip_vitb16"])
def test_pretrained_encoder_gives_the_cached_features(name):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from flcore.models.network import build_network, configure_determinism
    from utils import dataset_code

    configure_determinism()          # the math attention kernel: the fused ones differ between two passes
    bundle = build_network(_settings(name, "head"), num_classes=10, device="cuda", seed=0)
    raw = H.load_dataset_raw_utils()
    images = raw.SyntheticAdapter(num_classes=2, n_train=4, n_test=0, image_size=96, pattern="class")
    x = torch.stack([bundle.pipeline.eval(images.load_image(i)) for i in range(8)]).cuda()
    enc = bundle.net.encoder
    last = enc.num_blocks - 1
    bu = dataset_code.load("backbone_utils")                    # the module feature extraction runs
    with bu.BlockRecorder(enc.adapter, [], {last: {("cls", True)}}) as rec, torch.no_grad():
        rec.reset(8)
        enc.adapter.reference_pooled(x)
        cached = rec.get(last, "cls", True)
        mine = enc(x)
    assert torch.allclose(mine, cached, rtol=1e-6, atol=1e-6), (mine - cached).abs().max().item()


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["vit_b16_in21k", "dinov2_vitb14", "clip_vitb16"])
@pytest.mark.parametrize("mode", ["full", "last_k", "lora", "prompt"])
def test_every_trainable_parameter_gets_a_gradient(name, mode):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from flcore.models.network import build_network

    extra = {"last_k": {"last_k": 2}, "prompt": {"prompt_length": 4, "prompt_deep": True},
             "lora": {"lora_rank": 4, "lora_alpha": 4.0, "lora_targets": ["q", "k", "v", "proj", "fc1", "fc2"]}}
    bundle = build_network(_settings(name, mode, **extra.get(mode, {})), num_classes=10, device="cuda", seed=0)
    bundle.train_mode()
    bundle.net(torch.randn(2, 3, 224, 224, device="cuda"))[:, :3].logsumexp(dim=1).sum().backward()
    assert [n for n in bundle.trainable if bundle._params[n].grad is None] == []
    assert [n for n, p in bundle.net.named_parameters() if not p.requires_grad and p.grad is not None] == []


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["vit_b16_in21k", "dinov2_vitb14", "clip_vitb16"])
def test_vit_training_steps_repeat_exactly_on_the_gpu(name):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from flcore.models.network import build_network, configure_determinism, cross_entropy

    configure_determinism()
    x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(0)).cuda()
    y = torch.tensor([0, 1, 2, 3]).cuda()
    states = []
    for _repeat in range(2):
        bundle = build_network(_settings(name, "lora", lora_rank=4, lora_alpha=4.0, lora_targets=["q", "v"]),
                               num_classes=10, device="cuda", seed=0)
        opt = torch.optim.SGD(bundle.trainable_parameters(), lr=0.1)
        bundle.train_mode()
        for _step in range(2):
            loss = cross_entropy(bundle.net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        states.append(bundle.get_state())
        del bundle, opt
    assert [k for k in states[0] if not torch.equal(states[0][k], states[1][k])] == []


@pytest.mark.gpu
def test_gpu_runs_are_reproducible(synthetic_root):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    for tag in ("g1", "g2"):
        assert main.main(_argv(synthetic_root, tag=tag, device="cuda")) == 0
    texts = []
    for tag in ("g1", "g2"):
        with open(os.path.join(synthetic_root["results"], tag, "times_0", "acc_global.csv")) as f:
            texts.append(f.read())
    assert texts[0] == texts[1]
