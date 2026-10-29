"""The image path without torch: settings of the model section, ImageStore and its guard, augmentation
parameters, listings and dry runs. (The training itself is tested in test_images_torch.py.)"""
import contextlib
import io
import os

import numpy as np
import pytest

import helpers as H
import main
from flcore.servers.serverfedavg import FedAvg, FedAvgJoint
from flcore.servers.serverfedavglin import SmokeFedAvgLinear
from utils.config_utils import ConfigError
from utils.data_utils import TestDataAccessError
from utils.image_utils import (ImageStore, flip_coin, random_crop_params, random_resized_crop_params)


def _args(sr, model=None, hparams=None, run=None, name="FedAvg", cls=FedAvg, tmp="exp"):
    cfg = H.make_image_cfg(sr["root"], sr["results"], name, model=model, hparams=hparams, run=run)
    return H.build_args(cls, cfg, os.path.join(sr["results"], tmp))


def test_model_defaults_are_filled_and_recorded(synthetic_root):
    a = _args(synthetic_root)
    assert (a.model, a.pretrained, a.trainable, a.image_size, a.augment) == ("tiny_cnn", False, "full", 32,
                                                                              "crop_flip")
    assert a.sources["trainable"] == "default" and a.input_kind == "images" and a.num_workers == 0
    assert "SMOKE_TEST" in a.run_flags and "REFERENCE" in a.run_flags          # tiny_cnn is for pipeline checks
    v = _args(synthetic_root, model={"backbone": "vit_b16_in21k", "trainable": "lora"})
    assert (v.pretrained, v.lora_rank, v.lora_alpha, v.lora_targets, v.augment, v.image_size) == \
           (True, 8, 8.0, ["q", "v"], "rrc_flip", 224)
    assert v.prompt_length is None and v.last_k is None and "SMOKE_TEST" not in v.run_flags
    p = _args(synthetic_root, model={"backbone": "clip_vitb16", "trainable": "prompt", "prompt_deep": True})
    assert (p.prompt_length, p.prompt_deep) == (10, True)


@pytest.mark.parametrize("model, match", [
    ({"backbone": "dinov2_vitb14"}, "missing model.trainable"),
    ({"backbone": "resnet32", "trainable": "lora"}, "trained in full"),
    ({"backbone": "resnet18_cifar", "pretrained": True}, "no pretrained weights"),
    ({"backbone": "vit_b16_in21k", "pretrained": False, "trainable": "full"}, "pretrained weights only"),
    ({"backbone": "vit_b16_in21k", "trainable": "lora", "prompt_length": 4}, "only applies to trainable prompt"),
    ({"backbone": "vit_b16_in21k", "trainable": "head", "lora_rank": 4}, "only applies to trainable lora"),
    ({"backbone": "vit_b16_in21k", "trainable": "last_k"}, "missing model.last_k"),
    ({"backbone": "vit_b16_in21k", "trainable": "last_k", "last_k": 0}, "must be 1 to 12"),
    ({"backbone": "clip_vitb16", "trainable": "head", "image_size": 112}, "takes 224 x 224"),
    ({"backbone": "tiny_cnn", "image_size": 8}, "must be >= 16"),
    ({"backbone": "vit_b16_in21k", "trainable": "lora", "lora_targets": ["q", "q"]}, "listed twice"),
    ({"backbone": "nonsense"}, "not one of"),
])
def test_model_settings_that_mean_nothing_are_refused(synthetic_root, model, match):
    with pytest.raises(ConfigError, match=match):
        _args(synthetic_root, model=model)


def test_each_algorithm_reads_one_input(synthetic_root):
    cfg = H.make_image_cfg(synthetic_root["root"], synthetic_root["results"], "FedAvg")
    cfg["features"] = {"backbone": "synthetic", "group": "pre"}
    with pytest.raises(ConfigError, match="reads no feature cache"):
        H.build_args(FedAvg, cfg, os.path.join(synthetic_root["results"], "a"))
    cfg = H.make_cfg(synthetic_root["root"], synthetic_root["results"], "SmokeFedAvgLinear", schedule=H.MULTI_SCHEDULE)
    for section, body in (("model", {"backbone": "tiny_cnn"}), ("run", {"num_workers": 2})):
        bad = {**cfg, section: {**cfg.get(section, {}), **body}}
        with pytest.raises(ConfigError, match="reads cached features"):
            H.build_args(SmokeFedAvgLinear, bad, os.path.join(synthetic_root["results"], "b"))


def test_momentum_needs_sgd(synthetic_root):
    with pytest.raises(ConfigError, match="optimizer sgd only"):
        _args(synthetic_root, hparams={"optimizer": "adam", "momentum": 0.9})
    a = _args(synthetic_root, hparams={"optimizer": "adamw", "weight_decay": 0.05})
    assert a.hparams["momentum"] == 0.0 and a.hparams["optimizer"] == "adamw"
    j = _args(synthetic_root, name="FedAvgJoint", cls=FedAvgJoint)
    assert "UPPER_BOUND" in j.run_flags and "loss_classes" not in j.hparams


def test_image_store_reads_the_index_rows_and_guards_test_rows(synthetic_root):
    pytest.importorskip("PIL")
    store = ImageStore("Synthetic", dataset_root=synthetic_root["root"])
    raw = H.load_dataset_raw_utils()
    reference = raw.SyntheticAdapter(num_classes=8, n_train=60, n_val=0, n_test=20, num_domains=3, image_size=32,
                                     seed=0, pattern="class")
    split = np.load(os.path.join(synthetic_root["root"], "Synthetic", "cl_index", "split.npy"))
    train_row, test_row = int(np.flatnonzero(split == 0)[7]), int(np.flatnonzero(split == 2)[3])
    assert np.array_equal(np.asarray(store.load_image(train_row)), np.asarray(reference.load_image(train_row)))
    with pytest.raises(TestDataAccessError):
        store.load_image(test_row)
    with pytest.raises(TestDataAccessError):
        store.labels([train_row, test_row])
    with pytest.raises(TestDataAccessError):                     # refused before any image is read
        store.train_batches(np.array([train_row, test_row]), 4, key=[0])
    with pytest.raises(TestDataAccessError):
        store.prediction_batches(np.array([test_row]))
    with store.evaluation():
        assert store.labels([test_row]).tolist() == [int(reference.label_and_domain(test_row)[0])]
        store.load_image(test_row)
    with pytest.raises(RuntimeError, match="set_pipeline"):
        store.train_batches(np.array([train_row]), 4, key=[0])


def test_augmentation_parameters_are_valid_and_fixed_by_the_generator():
    for h, w in ((32, 32), (224, 300), (500, 120)):
        for seed in range(50):
            top, left, ch, cw = random_resized_crop_params(h, w, np.random.default_rng(seed))
            assert 0 < ch <= h and 0 < cw <= w and 0 <= top <= h - ch and 0 <= left <= w - cw
            assert (top, left, ch, cw) == random_resized_crop_params(h, w, np.random.default_rng(seed))
    # the centre-crop fallback of torchvision for an image too extreme for any sampled crop
    assert random_resized_crop_params(10, 1000, np.random.default_rng(0), scale=(0.99, 1.0)) == (0, 493, 10, 13)
    assert random_crop_params(40, 40, 32, np.random.default_rng(1)) == random_crop_params(40, 40, 32,
                                                                                        np.random.default_rng(1))
    with pytest.raises(ValueError):
        random_crop_params(30, 40, 32, np.random.default_rng(0))
    flips = [flip_coin(np.random.default_rng(s)) for s in range(2000)]
    assert 0.45 < np.mean(flips) < 0.55


def test_listing_and_help_show_the_image_algorithms():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main.main(["--list_algorithms"]) == 0
        assert main.main(["--list_settings"]) == 0
    text = out.getvalue()
    assert "FedAvg   [multi-round, reads images (model section)]   flags: REFERENCE" in text
    assert "FedAvgJoint   [multi-round, reads images (model section)]   flags: UPPER_BOUND" in text
    assert "--loss_classes STR" in text and "--lr_schedule STR" in text
    assert "trainable" in text and "-tm" in text and "resnet18_cifar" in text and "lora_targets" in text
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main.main(["-algo", "FedAvg", "--help"])
    assert "hyperparameters of FedAvg" in out.getvalue() and "--optimizer STR" in out.getvalue()


def test_dry_run_of_an_image_algorithm_checks_the_images_and_writes_nothing(synthetic_root):
    pytest.importorskip("PIL")
    argv = ["-data", "Synthetic", "-partition", "dir", "-nc", "5", "-alpha", "0.5", "-cpt", "2",
            "-droot", synthetic_root["root"], "-algo", "FedAvg", "-m", "resnet18_cifar", "-rpt", "2", "-ls", "1",
            "-lbs", "16", "-lr", "0.05", "-dev", "cpu", "-rroot", synthetic_root["results"]]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main.main(argv + ["-t", "3", "--dry_run"]) == 0
    text = out.getvalue()
    assert "model       resnet18_cifar  from scratch" in text and "checked against the raw files" in text
    assert not os.path.exists(synthetic_root["results"])
    assert main.main(argv + ["--dry_run", "--verify_cache"]) == 2           # there is no cache to verify
    assert main.main(argv + ["-bb", "synthetic", "--dry_run"]) == 2         # features settings are refused
