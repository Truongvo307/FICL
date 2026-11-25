"""compare_reference.py: reading fed-mammoth logs, the agreement rule, the reference configs."""
import glob
import json
import os

import pytest

import compare_reference as cr
from utils.config_utils import read_yaml

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mammoth_log_gives_the_evaluation_after_the_last_task(tmp_path):
    log = tmp_path / "run.log"
    # progress bars use carriage returns; per-round evaluations of the last task come before its end-of-task one
    log.write_text("Mean accuracy up to task 1: 90.0 % Task accuracies: [90.0]\r[progress]\n"
                   "Mean accuracy up to task 2: 41.5 % Task accuracies: [0.0, 83.0]\n"
                   "Mean accuracy up to task 2: 42.25 % Task accuracies: [0.5, 84.0]\n"
                   "Total training time: 0:01:00\n")
    assert cr.read_mammoth(str(log), 2) == 42.25
    with pytest.raises(SystemExit, match="no evaluation after task 3"):
        cr.read_mammoth(str(log), 3)
    unfinished = tmp_path / "cut.log"
    unfinished.write_text("Mean accuracy up to task 2: 42.25 %\n")
    with pytest.raises(SystemExit, match="did not finish"):
        cr.read_mammoth(str(unfinished), 2)


def test_agreement_rule_is_max_of_two_points_and_two_sd():
    a = {"mean": 30.0, "sd": 0.5}
    assert cr.verdict(a, {"mean": 31.9, "sd": 0.4})[0]
    assert not cr.verdict(a, {"mean": 32.1, "sd": 0.4})[0]
    ok, gap, tol = cr.verdict(a, {"mean": 35.0, "sd": 3.0})
    assert ok and tol == 6.0 and gap == -5.0
    assert cr.verdict({"mean": 1.0, "sd": None}, {"mean": 2.5, "sd": None}) == (True, -1.5, 2.0)


def test_every_reference_config_has_an_entry_and_the_fed_mammoth_settings():
    with open(cr.REFERENCE_FILE) as f:
        entries = json.load(f)["entries"]
    paths = sorted(glob.glob(os.path.join(SYSTEM_DIR, "configs", "reference", "*.yaml")))
    assert paths
    for path in paths:
        key = os.path.splitext(os.path.basename(path))[0]
        assert key in entries, key
        values, hp = read_yaml(path)
        assert values["algorithm"] == "FedAvg" and values["model"] == "vit_b16_in21k"
        assert values["trainable"] == "full" and values["augment"] == "rrc_flip"
        assert (values["num_clients"], values["rounds_per_task"], values["local_epochs"], values["batch_size"]) == \
            (10, 5, 5, 16)
        assert values["class_order"] == "sorted" and values["min_train_per_client"] == 6
        assert values["local_learning_rate"] == 0.0003 and values["times"] == 3
        assert hp == {"optimizer": "adamw", "weight_decay": 0.0, "lr_schedule": "constant", "loss_classes": "task"}
