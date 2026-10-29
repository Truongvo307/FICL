"""End-to-end tests of main.py and check_run.py on the synthetic dataset."""
import glob
import os

import numpy as np
import pytest

import helpers as H
import main
from check_run import check_experiment

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI_FLAGS = ("-rpt", "3", "-ls", "1", "-lbs", "16", "-lr", "0.01")


def _argv(sr, *extra, algo="SmokeClassMean"):
    return ["-data", "Synthetic", "-partition", "dir", "-nc", "5", "-alpha", "0.5", "-cpt", "2",
            "-droot", sr["root"], "-bb", "synthetic", "-fg", "pre", "-blocks", "3", "-tokens", "cls",
            "-algo", algo, "-dev", "cpu", "-rroot", sr["results"], *extra]


def _failed(checks):
    return [c for c in checks if not c[1]]


def test_dry_run_checks_every_seed_and_writes_nothing(synthetic_root):
    assert main.main(_argv(synthetic_root, "-t", "3", "--dry_run")) == 0
    assert not os.path.exists(synthetic_root["results"])
    assert main.main(_argv(synthetic_root, "-t", "4", "--dry_run")) == 2        # seed 3 was never generated
    assert not os.path.exists(synthetic_root["results"])


def test_single_round_end_to_end_with_three_seeds(synthetic_root):
    exp = os.path.join(synthetic_root["results"], "single")
    assert main.main(_argv(synthetic_root, "-t", "3", "-exp", exp)) == 0
    assert _failed(check_experiment(exp)) == []
    summary = H.read_json(os.path.join(exp, "avg", "summary.json"))
    assert summary["complete"] is True and summary["n_runs_complete"] == 3
    finals = [H.read_json(os.path.join(exp, f"times_{i}", "final.json"))["metrics"] for i in range(3)]
    for name, m in summary["metrics"].items():
        values = [f[name] for f in finals]
        assert m["values"] == values and m["n"] == 3, name
        assert abs(m["mean"] - np.mean(values)) <= 1e-9 * max(1.0, abs(np.mean(values))), name
        assert abs(m["sd"] - np.std(values, ddof=1)) <= 1e-9 * max(1.0, abs(m["sd"])), name
    # the class order differs between seeds, so A_avg varies and the ddof check above is not vacuous
    assert summary["metrics"]["acc_avg_incremental"]["sd"] > 0
    # nearest class mean ends with the centralized model for every client split and class order
    acc = summary["metrics"]["acc_final"]["values"]
    assert max(acc) - min(acc) < 1e-12
    runs = [H.read_json(os.path.join(exp, f"times_{i}", "run.json"))["scenario"] for i in range(3)]
    assert [r["config_name"] for r in runs] == [f"cl_5clients_dir_0.5_cpt2_seed{i}" for i in range(3)]
    assert len({r["content_sha256"] for r in runs}) == 3 and all(len(r["content_sha256"]) == 64 for r in runs)
    record = H.read_json(os.path.join(exp, "experiment.json"))
    assert record["regime"] == "single_round" and record["run_flags"] == ["SMOKE_TEST"]


def test_multi_round_end_to_end(synthetic_root):
    exp = os.path.join(synthetic_root["results"], "multi")
    argv = _argv(synthetic_root, *MULTI_FLAGS, "-eg", "1", "-t", "2", "-exp", exp, "-hp", "weight_decay=0.001",
                 algo="SmokeFedAvgLinear")
    assert main.main(argv) == 0
    assert _failed(check_experiment(exp)) == []
    run = H.read_json(os.path.join(exp, "times_0", "run.json"))
    assert run["rounds_per_task"] == 3 and run["total_rounds"] == 3 * run["scenario"]["num_tasks"]
    assert run["hparams"] == {"weight_decay": 0.001}
    args, _cls, _control = main.resolve_args(_argv(synthetic_root, *MULTI_FLAGS, "--weight_decay", "0.01",
                                                   algo="SmokeFedAvgLinear"))
    assert args.hparams == {"weight_decay": 0.01} and args.hparam_sources == {"weight_decay": "cli"}


def test_help_and_listings_show_settings_and_hyperparameters():
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stop:
        main.main(["-algo", "SmokeFedAvgLinear", "--help"])
    assert stop.value.code == 0
    assert "hyperparameters of SmokeFedAvgLinear" in out.getvalue() and "--weight_decay FLOAT" in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main.main(["--list_algorithms"]) == 0
        assert main.main(["--list_settings"]) == 0
    text = out.getvalue()
    assert "SmokeClassMean   [single-round (closed-form), reads cached features]" in text
    assert "(no hyperparameters)" in text
    assert "--weight_decay FLOAT" in text and "L2 penalty" in text
    assert "class_per_task" in text and "-cpt" in text and "rounds_per_task" in text


def test_automatic_folder_name_shows_scenario_algorithm_and_flags(synthetic_root):
    assert main.main(_argv(synthetic_root, "-go", "demo")) == 0
    pattern = os.path.join(synthetic_root["results"], "Synthetic", "cl_5clients_dir_0.5_cpt2",
                           "SmokeClassMean_SMOKE_TEST_synthetic-pre_demo_*")
    assert len(glob.glob(pattern)) == 1


def test_config_errors_exit_with_code_2(synthetic_root):
    assert main.main(_argv(synthetic_root, "-rpt", "5")) == 2                  # closed-form has no rounds
    assert main.main(_argv(synthetic_root, algo="NoSuchAlgo")) == 2
    assert main.main(_argv(synthetic_root, "-rpt", "3", "-ls", "1", "-lbs", "16", algo="SmokeFedAvgLinear")) == 2
    with pytest.raises(SystemExit):
        main.main(_argv(synthetic_root, "-gr", "100"))        # HtFL's total-round flag does not exist here
    assert not os.path.exists(synthetic_root["results"])


def test_existing_experiment_folder_is_never_reused(synthetic_root):
    exp = os.path.join(synthetic_root["results"], "once")
    assert main.main(_argv(synthetic_root, "-exp", exp)) == 0
    assert main.main(_argv(synthetic_root, "-exp", exp)) == 2


def test_checker_finds_a_missing_task_row(synthetic_root):
    exp = os.path.join(synthetic_root["results"], "tamper")
    assert main.main(_argv(synthetic_root, "-exp", exp)) == 0
    path = os.path.join(exp, "times_0", "tasks.csv")
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:-1])
    assert any("tasks.csv" in name for name, _ok, _detail in _failed(check_experiment(exp)))


def test_checker_rebuilds_counts_and_finds_an_edited_accuracy(synthetic_root):
    exp = os.path.join(synthetic_root["results"], "edit")
    assert main.main(_argv(synthetic_root, "-exp", exp)) == 0
    names = [name for name, _ok, _d in check_experiment(exp)]
    assert "run 0: every accuracy in acc_global.csv is a count over its test set" in names
    path = os.path.join(exp, "times_0", "acc_global.csv")
    with open(path) as f:
        text = f.read()
    header, first = text.splitlines()[:2]
    cells = first.split(",")
    cells[2] = "0.123457"                                     # not k / 40 for any whole k
    with open(path, "w") as f:
        f.write(text.replace(first, ",".join(cells), 1))
    failed = [name for name, _ok, _d in _failed(check_experiment(exp))]
    assert "run 0: every accuracy in acc_global.csv is a count over its test set" in failed


def test_a_failed_run_is_recorded_without_status_files(synthetic_root):
    from utils.result_utils import compute_averages

    exp = os.path.join(synthetic_root["results"], "fail")
    assert main.main(_argv(synthetic_root, "-t", "2", "-exp", exp)) == 0
    assert not glob.glob(os.path.join(exp, "times_*", "status.json"))
    assert not glob.glob(os.path.join(exp, "times_*", "counts.npz"))
    os.remove(os.path.join(exp, "times_1", "final.json"))                 # as if run 1 had failed
    summary = compute_averages(exp, {1: "RuntimeError: out of memory"})
    assert summary["n_runs_complete"] == 1 and summary["complete"] is False
    assert summary["runs"][1]["status"] == "failed" and "out of memory" in summary["runs"][1]["error"]
    assert summary["runs"][0]["status"] == "complete"
    assert compute_averages(exp)["runs"][1]["status"] == "incomplete"
    assert "run 1: complete (final.json written)" in [n for n, _ok, _d in _failed(check_experiment(exp))]


def test_shipped_configs_resolve():
    pytest.importorskip("yaml")
    paths = sorted(glob.glob(os.path.join(SYSTEM_DIR, "configs", "*.yaml")))
    assert paths
    for path in paths:
        args, _server_cls, _control = main.resolve_args(["-cfg", path])
        expected = "multi_round" if "multi" in os.path.basename(path) else "single_round"
        assert args.regime == expected, path
