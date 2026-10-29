"""Tests for system/utils/config_utils.py: precedence, regime rules, hyperparameters, partitions."""
import pytest

from flcore.clients.clientbase import Client
from flcore.servers.serverbase import Server
from utils.config_utils import (
    MULTI_ROUND,
    SINGLE_ROUND,
    ConfigError,
    HParam,
    build_parser,
    config_from_dict,
    describe_protocol,
    finalize_config,
    merge_sources,
    parse_cli,
    parse_value,
)


class _Client(Client):
    def train_round(self, t, r):
        pass


class _Single(Server):
    REGIME = SINGLE_ROUND
    CLIENT_CLASS = _Client
    HPARAMS = {
        "lam": HParam("float", 1.0, "penalty"),
        "grid": HParam("float_list", [0.1, 1.0]),
        "mode": HParam("str", "a", choices=("a", "b")),
        "flag": HParam("bool", False),
        "steps": HParam("int", 3, low=1),
    }

    def aggregate(self, t, r, uploads):
        pass

    def predict(self, Z, seen):
        return seen[:1].repeat(len(Z))


class _Multi(_Single):
    REGIME = MULTI_ROUND
    HPARAMS = {}


DIR = {"dataset": "X", "partition": "dir", "num_clients": 4, "alpha": 0.5, "class_per_task": 2}
MULTI = {"rounds_per_task": 5, "local_epochs": 1, "batch_size": 8, "learning_rate": 0.1}


def _cfg(scenario=None, schedule=None, name="_Single", hparams=None, features=None, run=None):
    cfg = {"scenario": dict(DIR if scenario is None else scenario),
           "features": {"backbone": "b", "group": "g", **(features or {})},
           "algorithm": {"name": name}}
    if schedule is not None:
        cfg["schedule"] = dict(schedule)
    if hparams is not None:
        cfg["algorithm"]["hparams"] = dict(hparams)
    if run is not None:
        cfg["run"] = dict(run)
    return cfg


def _resolve(server_cls, cfg=None, argv=()):
    cli_values, cli_hp, _control = parse_cli(list(argv), server_cls.HPARAMS, server_cls.__name__)
    file_values, file_hp = config_from_dict(cfg, "yaml:test.yaml") if cfg is not None else ({}, {})
    values, sources, hp_raw, hp_src = merge_sources(cli_values, cli_hp, file_values, file_hp, "yaml:test.yaml")
    return finalize_config(values, sources, hp_raw, hp_src, server_cls)


def test_cli_overrides_yaml_and_sources_are_recorded():
    args = _resolve(_Single, _cfg(), ["-cpt", "3", "-t", "2"])
    assert args.class_per_task == 3 and args.sources["class_per_task"] == "cli"
    assert args.dataset == "X" and args.sources["dataset"] == "yaml:test.yaml"
    assert args.seed == 0 and args.sources["seed"] == "default"
    assert args.times == 2 and args.regime == SINGLE_ROUND and args.rounds_per_task == 1


def test_unknown_sections_and_keys_are_refused():
    with pytest.raises(ConfigError, match="unknown section"):
        config_from_dict({"scenrio": {}}, "yaml:x")
    with pytest.raises(ConfigError, match="scenario.clients"):
        config_from_dict({"scenario": {"clients": 4}}, "yaml:x")
    with pytest.raises(ConfigError, match="must be a mapping"):
        config_from_dict({"scenario": [1, 2]}, "yaml:x")


@pytest.mark.parametrize("key,value", [("rounds_per_task", 5), ("local_epochs", 1), ("batch_size", 32),
                                       ("learning_rate", 0.1), ("eval_gap", 5), ("join_ratio", 0.5)])
def test_single_round_refuses_round_settings(key, value):
    with pytest.raises(ConfigError, match="no meaning"):
        _resolve(_Single, _cfg(schedule={key: value}))


def test_single_round_on_the_command_line():
    with pytest.raises(ConfigError, match="no meaning"):
        _resolve(_Single, _cfg(), ["-rpt", "5"])
    assert _resolve(_Single, _cfg(), ["-rpt", "1"]).rounds_per_task == 1


@pytest.mark.parametrize("missing", ["rounds_per_task", "local_epochs", "batch_size", "learning_rate"])
def test_multi_round_requires_its_schedule(missing):
    schedule = {k: v for k, v in MULTI.items() if k != missing}
    with pytest.raises(ConfigError, match=f"schedule.{missing}"):
        _resolve(_Multi, _cfg(schedule=schedule, name="_Multi"))


def test_multi_round_defaults_and_ranges():
    args = _resolve(_Multi, _cfg(schedule=MULTI, name="_Multi"))
    assert args.rounds_per_task == 5 and args.join_ratio == 1.0 and args.eval_gap == 0
    for key, bad in (("join_ratio", 0.0), ("join_ratio", 1.5), ("rounds_per_task", 0), ("learning_rate", -0.1),
                     ("eval_gap", -1), ("batch_size", 0), ("local_epochs", 0)):
        with pytest.raises(ConfigError):
            _resolve(_Multi, _cfg(schedule={**MULTI, key: bad}, name="_Multi"))


def test_declared_regime_must_match_the_algorithm():
    with pytest.raises(ConfigError, match="multi_round"):
        _resolve(_Single, _cfg(schedule={"regime": "multi_round"}))
    with pytest.raises(ConfigError, match="single_round"):
        _resolve(_Multi, _cfg(schedule={"regime": "single_round", **MULTI}, name="_Multi"))
    assert _resolve(_Single, _cfg(schedule={"regime": "single_round"})).regime == SINGLE_ROUND


def test_hparams_defaults_parsing_and_errors():
    args = _resolve(_Single, _cfg())
    assert args.hparams == {"lam": 1.0, "grid": [0.1, 1.0], "mode": "a", "flag": False, "steps": 3}
    args = _resolve(_Single, _cfg(hparams={"lam": 0.5, "grid": [1, 2]}),
                    ["-hp", "lam=1e-3", "-hp", "flag=true", "-hp", "mode=b"])
    assert args.hparams["lam"] == 1e-3 and args.hparam_sources["lam"] == "cli"
    assert args.hparams["grid"] == [1.0, 2.0] and args.hparam_sources["grid"] == "yaml:test.yaml"
    assert args.hparams["flag"] is True and args.hparams["mode"] == "b"
    assert _resolve(_Single, _cfg(), ["-hp", "grid=0.1,1,10"]).hparams["grid"] == [0.1, 1.0, 10.0]
    with pytest.raises(ConfigError, match="unknown hyperparameter"):
        _resolve(_Single, _cfg(), ["-hp", "lamda=1"])
    with pytest.raises(ConfigError, match="mode"):
        _resolve(_Single, _cfg(), ["-hp", "mode=c"])
    with pytest.raises(ConfigError, match="steps"):
        _resolve(_Single, _cfg(), ["-hp", "steps=2.5"])
    with pytest.raises(ConfigError, match="twice"):
        _resolve(_Single, _cfg(), ["-hp", "lam=1", "-hp", "lam=2"])
    with pytest.raises(ConfigError, match="NAME=VALUE"):
        _resolve(_Single, _cfg(), ["-hp", "lam"])


def test_hyperparameters_are_flags_with_help_and_ranges():
    args = _resolve(_Single, _cfg(hparams={"lam": 0.5}), ["--lam", "1e-3", "--flag", "true", "--grid", "1,2"])
    assert args.hparams["lam"] == 1e-3 and args.hparam_sources["lam"] == "cli"
    assert args.hparams["flag"] is True and args.hparams["grid"] == [1.0, 2.0]
    with pytest.raises(ConfigError, match="twice"):
        _resolve(_Single, _cfg(), ["--lam", "1", "-hp", "lam=2"])
    with pytest.raises(ConfigError, match="lowest allowed"):
        _resolve(_Single, _cfg(), ["--steps", "0"])
    with pytest.raises(SystemExit):                                   # an unknown flag is an argparse error
        _resolve(_Single, _cfg(), ["--lamda", "1"])
    text = build_parser(_Single.HPARAMS, "_Single").format_help()
    assert "hyperparameters of _Single" in text and "--lam FLOAT" in text and "penalty (float; default 1.0)" in text
    assert "--steps INT" in text and ">= 1" in text and "one of a, b" in text
    assert "--list_algorithms" in build_parser().format_help()


def test_hyperparameter_declarations_are_checked():
    with pytest.raises(ValueError, match="low / high"):
        HParam("str", "a", low=0)
    with pytest.raises(ValueError):
        HParam("int", 0, low=1)                                       # the default must satisfy the range

    class _Clash(_Single):
        HPARAMS = {"seed": HParam("int", 1)}

    class _BadName(_Single):
        HPARAMS = {"Lam": HParam("float", 1.0)}

    with pytest.raises(TypeError, match="taken by a general setting"):
        _Clash.check_contract()
    with pytest.raises(TypeError, match="lower case"):
        _BadName.check_contract()


def test_partition_rules():
    with pytest.raises(ConfigError, match="scenario.alpha"):
        _resolve(_Single, _cfg(scenario={k: v for k, v in DIR.items() if k != "alpha"}))
    with pytest.raises(ConfigError, match="scenario.num_clients"):
        _resolve(_Single, _cfg(scenario={k: v for k, v in DIR.items() if k != "num_clients"}))
    with pytest.raises(ConfigError, match="no meaning for partition iid"):
        _resolve(_Single, _cfg(scenario={**DIR, "partition": "iid"}))
    natural = {"dataset": "X", "partition": "natural", "class_per_task": 2, "num_clients": 6}
    args = _resolve(_Single, _cfg(scenario=natural))
    assert args.num_clients is None and any("natural" in n for n in args.notes)
    with pytest.raises(ConfigError, match="partition"):
        _resolve(_Single, _cfg(scenario={**DIR, "partition": "pat"}))


def test_block_and_token_lists():
    args = _resolve(_Single, _cfg(), ["-blocks", "6,7,8", "-tokens", "cls,mean"])
    assert args.blocks == [6, 7, 8] and args.tokens == ["cls", "mean"]
    assert _resolve(_Single, _cfg(features={"blocks": [11]}), ["-blocks", "all"]).blocks is None
    with pytest.raises(ConfigError, match="tokens"):
        _resolve(_Single, _cfg(), ["-tokens", "max"])
    with pytest.raises(ConfigError, match="blocks"):
        _resolve(_Single, _cfg(), ["-blocks", "1,1"])
    with pytest.raises(ConfigError, match="proj_seed"):
        _resolve(_Single, _cfg(features={"proj_seed": 3}))


def test_parse_value_types():
    assert parse_value("int", "3", "x") == 3 and parse_value("int", 3.0, "x") == 3
    assert parse_value("float", "1e-3", "x") == 1e-3 and parse_value("float", 2, "x") == 2.0
    assert parse_value("bool", "no", "x") is False and parse_value("bool", True, "x") is True
    assert parse_value("str", 0, "x") == "0"
    for kind, bad in (("int", "3.5"), ("int", True), ("float", "abc"), ("bool", "maybe"), ("str", True)):
        with pytest.raises(ConfigError):
            parse_value(kind, bad, "x")


def test_yaml_file_with_scientific_notation(tmp_path):
    pytest.importorskip("yaml")
    from utils.config_utils import read_yaml
    path = tmp_path / "c.yaml"
    path.write_text("scenario: {dataset: X, partition: dir, num_clients: 4, alpha: 0.5, class_per_task: 2}\n"
                    "features: {backbone: b, group: g}\n"
                    "schedule: {rounds_per_task: 5, local_epochs: 1, batch_size: 8, learning_rate: 1e-3}\n"
                    "algorithm: {name: _Multi}\n")
    values, _hp = read_yaml(str(path))
    assert values["local_learning_rate"] == 1e-3   # PyYAML reads 1e-3 as a string; it is parsed as a float


def test_describe_protocol_mentions_rounds_only_for_multi_round():
    plan = {"num_tasks": 4, "num_classes": 8, "task_sizes": [2, 2, 2, 2], "num_clients": 4, "feature_dim": 16,
            "blocks": [3], "tokens": ["cls"], "runs": [{"run": 0, "seed": 0, "scenario": "cl_x_seed0"}]}
    multi = "\n".join(describe_protocol(_resolve(_Multi, _cfg(schedule={**MULTI, "eval_gap": 5}, name="_Multi")), plan))
    assert "5 rounds per task" in multi and "20 rounds in total" in multi
    single = "\n".join(describe_protocol(_resolve(_Single, _cfg()), plan))
    assert "no rounds" in single and "rounds per task" not in single
