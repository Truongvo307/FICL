"""Settings of FICL: scenario, features or model, schedule, algorithm hyperparameters, run.

main.py reads them in three layers; a later layer overrides an earlier one:
    1. defaults built into this file
    2. a YAML file given with -cfg
    3. command-line flags (HtFL-style short flags)
Every resolved value remembers where it came from (args.sources, written to experiment.json).

YAML sections and what they control
    scenario   which data arrives when. Must match a folder written by dataset/generate_scenario.py:
               dataset, partition (dir | iid | natural), num_clients, alpha, class_per_task (the class
               increment: tasks = ceil(classes / class_per_task)), use_val, min_train_per_client,
               dataset_root
    features   algorithms on cached features of a frozen backbone (INPUT = "features"): backbone, group,
               blocks, tokens, proj_rank, proj_seed
    model      algorithms that train a network on images (INPUT = "images"): backbone (tiny_cnn,
               resnet18_cifar, resnet32 from scratch; vit_b16_in21k, dinov2_vitb14, clip_vitb16
               pretrained), pretrained, trainable (full | head | last_k | lora | prompt), last_k,
               lora_rank, lora_alpha, lora_targets, prompt_length, prompt_deep, image_size, augment.
               An algorithm reads either features or images: the other section is refused.
    schedule   how time runs inside a task. The algorithm's server class fixes the regime:
               single_round (closed-form): every client uploads once per task. There are no rounds,
                   so rounds_per_task, local_epochs, batch_size, learning_rate, join_ratio and
                   eval_gap must not be set (rounds_per_task: 1 is accepted).
               multi_round: rounds_per_task (a new task arrives every rounds_per_task rounds),
                   local_epochs, batch_size, learning_rate (all four required), join_ratio
                   (default 1.0), eval_gap (curve points every eval_gap rounds; default 0 = off).
               regime: optional statement of the regime you expect; refused if the algorithm differs.
    algorithm  name, plus hparams: the algorithm's own hyperparameters, declared by its server class
               in HPARAMS (unknown names are refused). On the command line: -hp NAME=VALUE.
    run        seed (run i uses scenario seed seed + i), times, goal, device, device_id,
               results_root, exp_dir; image algorithms also num_workers and eval_batch_size

Settings that mean nothing for the chosen algorithm are refused, never silently ignored.

Where settings live
    general settings      OPTIONS below: one row per setting (attribute, YAML section and key, flags, type,
                          default, help). python main.py --list_settings prints them all.
    algorithm settings    HPARAMS in each algorithm's server class, {name: HParam(kind, default, help, ...)}.
                          They become --NAME flags (python main.py -algo NAME --help lists them, and
                          python main.py --list_algorithms lists every algorithm's), or YAML
                          algorithm.hparams; the server and its clients read them as self.hp["NAME"].
"""
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from utils.model_catalog import (
    AUGMENTATIONS,
    DEFAULT_AUGMENT,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGETS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PROMPT_LENGTH,
    LORA_TARGETS,
    MIN_IMAGE_SIZE,
    NETWORKS,
    TRAINABLE_MODES,
    describe_networks,
)

SINGLE_ROUND = "single_round"
MULTI_ROUND = "multi_round"
REGIMES = (SINGLE_ROUND, MULTI_ROUND)
REGIME_TEXT = {SINGLE_ROUND: "single-round (closed-form)", MULTI_ROUND: "multi-round"}
FEATURES, IMAGES = "features", "images"
INPUTS = (FEATURES, IMAGES)
PARTITIONS = ("dir", "iid", "natural")
SECTIONS = ("scenario", "features", "model", "schedule", "algorithm", "run")
SCHEDULE_KEYS = ("rounds_per_task", "local_epochs", "batch_size", "local_learning_rate", "join_ratio", "eval_gap")
SGD_SCHEDULE = ("rounds_per_task", "local_epochs", "batch_size", "local_learning_rate")
TOKEN_NAMES = ("cls", "mean")
KINDS = ("int", "float", "str", "bool", "int_list", "float_list", "str_list")
NUMBER_KINDS = ("int", "float", "int_list", "float_list")
CONTROL_NAMES = ("config", "hparam", "dry_run", "verify_cache", "list_algorithms", "list_settings", "help")
HPARAM_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ConfigError(ValueError):
    """A setting is missing, unknown, of the wrong type, or contradicts the algorithm's regime."""


# ---------------------------------------------------------------------------
# Typed values
# ---------------------------------------------------------------------------
def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise ConfigError(f"expected a boolean (true/false), got {value!r}")


def _parse_scalar(kind: str, value, name: str):
    if kind == "int":
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    if kind == "float":
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            out = float(value)
        elif isinstance(value, str):
            try:
                out = float(value.strip())
            except ValueError:
                raise ConfigError(f"{name}: expected a number, got {value!r}") from None
        else:
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        if not math.isfinite(out):
            raise ConfigError(f"{name}: expected a finite number, got {value!r}")
        return out
    if kind == "bool":
        if isinstance(value, (bool, str)):
            try:
                return str2bool(value)
            except ConfigError:
                raise ConfigError(f"{name}: expected true or false, got {value!r}") from None
        raise ConfigError(f"{name}: expected true or false, got {value!r}")
    if kind == "str":
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected text, got the boolean {value!r} (quote it in YAML)")
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        raise ConfigError(f"{name}: expected text, got {value!r}")
    raise ConfigError(f"{name}: unknown kind {kind!r}")


def parse_value(kind: str, value, name: str, choices=None):
    """Typed value of a setting. None stays None; for list kinds, 'all' means None (everything stored)."""
    if kind not in KINDS:
        raise ConfigError(f"{name}: unknown kind {kind!r}")
    if value is None:
        return None
    if kind.endswith("_list"):
        if isinstance(value, str) and value.strip().lower() == "all":
            return None
        base = kind[:-5]
        if isinstance(value, str):
            items = [s.strip() for s in value.split(",") if s.strip() != ""]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        if not items:
            raise ConfigError(f"{name}: empty list")
        out = [_parse_scalar(base, v, name) for v in items]
        if choices is not None:
            bad = [v for v in out if v not in choices]
            if bad:
                raise ConfigError(f"{name}: {bad} not in {list(choices)}")
        return out
    out = _parse_scalar(kind, value, name)
    if choices is not None and out not in choices:
        raise ConfigError(f"{name}: {out!r} is not one of {list(choices)}")
    return out


@dataclass(frozen=True)
class HParam:
    """One hyperparameter of an algorithm, declared in its server class: HPARAMS = {name: HParam(...)}.

        kind       int | float | str | bool | int_list | float_list | str_list
        default    the value used when neither the YAML file nor the command line sets it
        help       what it controls; shown by  python main.py -algo NAME --help  and  --list_algorithms
        choices    allowed values (or None for any)
        low, high  inclusive bounds for int and float kinds (each element for lists), or None

    Command line: --NAME VALUE (lists comma-separated, booleans true / false). YAML: algorithm.hparams.
    In code: self.hp["NAME"], on the server and on every client.
    """

    kind: str
    default: Any
    help: str = ""
    choices: tuple | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"HParam kind must be one of {KINDS}, got {self.kind!r}")
        if (self.low is not None or self.high is not None) and self.kind not in NUMBER_KINDS:
            raise ValueError(f"low / high apply to number kinds {NUMBER_KINDS}, not {self.kind!r}")
        self.parse(self.default, "default")

    def parse(self, value, name: str):
        out = parse_value(self.kind, value, name, self.choices)
        for v in (out if isinstance(out, list) else [out]):
            if v is None:
                continue
            if self.low is not None and v < self.low:
                raise ConfigError(f"{name}: {v} is below the lowest allowed value {self.low:g}")
            if self.high is not None and v > self.high:
                raise ConfigError(f"{name}: {v} is above the highest allowed value {self.high:g}")
        return out

    def describe(self) -> str:
        """One line for help texts: what it is, then its type, default, choices and range."""
        facts = [self.kind, f"default {self.default}"]
        if self.choices:
            facts.append("one of " + ", ".join(str(c) for c in self.choices))
        if self.low is not None and self.high is not None:
            facts.append(f"{self.low:g} to {self.high:g}")
        elif self.low is not None:
            facts.append(f">= {self.low:g}")
        elif self.high is not None:
            facts.append(f"<= {self.high:g}")
        text = "; ".join(facts)
        return f"{self.help} ({text})" if self.help else f"({text})"


# ---------------------------------------------------------------------------
# The settings table: one row per setting, used for the YAML keys, the flags and the help text
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Option:
    dest: str                    # attribute name on args (HtFL-style where HtFL has one)
    section: str                 # YAML section
    key: str                     # key inside the YAML section
    flags: tuple                 # command-line flags
    kind: str
    default: Any = None
    help: str = ""
    choices: tuple | None = None


OPTIONS = (
    # scenario: which data arrives when
    Option("dataset", "scenario", "dataset", ("-data", "--dataset"), "str", None,
           "dataset folder under dataset/, e.g. TissueMNIST"),
    Option("partition", "scenario", "partition", ("-partition", "--partition"), "str", None,
           "dir | iid | natural", PARTITIONS),
    Option("num_clients", "scenario", "num_clients", ("-nc", "--num_clients"), "int", None,
           "clients for dir and iid; natural takes it from the scenario"),
    Option("data_alpha", "scenario", "alpha", ("-alpha", "--data_alpha"), "float", None,
           "Dirichlet alpha (dir only)"),
    Option("class_per_task", "scenario", "class_per_task", ("-cpt", "--class_per_task"), "int", None,
           "classes per task, the class increment; tasks = ceil(classes / cpt)"),
    Option("use_val", "scenario", "use_val", ("-useval", "--use_val"), "str", "ignore",
           "official validation split: ignore | merge", ("ignore", "merge")),
    Option("min_train_per_client", "scenario", "min_train_per_client", ("-mintr", "--min_train_per_client"),
           "int", 0, "as given to generate_scenario.py"),
    Option("dataset_root", "scenario", "dataset_root", ("-droot", "--dataset_root"), "str",
           os.path.join("..", "dataset"), "folder that holds the dataset folders"),
    # features: which cached features a run reads
    Option("backbone", "features", "backbone", ("-bb", "--backbone"), "str", None,
           "feature cache backbone, e.g. dinov2_vitb14"),
    Option("feature_group", "features", "group", ("-fg", "--feature_group"), "str", None,
           "cache group, e.g. pre or last_ln"),
    Option("blocks", "features", "blocks", ("-blocks", "--blocks"), "int_list", None,
           "stored block ids to use, e.g. 11 or 6,7,8,9,10,11 (default: all stored)"),
    Option("tokens", "features", "tokens", ("-tokens", "--tokens"), "str_list", None,
           "cls, mean or cls,mean (default: all stored)", TOKEN_NAMES),
    Option("proj_rank", "features", "proj_rank", ("-prank", "--proj_rank"), "int", None,
           "fixed random projection of every block to this many dims"),
    Option("proj_seed", "features", "proj_seed", ("-pseed", "--proj_seed"), "int", None,
           "projection seed (default: derived from the run seed)"),
    # model: the network an image algorithm trains (all None until finalize_config fills the defaults)
    Option("model", "model", "backbone", ("-m", "--model"), "str", None,
           "network trained on images: tiny_cnn, resnet18_cifar, resnet32 (from scratch) or vit_b16_in21k, "
           "dinov2_vitb14, clip_vitb16 (pretrained)", tuple(NETWORKS)),
    Option("pretrained", "model", "pretrained", ("-pre", "--pretrained"), "bool", None,
           "start from pretrained weights (default: true for the ViTs, false for the CNNs, which have none)"),
    Option("trainable", "model", "trainable", ("-tm", "--trainable"), "str", None,
           "what is trained: full | head | last_k | lora | prompt (required for a pretrained network; "
           "a network from scratch trains in full)", TRAINABLE_MODES),
    Option("last_k", "model", "last_k", ("-lastk", "--last_k"), "int", None,
           "trainable last_k: how many of the last transformer blocks are trained (with the final norm and the head)"),
    Option("lora_rank", "model", "lora_rank", ("-lrank", "--lora_rank"), "int", None,
           f"trainable lora: rank r of every LoRA update (default {DEFAULT_LORA_RANK})"),
    Option("lora_alpha", "model", "lora_alpha", ("-lalpha", "--lora_alpha"), "float", None,
           "trainable lora: the update is W + (alpha / r) B A (default alpha = r)"),
    Option("lora_targets", "model", "lora_targets", ("-ltgt", "--lora_targets"), "str_list", None,
           f"trainable lora: weights adapted in every block, from {','.join(LORA_TARGETS)} "
           f"(default {','.join(DEFAULT_LORA_TARGETS)})", LORA_TARGETS),
    Option("prompt_length", "model", "prompt_length", ("-plen", "--prompt_length"), "int", None,
           f"trainable prompt: number of learnable prompt tokens (default {DEFAULT_PROMPT_LENGTH})"),
    Option("prompt_deep", "model", "prompt_deep", ("-pdeep", "--prompt_deep"), "bool", None,
           "trainable prompt: new prompt tokens before every block (deep) instead of before the first only "
           "(shallow; default false)"),
    Option("image_size", "model", "image_size", ("-isize", "--image_size"), "int", None,
           "input size in pixels (CNNs: default 32; the ViTs take 224 only)"),
    Option("augment", "model", "augment", ("-aug", "--augment"), "str", None,
           "training augmentation: none | flip | crop_flip | rrc_flip (default: crop_flip for CNNs, rrc_flip "
           "for ViTs)", AUGMENTATIONS),
    # schedule: how time runs inside a task
    Option("regime", "schedule", "regime", ("-regime", "--regime"), "str", None,
           "the regime you expect: single_round | multi_round (must match the algorithm)", REGIMES),
    Option("rounds_per_task", "schedule", "rounds_per_task", ("-rpt", "--rounds_per_task"), "int", None,
           "multi-round: rounds per task; a new task arrives every rpt rounds"),
    Option("local_epochs", "schedule", "local_epochs", ("-ls", "--local_epochs"), "int", None,
           "multi-round: local epochs per round"),
    Option("batch_size", "schedule", "batch_size", ("-lbs", "--batch_size"), "int", None,
           "multi-round: local batch size"),
    Option("local_learning_rate", "schedule", "learning_rate", ("-lr", "--local_learning_rate"), "float", None,
           "multi-round: local learning rate"),
    Option("join_ratio", "schedule", "join_ratio", ("-jr", "--join_ratio"), "float", None,
           "multi-round: fraction of clients per round (default 1.0)"),
    Option("eval_gap", "schedule", "eval_gap", ("-eg", "--eval_gap"), "int", None,
           "multi-round: curve point every eg rounds (default 0 = off; curves never choose anything)"),
    # algorithm
    Option("algorithm", "algorithm", "name", ("-algo", "--algorithm"), "str", None,
           "algorithm name (python main.py --list_algorithms)"),
    # run
    Option("seed", "run", "seed", ("-seed", "--seed"), "int", 0,
           "first seed; run i uses scenario seed seed + i"),
    Option("times", "run", "times", ("-t", "--times"), "int", 1, "number of runs (seeds)"),
    Option("goal", "run", "goal", ("-go", "--goal"), "str", "test", "free label, part of the result folder name"),
    Option("device", "run", "device", ("-dev", "--device"), "str", "cuda",
           "cuda | cpu (used only by algorithms that use torch)", ("cuda", "cpu")),
    Option("device_id", "run", "device_id", ("-did", "--device_id"), "str", "0", "CUDA_VISIBLE_DEVICES"),
    Option("results_root", "run", "results_root", ("-rroot", "--results_root"), "str",
           os.path.join("..", "results"), "root of the result folders"),
    Option("exp_dir", "run", "exp_dir", ("-exp", "--exp_dir"), "str", None,
           "explicit result folder for this experiment (must not exist yet)"),
    Option("num_workers", "run", "num_workers", ("-nw", "--num_workers"), "int", None,
           f"image algorithms: worker processes that decode images (default {DEFAULT_NUM_WORKERS}; 0 = none); "
           f"results do not depend on it"),
    Option("eval_batch_size", "run", "eval_batch_size", ("-ebs", "--eval_batch_size"), "int", None,
           f"image algorithms: images per evaluation forward pass (default {DEFAULT_EVAL_BATCH_SIZE})"),
)
FEATURE_DESTS = ("backbone", "feature_group", "blocks", "tokens", "proj_rank", "proj_seed")
MODEL_DESTS = ("model", "pretrained", "trainable", "last_k", "lora_rank", "lora_alpha", "lora_targets",
               "prompt_length", "prompt_deep", "image_size", "augment")
IMAGE_RUN_DESTS = ("num_workers", "eval_batch_size")
MODE_KEYS = {"last_k": ("last_k",), "lora": ("lora_rank", "lora_alpha", "lora_targets"),
             "prompt": ("prompt_length", "prompt_deep")}
OPTION_BY_DEST = {o.dest: o for o in OPTIONS}
OPTION_BY_KEY = {(o.section, o.key): o for o in OPTIONS}


def describe(dest: str) -> str:
    o = OPTION_BY_DEST[dest]
    return f"{o.section}.{o.key} ({o.flags[0]})"


# ---------------------------------------------------------------------------
# Command line and YAML
# ---------------------------------------------------------------------------
def reserved_names() -> set:
    """Names an algorithm hyperparameter may not take: every general setting and flag, and the control flags."""
    names = set(CONTROL_NAMES)
    for o in OPTIONS:
        names.add(o.dest)
        names.update(f.lstrip("-") for f in o.flags)
    return names


def check_hparam_names(hparams: dict, owner: str) -> None:
    """Hyperparameter names must be lower-case identifiers and must not shadow a general setting or flag."""
    reserved = reserved_names()
    for name in hparams:
        if not HPARAM_NAME.match(name):
            raise TypeError(f"{owner}: hyperparameter name {name!r} must be lower case: letters, digits, _")
        if name in reserved:
            raise TypeError(f"{owner}: hyperparameter name {name!r} is taken by a general setting or flag "
                            f"(--{name}); choose another name")


def build_parser(hparams: dict | None = None, algorithm: str | None = None) -> argparse.ArgumentParser:
    """The command line: general settings by YAML section, the algorithm's hyperparameters, control flags."""
    if hparams:
        epilog = f"{algorithm}'s hyperparameters are listed above; YAML: algorithm.hparams."
    else:
        epilog = ("Algorithm hyperparameters: add -algo NAME (or -cfg FILE) to this command to see that "
                  "algorithm's, or run  python main.py --list_algorithms.")
    p = argparse.ArgumentParser(
        prog="main.py", argument_default=argparse.SUPPRESS, allow_abbrev=False, epilog=epilog,
        description="FICL: federated class-incremental learning on cached features or on images. Settings "
                    "come from the built-in defaults, then a YAML file (-cfg), then these flags. Algorithms on "
                    "cached features use the features section, algorithms that train a network the model section.")
    p.add_argument("-cfg", "--config", dest="config", metavar="FILE",
                   help="YAML file with the sections scenario, features or model, schedule, algorithm, run")
    groups = {s: p.add_argument_group(s) for s in SECTIONS}
    for o in OPTIONS:
        default = "" if o.default is None else f"; default {o.default}"
        groups[o.section].add_argument(*o.flags, dest=o.dest, metavar=o.key.upper(),
                                       help=f"{o.help} [{o.section}.{o.key}{default}]".replace("%", "%%"))
    groups["algorithm"].add_argument("-hp", "--hparam", dest="hparam", action="append", metavar="NAME=VALUE",
                                     help="algorithm hyperparameter, same as --NAME VALUE (repeatable)")
    if hparams:
        g = p.add_argument_group(f"hyperparameters of {algorithm}",
                                 "set with --NAME VALUE, or in the YAML file under algorithm.hparams")
        for name, spec in hparams.items():
            g.add_argument(f"--{name}", dest=f"hp__{name}", metavar=spec.kind.upper(),
                           help=spec.describe().replace("%", "%%"))
    ctl = p.add_argument_group("control")
    ctl.add_argument("--dry_run", dest="dry_run", action="store_true",
                     help="resolve the settings, check the data of every seed, print the protocol, run nothing")
    ctl.add_argument("--verify_cache", dest="verify_cache", action="store_true",
                     help="also recompute the sha256 of the feature cache (reads the whole file)")
    ctl.add_argument("--list_algorithms", dest="list_algorithms", action="store_true",
                     help="list the registered algorithms, their regimes and their hyperparameters")
    ctl.add_argument("--list_settings", dest="list_settings", action="store_true",
                     help="list every general setting: YAML key, flag, default, meaning")
    return p


def find_algorithm(argv) -> tuple[str | None, str | None]:
    """(algorithm, device_id) named on the command line, else in the YAML file of -cfg; None when absent.

    main.py needs the algorithm before parsing the rest (its hyperparameters become flags) and the
    device id before importing it (CUDA_VISIBLE_DEVICES must be set before torch is imported)."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("-algo", "--algorithm", dest="algorithm")
    pre.add_argument("-did", "--device_id", dest="device_id")
    pre.add_argument("-cfg", "--config", dest="config")
    ns, _rest = pre.parse_known_args(list(argv))
    algorithm, device_id = ns.algorithm, ns.device_id
    if ns.config and (algorithm is None or device_id is None):
        values, _hp = read_yaml(ns.config)
        algorithm = algorithm if algorithm is not None else values.get("algorithm")
        device_id = device_id if device_id is not None else values.get("device_id")
    return algorithm, device_id


def parse_cli(argv, hparams: dict | None = None, algorithm: str | None = None) -> tuple[dict, dict, dict]:
    """Only the flags that were given: (values by dest, raw hyperparameters, control flags)."""
    ns = vars(build_parser(hparams, algorithm).parse_args(list(argv)))
    control = {"config": ns.pop("config", None), "dry_run": bool(ns.pop("dry_run", False)),
               "verify_cache": bool(ns.pop("verify_cache", False)),
               "list_algorithms": bool(ns.pop("list_algorithms", False)),
               "list_settings": bool(ns.pop("list_settings", False))}
    hp_items = ns.pop("hparam", None) or []
    hparams_raw = {key[4:]: ns.pop(key) for key in [k for k in ns if k.startswith("hp__")]}
    values = {}
    for dest, raw in ns.items():
        o = OPTION_BY_DEST[dest]
        values[dest] = parse_value(o.kind, raw, o.flags[0], o.choices)
    for item in hp_items:
        if "=" not in item:
            raise ConfigError(f"-hp expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ConfigError(f"-hp expects NAME=VALUE, got {item!r}")
        if name in hparams_raw:
            raise ConfigError(f"hyperparameter {name} is given twice on the command line")
        hparams_raw[name] = value
    return values, hparams_raw, control


def settings_table() -> list[str]:
    """Every general setting by YAML section, for --list_settings."""
    lines = [f"  {'YAML key':21s} {'flag':11s} {'default':11s} meaning"]
    for section in SECTIONS:
        lines.append(f"{section}:")
        for o in OPTIONS:
            if o.section != section:
                continue
            default = "-" if o.default is None else str(o.default)
            lines.append(f"  {o.key:21s} {o.flags[0]:11s} {default:11s} {o.help}")
        if section == "algorithm":
            lines.append(f"  {'hparams':21s} {'--NAME':11s} {'per algo':11s} the algorithm's own hyperparameters "
                         f"(python main.py --list_algorithms)")
        if section == "model":
            lines.append("  networks (model.backbone):")
            lines.extend(f"    {line}" for line in describe_networks())
    return lines


def config_from_dict(cfg: dict, origin: str = "config") -> tuple[dict, dict]:
    """(values by dest, raw hyperparameters) from a dict with the YAML sections."""
    if cfg is None:
        return {}, {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{origin}: the top level must be a mapping of sections {list(SECTIONS)}")
    values, hparams = {}, {}
    for section, body in cfg.items():
        if section not in SECTIONS:
            raise ConfigError(f"{origin}: unknown section {section!r}; sections are {list(SECTIONS)}")
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigError(f"{origin}: section {section} must be a mapping")
        for key, raw in body.items():
            if section == "algorithm" and key == "hparams":
                if raw is None:
                    continue
                if not isinstance(raw, dict):
                    raise ConfigError(f"{origin}: algorithm.hparams must be a mapping")
                hparams.update(raw)
                continue
            o = OPTION_BY_KEY.get((section, key))
            if o is None:
                allowed = sorted(k for (s, k) in OPTION_BY_KEY if s == section)
                if section == "algorithm":
                    allowed.append("hparams")
                raise ConfigError(f"{origin}: unknown key {section}.{key}; allowed in {section}: {allowed}")
            values[o.dest] = parse_value(o.kind, raw, f"{section}.{key}", o.choices)
    return values, hparams


def read_yaml(path: str) -> tuple[dict, dict]:
    try:
        import yaml
    except ImportError:
        raise ConfigError("PyYAML is needed for -cfg (pip install pyyaml)") from None
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return config_from_dict(cfg or {}, f"yaml:{path}")


def merge_sources(cli_values: dict, cli_hparams: dict, file_values: dict | None = None,
                  file_hparams: dict | None = None, file_origin: str = "yaml"):
    """Defaults < file < command line. Returns (values, sources, raw hparams, hparam sources)."""
    values = {o.dest: o.default for o in OPTIONS}
    sources = {o.dest: "default" for o in OPTIONS}
    for dest, v in (file_values or {}).items():
        values[dest], sources[dest] = v, file_origin
    for dest, v in cli_values.items():
        values[dest], sources[dest] = v, "cli"
    hparams, hparam_sources = {}, {}
    for k, v in (file_hparams or {}).items():
        hparams[k], hparam_sources[k] = v, file_origin
    for k, v in cli_hparams.items():
        hparams[k], hparam_sources[k] = v, "cli"
    return values, sources, hparams, hparam_sources


# ---------------------------------------------------------------------------
# Validation against the algorithm
# ---------------------------------------------------------------------------
def finalize_config(values: dict, sources: dict, hparams_raw: dict, hparam_sources: dict,
                    server_cls) -> argparse.Namespace:
    """Check every setting against the scenario rules and the algorithm's regime; fill regime defaults."""
    v = dict(values)
    notes: list[str] = []
    algo = v.get("algorithm") or server_cls.__name__

    def src(dest):
        return sources.get(dest, "default")

    def require(dest, why=""):
        if v[dest] is None:
            raise ConfigError(f"missing {describe(dest)}{why}")

    # ---- scenario
    for dest in ("dataset", "partition", "class_per_task"):
        require(dest)
    if v["class_per_task"] < 1:
        raise ConfigError(f"{describe('class_per_task')} must be >= 1")
    part = v["partition"]
    if part in ("dir", "iid"):
        require("num_clients", f" (needed for partition {part})")
        if v["num_clients"] < 1:
            raise ConfigError(f"{describe('num_clients')} must be >= 1")
    else:
        if v["num_clients"] is not None:
            notes.append(f"num_clients={v['num_clients']} ({src('num_clients')}) is ignored for natural "
                         f"partitions: the scenario gives one client per domain")
            v["num_clients"] = None
        if v["min_train_per_client"]:
            raise ConfigError("natural scenarios have no min_train_per_client")
    if part == "dir":
        require("data_alpha", " (needed for partition dir)")
        if not v["data_alpha"] > 0:
            raise ConfigError(f"{describe('data_alpha')} must be > 0")
    elif v["data_alpha"] is not None:
        raise ConfigError(f"{describe('data_alpha')} = {v['data_alpha']} ({src('data_alpha')}) has no meaning "
                          f"for partition {part}")
    if v["min_train_per_client"] < 0:
        raise ConfigError(f"{describe('min_train_per_client')} must be >= 0")

    # ---- what the algorithm reads: cached features or images
    input_kind = getattr(server_cls, "INPUT", FEATURES)
    if input_kind not in INPUTS:
        raise ConfigError(f"{algo}: INPUT must be one of {INPUTS}, got {input_kind!r}")
    run_flags = tuple(server_cls.RUN_FLAGS)
    if input_kind == FEATURES:
        for dest in MODEL_DESTS + IMAGE_RUN_DESTS:
            if v[dest] is not None:
                raise ConfigError(
                    f"{describe(dest)} = {v[dest]} ({src(dest)}) has no meaning for {algo}: it reads cached features "
                    f"(features section). The model section, num_workers and eval_batch_size are for algorithms "
                    f"that train a network on images. Remove the setting or choose another algorithm.")
        require("backbone")
        require("feature_group")
        if v["blocks"] is not None:
            if any(b < 0 for b in v["blocks"]) or len(set(v["blocks"])) != len(v["blocks"]):
                raise ConfigError(f"{describe('blocks')}: block ids must be distinct and >= 0, got {v['blocks']}")
        if v["tokens"] is not None and len(set(v["tokens"])) != len(v["tokens"]):
            raise ConfigError(f"{describe('tokens')}: duplicate tokens in {v['tokens']}")
        if v["proj_rank"] is not None and v["proj_rank"] < 1:
            raise ConfigError(f"{describe('proj_rank')} must be >= 1")
        if v["proj_seed"] is not None and v["proj_rank"] is None:
            raise ConfigError(f"{describe('proj_seed')} is set but {describe('proj_rank')} is not")
    else:
        for dest in FEATURE_DESTS:
            if v[dest] is not None:
                raise ConfigError(
                    f"{describe(dest)} = {v[dest]} ({src(dest)}) has no meaning for {algo}: it trains a network on "
                    f"images (model section) and reads no feature cache. Remove the setting or choose another "
                    f"algorithm.")
        _check_model_settings(v, src, algo)
        if NETWORKS[v["model"]].smoke and "SMOKE_TEST" not in run_flags:
            run_flags += ("SMOKE_TEST",)
            notes.append(f"{v['model']} exists for pipeline checks: the runs carry the SMOKE_TEST flag")

    # ---- regime and schedule
    regime = server_cls.REGIME
    if regime not in REGIMES:
        raise ConfigError(f"{algo}: REGIME must be one of {REGIMES}")
    if v["regime"] is not None and v["regime"] != regime:
        raise ConfigError(f"the config says regime {v['regime']} ({src('regime')}), but {algo} is a "
                          f"{regime} algorithm")
    v["regime"] = regime
    allowed = set(server_cls.schedule_allowed())
    for key in SCHEDULE_KEYS:
        val = v[key]
        if val is None or key in allowed:
            continue
        if regime == SINGLE_ROUND and key == "rounds_per_task" and val == 1:
            continue
        if regime == SINGLE_ROUND:
            hint = ("closed-form (single-round) algorithms get one upload per client per task; rounds, epochs, "
                    "batch size, learning rate, join ratio and curve gaps do not exist for them")
        else:
            hint = f"{algo} does not use it"
        raise ConfigError(f"{describe(key)} = {val} ({src(key)}) has no meaning for {algo}, a "
                          f"{REGIME_TEXT[regime]} algorithm: {hint}. Remove the setting or choose another algorithm.")
    for key in server_cls.schedule_required():
        if v[key] is None:
            raise ConfigError(f"{algo} is a {REGIME_TEXT[regime]} algorithm and needs {describe(key)}")
    if regime == SINGLE_ROUND:
        v["rounds_per_task"] = 1
    else:
        if "join_ratio" in allowed and v["join_ratio"] is None:
            v["join_ratio"] = 1.0
        if "eval_gap" in allowed and v["eval_gap"] is None:
            v["eval_gap"] = 0
        checks = (("rounds_per_task", lambda x: x >= 1, ">= 1"), ("local_epochs", lambda x: x >= 1, ">= 1"),
                  ("batch_size", lambda x: x >= 1, ">= 1"), ("local_learning_rate", lambda x: x > 0, "> 0"),
                  ("join_ratio", lambda x: 0 < x <= 1, "in (0, 1]"), ("eval_gap", lambda x: x >= 0, ">= 0"))
        for key, ok, text in checks:
            if v[key] is not None and not ok(v[key]):
                raise ConfigError(f"{describe(key)} must be {text}, got {v[key]}")

    # ---- algorithm hyperparameters
    schema = dict(server_cls.HPARAMS)
    unknown = sorted(set(hparams_raw) - set(schema))
    if unknown:
        raise ConfigError(f"unknown hyperparameter(s) {unknown} for {algo}; it declares "
                          f"{sorted(schema) if schema else 'none'}")
    hp, hp_src = {}, {}
    for name, spec in schema.items():
        if name in hparams_raw:
            hp[name] = spec.parse(hparams_raw[name], f"hyperparameter {name}")
            hp_src[name] = hparam_sources.get(name, "config")
        else:
            hp[name] = spec.parse(spec.default, f"hyperparameter {name}")
            hp_src[name] = "default"
    check = getattr(server_cls, "check_hparams", None)
    if check is not None:
        check(hp, hp_src)

    # ---- run
    if v["times"] < 1:
        raise ConfigError(f"{describe('times')} must be >= 1")
    if v["seed"] < 0:
        raise ConfigError(f"{describe('seed')} must be >= 0")

    args = argparse.Namespace(**v)
    args.algorithm = algo
    args.input_kind = input_kind
    args.hparams = hp
    args.sources = dict(sources)
    args.hparam_sources = hp_src
    args.notes = notes
    args.run_flags = run_flags
    args.algorithm_class = f"{server_cls.__module__}.{server_cls.__qualname__}"
    return args


def _check_model_settings(v: dict, src, algo: str) -> None:
    """Check the model section of an image algorithm and fill its defaults (in place)."""
    def text(dest):
        return f"{describe(dest)} = {v[dest]} ({src(dest)})"

    if v["model"] is None:
        raise ConfigError(f"missing {describe('model')}: {algo} trains a network on images; choose one of "
                          f"{list(NETWORKS)} (python main.py --list_settings describes them)")
    info = NETWORKS[v["model"]]
    name = info.name
    cnns = [n for n, i in NETWORKS.items() if i.scratch]
    if v["pretrained"] is None:
        v["pretrained"] = info.weights is not None
    elif v["pretrained"] and info.weights is None:
        raise ConfigError(f"{text('pretrained')}, but {name} has no pretrained weights here: it is trained from "
                          f"scratch (pretrained: false)")
    elif not v["pretrained"] and not info.scratch:
        raise ConfigError(f"{text('pretrained')}: {name} is available with its pretrained weights only. To train "
                          f"from scratch, choose one of {cnns}")
    if not v["pretrained"]:
        if v["trainable"] is None:
            v["trainable"] = "full"
        elif v["trainable"] != "full":
            raise ConfigError(f"{text('trainable')}: a network trained from scratch is trained in full; head, "
                              f"last_k, lora and prompt adapt a pretrained network")
    elif v["trainable"] is None:
        raise ConfigError(f"missing {describe('trainable')} for the pretrained {name}: choose "
                          f"{' | '.join(TRAINABLE_MODES)}")
    mode = v["trainable"]
    for other, keys in MODE_KEYS.items():
        if other == mode:
            continue
        for key in keys:
            if v[key] is not None:
                raise ConfigError(f"{text(key)} only applies to trainable {other}, but trainable is {mode}. "
                                  f"Remove the setting.")
    if mode == "last_k":
        if v["last_k"] is None:
            raise ConfigError(f"missing {describe('last_k')}: how many of the last transformer blocks of {name} "
                              f"to train (1 to {info.num_blocks})")
        if not 1 <= v["last_k"] <= info.num_blocks:
            raise ConfigError(f"{text('last_k')}: {name} has {info.num_blocks} blocks, so last_k must be 1 to "
                              f"{info.num_blocks}")
    elif mode == "lora":
        if v["lora_rank"] is None:
            v["lora_rank"] = DEFAULT_LORA_RANK
        if v["lora_rank"] < 1:
            raise ConfigError(f"{text('lora_rank')} must be >= 1")
        if v["lora_alpha"] is None:
            v["lora_alpha"] = float(v["lora_rank"])
        if not v["lora_alpha"] > 0:
            raise ConfigError(f"{text('lora_alpha')} must be > 0")
        if v["lora_targets"] is None:
            v["lora_targets"] = list(DEFAULT_LORA_TARGETS)
        if len(set(v["lora_targets"])) != len(v["lora_targets"]):
            raise ConfigError(f"{text('lora_targets')}: a weight is listed twice")
    elif mode == "prompt":
        if v["prompt_length"] is None:
            v["prompt_length"] = DEFAULT_PROMPT_LENGTH
        if v["prompt_length"] < 1:
            raise ConfigError(f"{text('prompt_length')} must be >= 1")
        if v["prompt_deep"] is None:
            v["prompt_deep"] = False
    if v["image_size"] is None:
        v["image_size"] = info.image_size
    elif not info.resizable and v["image_size"] != info.image_size:
        raise ConfigError(f"{text('image_size')}: {name} takes {info.image_size} x {info.image_size} images (the "
                          f"position embeddings of its weights)")
    elif v["image_size"] < MIN_IMAGE_SIZE:
        raise ConfigError(f"{text('image_size')} must be >= {MIN_IMAGE_SIZE}")
    if v["augment"] is None:
        v["augment"] = DEFAULT_AUGMENT[info.kind]
    if v["num_workers"] is None:
        v["num_workers"] = DEFAULT_NUM_WORKERS
    if v["num_workers"] < 0:
        raise ConfigError(f"{text('num_workers')} must be >= 0")
    if v["eval_batch_size"] is None:
        v["eval_batch_size"] = DEFAULT_EVAL_BATCH_SIZE
    if v["eval_batch_size"] < 1:
        raise ConfigError(f"{text('eval_batch_size')} must be >= 1")


def model_tag(args) -> str:
    """Short name of the network setup for folder names, e.g. resnet18_cifar-full or vit_b16_in21k-lora."""
    return f"{args.model}-{args.trainable}"


def model_summary(args) -> str:
    """The model settings in words (protocol printout)."""
    info = NETWORKS[args.model]
    origin = f"pretrained ({info.weights})" if args.pretrained else "from scratch (random initialisation)"
    mode = args.trainable
    detail = ""
    if mode == "last_k":
        detail = f" (the last {args.last_k} of {info.num_blocks} blocks, the final norm, the head)"
    elif mode == "lora":
        detail = (f" (rank {args.lora_rank}, alpha {args.lora_alpha:g}, on {','.join(args.lora_targets)} in every "
                  f"block, and the head)")
    elif mode == "prompt":
        detail = f" ({args.prompt_length} {'deep' if args.prompt_deep else 'shallow'} prompt tokens, and the head)"
    elif mode == "head":
        detail = " (the linear head only)"
    return f"{args.model}  {origin}  trainable: {mode}{detail}"


def settings_record(args) -> dict:
    """Every resolved setting (plus hyperparameters), for experiment.json."""
    out = {o.dest: getattr(args, o.dest, None) for o in OPTIONS}
    out["hparams"] = dict(getattr(args, "hparams", {}))
    return out


# ---------------------------------------------------------------------------
# The protocol in words (printed before every run and by --dry_run)
# ---------------------------------------------------------------------------
def describe_protocol(args, plan: dict) -> list[str]:
    multi = args.regime == MULTI_ROUND
    T, K = int(plan["num_tasks"]), int(plan["num_clients"])
    sizes = list(plan["task_sizes"])
    flags = " ".join(args.run_flags) if args.run_flags else "none"
    lines = [f"algorithm   {args.algorithm}  [{REGIME_TEXT[args.regime]}]  flags: {flags}"]
    scen = f"{args.dataset}  partition {args.partition}"
    if args.partition == "dir":
        scen += f" (alpha {args.data_alpha:g})"
    scen += f"  {K} clients"
    if args.use_val != "ignore":
        scen += f"  val split: {args.use_val}"
    lines.append(f"scenario    {scen}")
    inc = f"{args.class_per_task} classes per task -> {T} tasks over {plan['num_classes']} classes"
    if len(set(sizes)) > 1:
        inc += f" (task sizes {sizes})"
    lines.append(f"increment   {inc}")
    lines.append("seeds       " + "; ".join(f"run {r['run']}: {r['scenario']}" for r in plan["runs"]))
    if getattr(args, "input_kind", FEATURES) == FEATURES:
        blocks = "all stored" if args.blocks is None else str(args.blocks)
        tokens = "all stored" if args.tokens is None else ",".join(args.tokens)
        feat = f"{args.backbone}/{args.feature_group}  blocks {blocks}  tokens {tokens}  dim {plan['feature_dim']}"
        if args.proj_rank is not None:
            feat += f"  (projection rank {args.proj_rank} per block)"
        lines.append(f"features    {feat}")
    else:
        info = NETWORKS[args.model]
        lines.append(f"model       {model_summary(args)}")
        evaluation = ("the backbone's own preprocessing, as in feature extraction" if info.kind == "vit" else
                      f"resize and centre crop to {args.image_size} px, ImageNet mean and std")
        lines.append(f"input       raw images of {args.dataset}, {args.image_size} px; evaluation: {evaluation}; "
                     f"training augmentation: {args.augment}")
        if plan.get("image_check"):
            lines.append(f"images      {plan['images']} in cl_index, checked against the raw files; "
                         f"{plan['image_check']}")
    if multi:
        R = int(args.rounds_per_task)
        per_round = K if args.join_ratio >= 1 else max(1, int(K * args.join_ratio))
        lines.append(f"schedule    multi-round: {R} rounds per task -> {T * R} rounds in total; "
                     f"task t covers global rounds {R}*t .. {R}*t+{R - 1}")
        lines.append(f"            local epochs {args.local_epochs}, batch size {args.batch_size}, "
                     f"learning rate {args.local_learning_rate:g}, {per_round} of {K} clients per round")
        if args.eval_gap:
            lines.append(f"            curves: global test accuracy every {args.eval_gap} rounds "
                         f"(curves only, never used to choose anything)")
    else:
        lines.append("schedule    single-round (closed-form): every client uploads once per task; "
                     "there are no rounds, time is counted in tasks")
    if args.hparams:
        lines.append("hparams     " + ", ".join(f"{k}={v}" for k, v in args.hparams.items()))
    lines.append("evaluation  after every task: the global test set of all seen classes and each client's own "
                 "test shards; final values only, no best round")
    lines.extend(f"note        {n}" for n in getattr(args, "notes", []))
    return lines
