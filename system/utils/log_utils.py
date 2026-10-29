"""Run logs: one log file per run (inside its times_<i> folder) plus the console.

    logger = setup_logger(args, run_index, run_dir)   # <dataset>_<algorithm>_<goal>_run<i>.log
    log_config(logger, args)                          # every resolved setting, once per run
    close_log_files(logger)                           # when the run ends (the shared loop does this)
"""
from __future__ import annotations

import itertools
import logging
import os

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_ids = itertools.count()


def setup_logger(args, run_index: int, run_dir: str) -> logging.Logger:
    """A fresh logger per run: a file in run_dir and the console. It does not propagate to the root logger."""
    path = os.path.join(run_dir, f"{args.dataset}_{args.algorithm}_{args.goal}_run{run_index}.log")
    logger = logging.getLogger(f"ficl.run{run_index}.{next(_ids)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(LOG_FORMAT)
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setLevel(logging.INFO)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def log_config(logger: logging.Logger, args) -> None:
    """Every setting of the run, sorted by name."""
    logger.info("=" * 22 + " settings " + "=" * 22)
    for key in sorted(vars(args)):
        logger.info(f"{key}: {getattr(args, key)}")
    logger.info("=" * 54)


def close_log_files(logger: logging.Logger) -> None:
    """Close the run's log file instead of keeping every run's file open."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
