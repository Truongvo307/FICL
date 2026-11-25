"""Placeholder for wandb, used only when wandb is not installed.

FedCOF's server_aggregate.py and STSA's main.py import wandb at module level but call it only with --wandb 1,
which the reproduction scripts never pass. Any use raises, so a silent call is impossible.
"""


def __getattr__(name):
    raise RuntimeError(f"wandb.{name} called, but wandb is not installed (repro stub); install wandb or keep --wandb 0")
