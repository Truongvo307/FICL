"""
Layer convergence detection utilities for IDEA-6.

Implements layer-wise convergence metrics and freezing decisions.
"""

import torch
from typing import Dict


def compute_layer_convergence(
    prev_outputs: Dict[str, torch.Tensor],
    curr_outputs: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute layer-wise convergence scores.

    Conv_l = 1 - ||z_t - z_{t-1}|| / ||z_{t-1}||

    Args:
        prev_outputs: Cached layer outputs from previous checkpoint.
        curr_outputs: Current layer outputs.
        eps: Numerical stability constant.

    Returns:
        Dictionary mapping layer name to convergence score.
    """
    convergence = {}

    for layer, z_prev in prev_outputs.items():
        z_curr = curr_outputs[layer]
        delta = torch.norm(z_curr - z_prev)
        base = torch.norm(z_prev) + eps
        convergence[layer] = float(1.0 - delta / base)

    return convergence
