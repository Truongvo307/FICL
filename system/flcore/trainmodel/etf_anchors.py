import torch
import torch.nn.functional as F
import warnings

def build_etf_anchors(num_classes: int, dim: int) -> torch.Tensor:
    C, d = num_classes, dim
    assert d >= C - 1, f"d_shared ({d}) must be >= num_classes-1 ({C-1})"
    P = torch.eye(d)[:, :C]              # [d, C] - first C identity columns
    ones = torch.ones(C, C)
    M = ((C / (C - 1)) ** 0.5) * (torch.eye(C) - ones / C)  # [C, C]
    anchors = (P @ M).T                  # [C, d]
    anchors = F.normalize(anchors, dim=1)  # onto S^{d-1}
    return anchors       


def verify_etf(anchors: torch.Tensor) -> bool:
    """Unit test: ||mu_c|| == 1 and <mu_c, mu_c'> ≈ -1/(C-1)."""
    C = anchors.shape[0]
    norms = anchors.norm(dim=1)
    gram = anchors @ anchors.T
    off_diag = gram[~torch.eye(C, dtype=torch.bool)]
    expected_inner = -1.0 / (C - 1)
    norm_ok = torch.allclose(norms, torch.ones(C), atol=1e-5)
    inner_ok = torch.allclose(off_diag, torch.full_like(off_diag, expected_inner), atol=1e-4)
    return norm_ok and inner_ok