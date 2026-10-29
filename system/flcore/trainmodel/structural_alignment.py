"""
Structural Alignment losses for FedSAF.
Implements GCSA (Gram-Cosine) and RCSA (RDM-Cosine) structural alignment,
as an alternative to coordinate (MSE/cosine) alignment in prototype-based HtFL.
"""

import torch
import torch.nn.functional as F


def gcsa_loss(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gram-Cosine Structural Alignment (Eq. 13).
    P, Q: [n, d] matrices of prototypes/features (row-aligned by class/index).
    Invariant to translation, rotation, and positive scaling.
    """
    if P.shape[0] < 2:
        return torch.tensor(0.0, device=P.device)

    p_bar = P.mean(dim=0, keepdim=True)
    q_bar = Q.mean(dim=0, keepdim=True)

    Pc = P - p_bar
    Qc = Q - q_bar

    K_P = Pc @ Pc.T
    K_Q = Qc @ Qc.T

    numer = (K_P * K_Q).sum()
    denom = K_P.norm(p='fro') * K_Q.norm(p='fro') + eps

    return 1.0 - numer / denom


def rcsa_loss(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    RDM-Cosine Structural Alignment (Eq. 14).
    P, Q: [n, d] matrices, n >= 2.
    Invariant to orthogonal transformations (rotation/reflection).
    """
    n = P.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=P.device)

    P_norm = F.normalize(P, p=2, dim=1)
    Q_norm = F.normalize(Q, p=2, dim=1)

    # Pairwise squared distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    def pairwise_sqdist(X):
        sq = (X ** 2).sum(dim=1, keepdim=True)
        return sq + sq.T - 2.0 * (X @ X.T)

    RDM_P = pairwise_sqdist(P_norm)
    RDM_Q = pairwise_sqdist(Q_norm)

    triu_idx = torch.triu_indices(n, n, offset=1)
    vec_P = RDM_P[triu_idx[0], triu_idx[1]]
    vec_Q = RDM_Q[triu_idx[0], triu_idx[1]]

    numer = (vec_P * vec_Q).sum()
    denom = vec_P.norm() * vec_Q.norm() + eps

    return 1.0 - numer / denom


def structural_loss(P: torch.Tensor, Q: torch.Tensor, method: str = 'gcsa') -> torch.Tensor:
    """Dispatch to the selected structural alignment loss."""
    method = method.lower()
    if method == 'gcsa':
        return gcsa_loss(P, Q)
    elif method == 'rcsa':
        return rcsa_loss(P, Q)
    else:
        raise ValueError(f"Unknown structural alignment method: {method}. Use 'gcsa' or 'rcsa'.")