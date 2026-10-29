"""
Manifolds for GeoProtoFL.
SphereManifold is the default for prototype aggregation (S^{d-1}).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def get_manifold(manifold_type: str, dim: int, device: torch.device,
                 learnable_metric: bool = False):
    if manifold_type == 'sphere':
        return SphereManifold(dim, device)
    elif manifold_type == 'euclidean':
        return EuclideanManifold(dim, device, learnable_metric)
    elif manifold_type == 'hyperbolic':
        return HyperbolicManifold(dim, device)
    elif manifold_type == 'spd':
        return SPDManifold(dim, device)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def geometric_median_barycenter(points: torch.Tensor,
                                 weights: Optional[torch.Tensor] = None,
                                 max_iter: int = 50,
                                 tol: float = 1e-6) -> torch.Tensor:
    """Weiszfeld algorithm on the sphere (projected geometric median)."""
    n = points.shape[0]
    if weights is None:
        weights = torch.ones(n, device=points.device) / n
    mu = F.normalize(torch.sum(weights.unsqueeze(1) * points, dim=0), dim=0)
    for _ in range(max_iter):
        dists = torch.clamp(
            torch.sqrt(torch.sum((points - mu.unsqueeze(0))**2, dim=-1)), min=1e-8)
        w = weights / dists
        new_mu = F.normalize(torch.sum(w.unsqueeze(1) * points, dim=0), dim=0)
        if torch.norm(new_mu - mu) < tol:
            break
        mu = new_mu
    return mu


class BaseManifold:
    def __init__(self, dim: int, device: torch.device):
        self.dim = dim
        self.device = device

    def distance(self, x, y): raise NotImplementedError
    def exp_map(self, x, v): raise NotImplementedError
    def log_map(self, x, y): raise NotImplementedError
    def project(self, x): raise NotImplementedError
    def barycenter(self, points, weights=None, max_iter=10,
                   lr=1.0, tol=1e-6, init=None): raise NotImplementedError


class SphereManifold(BaseManifold):
    """
    Unit hypersphere S^{d-1}.
    Default manifold for GeoProtoFL prototype aggregation.
    Implements Fréchet barycenter via Riemannian gradient descent (Eq.9).
    """

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot = torch.clamp((x * y).sum(dim=-1), -1 + 1e-7, 1 - 1e-7)
        return torch.arccos(dot)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot = torch.clamp((x * y).sum(dim=-1, keepdim=True), -1 + 1e-7, 1 - 1e-7)
        theta = torch.arccos(dot)
        u = y - dot * x
        u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-7)
        return theta * u / u_norm

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def barycenter(self, points: torch.Tensor,
                   weights: Optional[torch.Tensor] = None,
                   max_iter: int = 10, lr: float = 1.0,
                   tol: float = 1e-6,
                   init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Karcher mean on S^{d-1} via Riemannian gradient descent.
        Initialized from previous global prototype (Eq.9: q^0 = p^{c,t-1}).
        """
        n = points.shape[0]
        if weights is None:
            weights = torch.ones(n, device=self.device) / n
        else:
            weights = weights / weights.sum()
        points = self.project(points)
        # Warm-start from previous prototype (Eq.9), else use first point
        mu = self.project(init) if init is not None else points[0].clone()
        for _ in range(max_iter):
            mu_exp = mu.unsqueeze(0).expand(n, -1)
            grad = torch.sum(
                weights.unsqueeze(1) * self.log_map(mu_exp, points), dim=0)
            if torch.norm(grad) < tol:
                break
            mu = self.project(self.exp_map(mu, lr * grad))
        return mu


class EuclideanManifold(BaseManifold):
    """Euclidean manifold with optional Mahalanobis metric."""

    def __init__(self, dim: int, device: torch.device,
                 learnable_metric: bool = False):
        super().__init__(dim, device)
        self.learnable_metric = learnable_metric
        if learnable_metric:
            self.L = nn.Parameter(torch.eye(dim, device=device))
        else:
            self.L = torch.eye(dim, device=device)

    def distance(self, x, y):
        diff = x - y
        if diff.dim() == 1:
            diff = diff.unsqueeze(0)
        if self.learnable_metric:
            M = self.L @ self.L.t()
            return torch.sqrt(torch.sum(diff @ M * diff, dim=-1) + 1e-8)
        return torch.norm(diff, dim=-1)

    def exp_map(self, x, v): return x + v
    def log_map(self, x, y): return y - x
    def project(self, x): return x

    def barycenter(self, points, weights=None, max_iter=10,
                   lr=0.5, tol=1e-6, init=None):
        n = points.shape[0]
        if weights is None:
            weights = torch.ones(n, device=self.device) / n
        else:
            weights = weights / weights.sum()
        mu = init.clone() if init is not None else \
            torch.sum(weights.unsqueeze(1) * points, dim=0)
        for _ in range(max_iter):
            grad = torch.sum(
                weights.unsqueeze(1) * (points - mu.unsqueeze(0)), dim=0)
            if torch.norm(grad) < tol:
                break
            mu = mu + lr * grad
        return mu


class HyperbolicManifold(BaseManifold):
    """Poincaré ball model of hyperbolic space."""

    def __init__(self, dim: int, device: torch.device, curvature: float = 1.0):
        super().__init__(dim, device)
        self.c = curvature
        self.eps = 1e-5
        self.max_norm = 1 - 1e-3

    def _lambda_x(self, x):
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / torch.clamp(1 - self.c * x_sqnorm, min=self.eps)

    def _mobius_add(self, x, y):
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2*self.c*xy_inner + self.c*y_sqnorm)*x + \
              (1 - self.c*x_sqnorm)*y
        denom = 1 + 2*self.c*xy_inner + self.c**2*x_sqnorm*y_sqnorm
        return num / torch.clamp(denom, min=self.eps)

    def distance(self, x, y):
        x, y = self.project(x), self.project(y)
        diff = self._mobius_add(-x, y)
        norm = torch.clamp(torch.norm(diff, dim=-1), max=self.max_norm)
        return (2.0/np.sqrt(self.c)) * torch.arctanh(np.sqrt(self.c) * norm)

    def exp_map(self, x, v):
        x = self.project(x)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=self.eps)
        lx = self._lambda_x(x)
        sc = np.sqrt(self.c)
        return self.project(self._mobius_add(
            x, torch.tanh(sc*lx*v_norm/2)*v/(sc*v_norm)))

    def log_map(self, x, y):
        x, y = self.project(x), self.project(y)
        diff = self._mobius_add(-x, y)
        dn = torch.clamp(torch.norm(diff, dim=-1, keepdim=True),
                          min=self.eps, max=self.max_norm)
        lx = self._lambda_x(x)
        sc = np.sqrt(self.c)
        return (2.0/(sc*lx)) * torch.arctanh(sc*dn) * diff/dn

    def project(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return torch.where(norm > self.max_norm, x/norm*self.max_norm, x)

    def barycenter(self, points, weights=None, max_iter=10,
                   lr=0.5, tol=1e-6, init=None):
        n = points.shape[0]
        if weights is None:
            weights = torch.ones(n, device=self.device) / n
        else:
            weights = weights / weights.sum()
        points = self.project(points)
        mu = self.project(init*0.5) if init is not None else \
            self.project(torch.sum(weights.unsqueeze(1)*points, dim=0)*0.5)
        for _ in range(max_iter):
            mu_exp = mu.unsqueeze(0).expand(n, -1)
            log_vecs = self.log_map(mu_exp, points)
            grad = torch.sum(weights.unsqueeze(1)*log_vecs, dim=0)
            if torch.norm(grad) < tol:
                break
            mu = self.exp_map(mu.unsqueeze(0), (lr*grad).unsqueeze(0)).squeeze(0)
        return mu