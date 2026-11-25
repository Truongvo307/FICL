"""Literal numpy restatements of the official computations, used only to build synthetic dumps for the tests.

Each function follows the released code statement by statement (file and function named in its docstring),
in float64, so a dump built with them must be reproduced exactly by our implementation through run_ours.py.
The real dumps are written by the official code itself (repro/run_*_official.py); these restatements only
test the replay machinery (import, configuration, patches, alignment of predictions and classifiers).
"""
from __future__ import annotations

import numpy as np


def onehot(y, C):
    out = np.zeros((len(y), C))
    out[np.arange(len(y)), y] = 1.0
    return out


def normalize_rows(W):
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)


def fedcof_repo_client_means(X, y):
    """FedCOF federate_classes.py FedCOFClient / FedNCMClient.compute_class_means."""
    return {int(c): (X[y == c].mean(axis=0), int(np.sum(y == c))) for c in np.unique(y)}


def fedcof_repo_fedcof(clients, C, d, gamma):
    """FedCOF federate_classes.py FedCOFServer.aggregate + get_classifier_weights -> W [C, d] (rows normalized)."""
    means = {c: [] for c in range(C)}
    counts = {c: [] for c in range(C)}
    sums, n_c = {c: np.zeros(d) for c in range(C)}, {c: 0 for c in range(C)}
    for stats in clients:
        for c, (mu, n) in stats.items():
            sums[c] += mu * float(n)
            n_c[c] += n
            means[c].append(mu)
            counts[c].append(float(n))
    cm = {c: sums[c] / n_c[c] if n_c[c] > 0 else np.zeros(d) for c in range(C)}
    G = np.zeros((d, d))
    all_count, g = 0, 0
    for c in range(C):
        if n_c[c] == 0:
            continue
        M = np.stack(means[c])
        all_count += n_c[c]
        g = g + cm[c] * n_c[c]
        dx = (M - cm[c]) * np.sqrt(np.asarray(counts[c]))[:, None]
        inside = dx.T @ dx
        delta = inside / (len(M) - 1) if len(M) > 1 else inside / len(M)
        G += (n_c[c] - 1) * (delta + gamma * np.eye(d))
    g = g / all_count
    G += all_count * np.outer(g, g)
    inv = np.linalg.pinv(G)
    W = np.stack([inv @ sums[c] for c in range(C)])
    return normalize_rows(W)


def fedcof_repo_fedncm(clients, C, d):
    """FedCOF federate_classes.py FedNCMServer -> normalized class means [C, d]."""
    sums, n_c = np.zeros((C, d)), np.zeros(C)
    for stats in clients:
        for c, (mu, n) in stats.items():
            sums[c] += mu * float(n)
            n_c[c] += n
    return normalize_rows(np.where(n_c[:, None] > 0, sums / np.maximum(n_c, 1)[:, None], 0.0))


def fed3r_official(parts, C, lam):
    """Fed3R src/fed3r.py fed3r_step over the clients, then calc_ridge_params and normalize_W.
    Returns (W [d, C], bias [C])."""
    d = parts[0][0].shape[1]
    A = lam * np.eye(d + 1)
    b = np.zeros((d + 1, C))
    for X, y in parts:
        F = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        A += F.T @ F
        b += F.T @ onehot(y, C)
    sol = np.linalg.solve(A, b)
    W, bias = sol[:-1], sol[-1]
    norm = np.linalg.norm(W, axis=0, keepdims=True)
    norm[norm == 0.0] = 1e-6
    return W / norm, bias / norm[0]


def afl_official(parts, C, gamma):
    """AFL CIFAR-100/afl.py local_update per client, aggregation, clean_regularization. Returns W [d, C]."""
    Ws, Rs, Cs = [], [], []
    for X, y in parts:
        R = np.linalg.inv(X.T @ X + gamma * np.eye(X.shape[1]))
        Ws.append(R @ (X.T @ onehot(y, C)))
        Rs.append(R)
        Cs.append(np.linalg.inv(R))
    eye = np.eye(Rs[0].shape[0])
    Wt = (eye - Rs[0] @ Cs[1] + Rs[0] @ Cs[1] @ np.linalg.inv(Cs[0] + Cs[1]) @ Cs[1]) @ Ws[0] + \
         (eye - Rs[1] @ Cs[0] + Rs[1] @ Cs[0] @ np.linalg.inv(Cs[0] + Cs[1]) @ Cs[0]) @ Ws[1]
    Ct = Cs[0] + Cs[1]
    Rt = np.linalg.inv(Ct)
    for i in range(1, len(Ws) - 1):
        Wt = (eye - Rt @ Cs[i + 1] + Rt @ Cs[i + 1] @ np.linalg.inv(Ct + Cs[i + 1]) @ Cs[i + 1]) @ Wt + \
             (eye - Rs[i + 1] @ Ct + Rs[i + 1] @ Ct @ np.linalg.inv(Ct + Cs[i + 1]) @ Ct) @ Ws[i + 1]
        Ct = Ct + Cs[i + 1]
        Rt = np.linalg.inv(Ct)
    K = len(Ws)
    R_origin = np.linalg.inv(Ct - K * gamma * eye)
    return Wt + (K * gamma * R_origin) @ Wt


def stsa_project(X, W_rand):
    return np.maximum(X @ W_rand, 0.0)


def stsa_cosine_predict(X, W_rand, W):
    H = normalize_rows(stsa_project(X, W_rand))
    return np.argmax(H @ normalize_rows(W).T, axis=1)


def stsa_e_task_gram(group_sums, group_counts, P_t, ridge_unused=None):
    """STSA methods/STSA.py replace_fc_ours3: G_t from the dummy-group class sums of one task.
    group_sums: list over groups of [nb_classes, M]; group_counts: list of [nb_classes]."""
    K = len(group_sums)
    N_class = np.sum(np.stack(group_counts), axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where((N_class - 1) / (K - 1) < 0, 0.0, (N_class - 1) / (K - 1))
        Bv = (N_class - K) / ((K - 1) * N_class)
        B = np.where(Bv < 0, 0.0, Bv)
    M = group_sums[0].shape[1]
    G_t = np.zeros((M, M))
    for s, n in zip(group_sums, group_counts, strict=True):
        for i in range(len(N_class)):
            if A[i] == 0 or n[i] == 0:
                continue
            G_t += A[i] * np.outer(s[i], s[i]) / n[i]
    for i in range(len(N_class)):
        if B[i] == 0:      # also skips the NaN of classes without samples, as torch.where(NaN < 0) keeps NaN
            continue
        if not np.isfinite(B[i]):
            continue
        G_t -= B[i] * np.outer(P_t[:, i], P_t[:, i])
    return G_t
