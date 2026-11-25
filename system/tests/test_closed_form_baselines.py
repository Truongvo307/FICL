"""Fed3R, FedCOF, STSA, STSA-E and AFL on cached features.

For each method: the classifier after every task equals the method computed directly (centrally, or by a
literal re-statement of the official server code) on the same data; the uploads hold what the method sends;
the class-incremental accumulation uses nothing of later tasks; and an end-to-end run works. Estimators
that the papers prove unbiased (FedCOF Prop. 1, STSA-E Eq. 9) are checked by simulation.
"""
import os

import numpy as np
import pytest

import helpers as H
from flcore.clients.clientfed3r import apply_rf, random_fourier_map
from flcore.clients.clientfedcof import split_for_means
from flcore.clients.clientstsa import dummy_groups, project
from flcore.closedform_utils import symmetrize_upper
from flcore.servers.serverafl import AFL, SingularGramError, aa_step, clean_regularization
from flcore.servers.serverfed3r import Fed3R
from flcore.servers.serverfedcof import FedCOF, fedcof_head
from flcore.servers.serverstsa import STSA, STSA_E, stsae_task_gram
from utils.comm_utils import Symmetric, wire_nbytes

SCENARIOS = (H.DIR_SCENARIO, H.NATURAL_SCENARIO)


def _snapshot(cls):
    """A subclass of a baseline server that keeps the classifier after every task."""
    class Snap(cls):
        def init_algorithm(self):
            super().init_algorithm()
            self.heads = []

        def on_end_task(self, t):
            super().on_end_task(t)
            self.heads.append({k: np.array(v, copy=True) for k, v in
                               (("classes", self.head_classes), ("W", self.W),
                                ("bias", getattr(self, "bias", None))) if v is not None})
    Snap.__name__ = cls.__name__
    return Snap


def _run(cls, sr, tmp_path, name, scenario=None, hparams=None):
    return H.run_server(_snapshot(cls), H.make_cfg(sr["root"], sr["results"], cls.__name__, scenario=scenario,
                                                   hparams=hparams), str(tmp_path / name))


def _train_data(sr, classes):
    """All training features (block 3, cls token) and labels of the given classes."""
    index = sr["index"]
    feats = np.load(sr["cache_file"], mmap_mode="r")[:, 3, 0, :].astype(np.float64)
    rows = np.flatnonzero((index.split == 0) & np.isin(index.labels, classes))
    return feats[rows], index.labels[rows]


def _onehot(y, classes):
    return (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(np.float64)


def _close(a, b, rel=1e-8):
    return np.max(np.abs(a - b)) <= rel * max(np.max(np.abs(b)), 1e-300)


# ---------------------------------------------------------------------------------------------- Fed3R
def _fed3r_central(Z, y, classes, lam, rf=None):
    F = np.hstack([apply_rf(Z, rf), np.ones((len(Z), 1))])
    sol = np.linalg.solve(lam * np.eye(F.shape[1]) + F.T @ F, F.T @ _onehot(y, classes))
    W, bias = sol[:-1], sol[-1]
    n = np.linalg.norm(W, axis=0)
    return W / n, bias / n


@pytest.mark.parametrize("rf_dim", [0, 50])
def test_fed3r_equals_central_ridge_with_bias_and_normalized_columns_after_every_task(synthetic_root, tmp_path,
                                                                                       rf_dim):
    for i, scen in enumerate(SCENARIOS):
        s = _run(Fed3R, synthetic_root, tmp_path, f"f{i}", scenario=scen, hparams={"rf_dim": rf_dim, "rf_sigma": 4.0})
        for h in s.heads:
            Z, y = _train_data(synthetic_root, h["classes"])
            W, bias = _fed3r_central(Z, y, h["classes"], 0.01, s.rf)
            assert _close(h["W"], W) and _close(h["bias"], bias)
        if rf_dim:
            assert s.rf[0].shape == (16, rf_dim)
            for c in s.clients:                       # every client got the seed once and built the same map
                assert c._rf is not None and np.array_equal(c._rf[0], s.rf[0]) and np.array_equal(c._rf[1], s.rf[1])


def test_fed3r_random_fourier_features_approximate_the_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 5))
    sigma = 2.0
    rf = random_fourier_map(1, 5, 20000, sigma)
    F = apply_rf(X, rf)
    K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / (2 * sigma ** 2))
    assert np.max(np.abs(F @ F.T - K)) < 0.05


# ---------------------------------------------------------------------------------------------- FedCOF
def _fedcof_literal(client_data, classes, gamma, d):
    """FedCOFServer.aggregate + get_classifier_weights of the official federate_classes.py, restated with
    numpy loops over dicts (client_data: {client: {"class_means": {c: mu}, "sample_per_mean": {c: n}}})."""
    covs, mean_clients, counts = {c: None for c in classes}, {c: [] for c in classes}, {c: [] for c in classes}
    sums = {c: np.zeros(d) for c in classes}
    N_c = {c: 0 for c in classes}
    for data in client_data.values():
        for c, mu in data["class_means"].items():
            sums[c] += mu * float(data["sample_per_mean"][c])
            N_c[c] += data["sample_per_mean"][c]
            mean_clients[c].append(mu)
            counts[c].append(float(data["sample_per_mean"][c]))
    means = {c: sums[c] / N_c[c] for c in classes}
    all_count, g = 0, 0
    for c in classes:
        mc = np.stack(mean_clients[c])
        all_count += N_c[c]
        g = g + means[c] * N_c[c]
        mu_x = (mc - means[c]) * np.sqrt(np.asarray(counts[c]))[:, None]
        inside = mu_x.T @ mu_x
        delta = inside / (len(mc) - 1) if len(mc) > 1 else inside / len(mc)
        covs[c] = delta + gamma * np.eye(d)
    g = g / all_count
    G = all_count * np.outer(g, g)
    for c in classes:
        G += (N_c[c] - 1) * covs[c]
    W = np.stack([np.linalg.pinv(G) @ sums[c] for c in classes])
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def test_fedcof_equals_the_official_server_computation_after_every_task(synthetic_root, tmp_path):
    for i, scen in enumerate(SCENARIOS):
        s = _run(FedCOF, synthetic_root, tmp_path, f"c{i}", scenario=scen, hparams={"gamma_shrink": 0.5})
        index = synthetic_root["index"]
        feats = np.load(synthetic_root["cache_file"], mmap_mode="r")[:, 3, 0, :].astype(np.float64)
        for t, h in enumerate(s.heads):
            client_data = {}
            for k in range(s.num_clients):
                rows = np.concatenate([s.scenario.train_rows[u][k] for u in range(t + 1)]).astype(np.int64)
                if len(rows) == 0:
                    continue
                y = index.labels[rows]
                client_data[k] = {"class_means": {int(c): feats[rows][y == c].mean(axis=0) for c in np.unique(y)},
                                  "sample_per_mean": {int(c): int((y == c).sum()) for c in np.unique(y)}}
            W = _fedcof_literal(client_data, [int(c) for c in h["classes"]], 0.5, feats.shape[1])
            assert _close(h["W"], W, rel=1e-7)


def test_fedcof_single_mean_class_gets_gamma_identity_and_the_estimator_is_unbiased():
    rng = np.random.default_rng(0)
    d, gamma = 3, 0.7
    W, _info = fedcof_head([rng.normal(size=(1, d)), rng.normal(size=(2, d))], [np.array([5]), np.array([3, 4])],
                          gamma)
    assert W.shape == (2, d) and np.allclose(np.linalg.norm(W, axis=1), 1.0)
    # Prop. 1: with iid features, 1/(K-1) sum_k n_k (mu_k - mu)(mu_k - mu)^T has expectation Sigma
    L = rng.normal(size=(d, d))
    Sigma = L @ L.T
    est = np.zeros((d, d))
    reps = 400
    for _ in range(reps):
        n = rng.integers(5, 30, size=20)
        mk = np.stack([rng.multivariate_normal(np.zeros(d), Sigma, size=k).mean(axis=0) for k in n])
        mu = (n[:, None] * mk).sum(0) / n.sum()
        diff = (mk - mu) * np.sqrt(n)[:, None]
        est += diff.T @ diff / (len(n) - 1)
    assert np.max(np.abs(est / reps - Sigma)) < 0.1 * np.max(np.abs(Sigma))


def test_fedcof_multiple_means_are_disjoint_subsets_of_at_least_two_samples():
    rng = np.random.default_rng(1)
    for n, M, expect in ((1, 5, 1), (3, 5, 1), (4, 2, 2), (10, 3, 3), (7, 10, 3)):
        parts = split_for_means(n, M, rng)
        assert len(parts) == expect
        allidx = np.concatenate(parts)
        assert sorted(allidx.tolist()) == list(range(n))
        assert expect == 1 or min(len(p) for p in parts) >= 2


def test_fedcof_with_two_means_per_client_runs_and_sends_more_means(synthetic_root, tmp_path):
    s1 = _run(FedCOF, synthetic_root, tmp_path, "m1")
    s2 = _run(FedCOF, synthetic_root, tmp_path, "m2", hparams={"means_per_client": 2})
    n1 = sum(len(v) for v in s1.class_counts.values())
    n2 = sum(len(v) for v in s2.class_counts.values())
    assert n2 > n1
    for c in range(s2.num_classes):                            # counts still add up to the class sizes
        assert sum(s2.class_counts[c]) == sum(s1.class_counts[c])


# ---------------------------------------------------------------------------------------------- STSA
@pytest.mark.parametrize("rp_dim", [40, 16])
def test_stsa_equals_central_ridge_on_projected_features_after_every_task(synthetic_root, tmp_path, rp_dim):
    for i, scen in enumerate(SCENARIOS):
        s = _run(STSA, synthetic_root, tmp_path, f"s{i}", scenario=scen, hparams={"rp_dim": rp_dim, "ridge": 1.0})
        assert (s.W_rand is None) == (rp_dim == 16)              # M = d: no projection
        for h in s.heads:
            Z, y = _train_data(synthetic_root, h["classes"])
            Hf = project(Z, s.W_rand)
            W = np.linalg.solve(Hf.T @ Hf + np.eye(rp_dim), Hf.T @ _onehot(y, h["classes"])).T
            assert _close(h["W"], W)
        for c in s.clients:
            assert (c._W_rand is None and s.W_rand is None) or np.array_equal(c._W_rand, s.W_rand)


def test_stsa_predicts_with_cosine_similarity(synthetic_root, tmp_path):
    s = _run(STSA, synthetic_root, tmp_path, "cos", hparams={"rp_dim": 40, "ridge": 1.0})
    Z = np.random.default_rng(0).normal(size=(50, 16))
    Hf = project(Z, s.W_rand)
    Wn = s.W / np.linalg.norm(s.W, axis=1, keepdims=True)
    expect = s.head_classes[np.argmax(Hf @ Wn.T, axis=1)]
    assert np.array_equal(s.predict(Z, s.head_classes), expect)


def _stsae_literal(sum_list, ss_list, P_t, nb_classes):
    """replace_fc_ours3's G_t of the official methods/STSA.py, restated with numpy loops."""
    K = len(sum_list)
    N_class = np.sum(ss_list, axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where((N_class - 1) / (K - 1) < 0, 0.0, (N_class - 1) / (K - 1))
        Bv = (N_class - K) / ((K - 1) * N_class)
    B = np.where(Bv < 0, 0.0, Bv)
    M = P_t.shape[0]
    G_t = np.zeros((M, M))
    for idx, s_t in enumerate(sum_list):
        for i in range(nb_classes):
            if A[i] == 0 or ss_list[idx][i] == 0:
                continue
            G_t += A[i] * np.outer(s_t[i], s_t[i]) / ss_list[idx][i]
    for i in range(nb_classes):
        if B[i] == 0 or not np.isfinite(B[i]):
            continue
        G_t -= B[i] * np.outer(P_t[:, i], P_t[:, i])
    return G_t


def test_stsae_gram_estimate_equals_the_official_server_computation():
    rng = np.random.default_rng(0)
    C, M, K = 4, 6, 7
    sum_list, ss_list = [], []
    for _ in range(K):
        n = rng.integers(0, 6, size=C)
        n[rng.integers(C)] = 0
        sums = np.stack([rng.normal(size=(int(k), M)).sum(axis=0) if k else np.zeros(M) for k in n])
        sum_list.append(sums)
        ss_list.append(n)
    P_t = np.sum(sum_list, axis=0).T
    expect = _stsae_literal(sum_list, ss_list, P_t, C)
    rows = [(g, i) for g in range(K) for i in range(C) if ss_list[g][i] > 0]
    G_t, C_t, n = stsae_task_gram(np.stack([sum_list[g][i] for g, i in rows]),
                                  np.array([ss_list[g][i] for g, i in rows], dtype=float),
                                  np.array([i for _, i in rows]), K, C)
    assert np.allclose(G_t, expect, rtol=1e-12, atol=1e-10)
    assert np.allclose(C_t, P_t) and np.array_equal(n, np.sum(ss_list, axis=0))


def test_stsae_gram_estimate_is_unbiased_for_iid_groups():
    """Eq. 9 estimates sum x x^T of each class from group class sums; with iid samples (and every class count
    at least K, so no coefficient is clamped) its average over draws matches the expectation N (Sigma + mu mu^T)."""
    rng = np.random.default_rng(2)
    M, K, n_per = 3, 5, 40
    L = rng.normal(size=(M, M))
    Sigma, mu = L @ L.T, rng.normal(size=M)
    acc, reps = np.zeros((M, M)), 3000
    for _ in range(reps):
        X = rng.multivariate_normal(mu, Sigma, size=n_per)
        groups = dummy_groups(n_per, K, rng)
        sums = np.stack([X[g].sum(axis=0) for g in groups])
        G_t, _, _ = stsae_task_gram(sums, np.array([len(g) for g in groups], float), np.zeros(K, int), K, 1)
        acc += G_t
    expect = n_per * (Sigma + np.outer(mu, mu))
    assert np.max(np.abs(acc / reps - expect)) < 0.05 * np.max(np.abs(expect))


def test_stsae_groups_follow_the_official_split():
    rng = np.random.default_rng(0)
    g = dummy_groups(23, 5, rng)
    assert [len(x) for x in g] == [4, 4, 4, 4, 7]
    assert sorted(np.concatenate(g).tolist()) == list(range(23))
    g = dummy_groups(3, 5, rng)                            # fewer samples than groups: all in the last group
    assert [len(x) for x in g] == [0, 0, 0, 0, 3]


def test_stsae_runs_and_is_close_to_stsa(synthetic_root, tmp_path):
    a = _run(STSA, synthetic_root, tmp_path, "a", hparams={"rp_dim": 40, "ridge": 1.0})
    e = _run(STSA_E, synthetic_root, tmp_path, "e", hparams={"rp_dim": 40, "ridge": 1.0, "subsets": 10})
    assert np.array_equal(a.W_rand, e.W_rand)
    assert np.array_equal(a.C, e.C) or np.allclose(a.C, e.C)       # first-order statistics are exact
    assert not np.allclose(a.G, e.G)                                # the second-order ones are estimated
    fa = H.read_json(os.path.join(a.run_dir, "final.json"))["metrics"]["acc_final"]
    fe = H.read_json(os.path.join(e.run_dir, "final.json"))["metrics"]["acc_final"]
    assert fe > 0.6 and fa > 0.6


# ---------------------------------------------------------------------------------------------- AFL
def test_afl_aggregation_law_equals_the_joint_solution_and_ri_gives_least_squares():
    rng = np.random.default_rng(0)
    d, C, gamma = 5, 3, 1.0
    parts = [rng.normal(size=(n, d)) for n in (12, 7, 30, 9)]
    labels = [rng.integers(0, C, size=len(p)) for p in parts]
    Ws, Cs, Rs = [], [], []
    for X, y in zip(parts, labels, strict=True):
        R = np.linalg.inv(X.T @ X + gamma * np.eye(d))
        Ws.append(R @ X.T @ _onehot(y, range(C)))
        Rs.append(R)
        Cs.append(np.linalg.inv(R))
    W, Cm, R = Ws[0], Cs[0], Rs[0]
    for k in range(1, len(parts)):
        W, Cm, R = aa_step(W, Cm, R, Ws[k], Cs[k], Rs[k])
    X, Y = np.vstack(parts), _onehot(np.concatenate(labels), range(C))
    assert np.allclose(W, np.linalg.solve(X.T @ X + len(parts) * gamma * np.eye(d), X.T @ Y))
    assert np.allclose(clean_regularization(W, Cm, len(parts), gamma), np.linalg.lstsq(X, Y, rcond=None)[0])


def test_afl_singular_gram_is_refused():
    X = np.random.default_rng(0).normal(size=(3, 6))
    Cm = X.T @ X + 2 * np.eye(6)
    with pytest.raises(SingularGramError, match="rank 3 < d = 6"):
        clean_regularization(np.zeros((6, 2)), Cm, 2, 1.0)


@pytest.mark.parametrize("clean_reg", [True, False])
def test_afl_after_every_task_is_least_squares_or_ridge_on_all_seen_data(synthetic_root, tmp_path, clean_reg):
    for i, scen in enumerate(SCENARIOS):
        s = _run(AFL, synthetic_root, tmp_path, f"a{i}", scenario=scen, hparams={"clean_reg": clean_reg})
        uploads = 0
        for t, h in enumerate(s.heads):
            uploads += sum(1 for c in s.clients if len(s.scenario.train_rows[t][c.id]))
            Z, y = _train_data(synthetic_root, h["classes"])
            Y = _onehot(y, h["classes"])
            if clean_reg:
                W = np.linalg.lstsq(Z, Y, rcond=None)[0]
            else:
                W = np.linalg.solve(Z.T @ Z + uploads * np.eye(Z.shape[1]), Z.T @ Y)
            assert _close(h["W"], W, rel=1e-7)


# ---------------------------------------------------------------------------------------------- shared
@pytest.mark.parametrize("cls", [Fed3R, STSA, AFL])
def test_partition_invariant_methods_give_the_same_final_head_for_any_client_split(synthetic_root, tmp_path, cls):
    hp = {"rp_dim": 40, "ridge": 1.0} if cls is STSA else {}
    a = _run(cls, synthetic_root, tmp_path, "d", scenario=H.DIR_SCENARIO, hparams=hp)
    b = _run(cls, synthetic_root, tmp_path, "n", scenario=H.NATURAL_SCENARIO, hparams=hp)
    order = np.argsort(a.head_classes)
    order_b = np.argsort(b.head_classes)
    Wa = a.W[:, order] if a.W.shape[0] != len(order) else a.W[order]
    Wb = b.W[:, order_b] if b.W.shape[0] != len(order_b) else b.W[order_b]
    assert np.allclose(Wa, Wb, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("cls,keys", [(Fed3R, ["A", "b", "classes"]), (FedCOF, ["classes", "counts", "means"]),
                                      (STSA, ["C", "G", "classes", "counts"]),
                                      (STSA_E, ["classes", "counts", "group", "num_groups", "sums"]),
                                      (AFL, ["R", "W", "classes", "counts"])])
def test_uploads_hold_the_method_statistics_of_current_task_classes_only(synthetic_root, tmp_path, cls, keys):
    hp = {"rp_dim": 40, "ridge": 1.0} if cls in (STSA, STSA_E) else {}
    s = H.build_server(cls, H.make_cfg(synthetic_root["root"], synthetic_root["results"], cls.__name__, hparams=hp),
                       str(tmp_path / "exp"))
    s.init_algorithm()
    c = s.clients[0]
    c._begin_task(0)
    down = s.download_payload(0, 0, c)
    if down is not None:
        assert sorted(down) == ["rp_seed"] and wire_nbytes(down) == 8
        c.receive(0, 0, down)
    c.train_round(0, 0)
    p = c.upload_payload(0, 0)
    assert sorted(p) == keys
    assert set(np.asarray(p["classes"]).tolist()) <= set(s.scenario.tasks[0].tolist())
    for v in p.values():
        if isinstance(v, Symmetric):
            assert np.allclose(v.matrix, v.matrix.T)


@pytest.mark.parametrize("cls", [Fed3R, FedCOF, STSA, STSA_E, AFL])
def test_end_to_end_run_is_a_reference_baseline(synthetic_root, tmp_path, cls):
    hp = {"rp_dim": 40, "ridge": 1.0} if cls in (STSA, STSA_E) else {}
    s = H.run_server(cls, H.make_cfg(synthetic_root["root"], synthetic_root["results"], cls.__name__, hparams=hp),
                     str(tmp_path / "exp"))
    final = H.read_json(os.path.join(s.run_dir, "final.json"))
    assert final["run_flags"] == ["REFERENCE"]
    assert final["metrics"]["acc_final"] > 0.6


def test_symmetric_messages_are_rebuilt_from_their_upper_triangle():
    M = np.arange(9.0).reshape(3, 3)
    S = symmetrize_upper(M)
    assert np.array_equal(S, S.T) and np.array_equal(np.triu(S), np.triu(M))
