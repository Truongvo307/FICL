"""FedCGS on cached features: the paper's formulas, exact federated = centralized statistics after every task,
the upload and its bytes, the singular-covariance guard, and an end-to-end run."""
import os

import numpy as np
import pytest

import helpers as H
from flcore.clients.clientfedcgs import client_statistics
from flcore.servers.serverfedcgs import FedCGS, SingularCovarianceError, fedcgs_head
from utils.comm_utils import Symmetric, wire_nbytes


def _gaussian_data(rng, n_per_class=(40, 25, 60), d=6):
    means = rng.normal(size=(len(n_per_class), d)) * 2
    Z = np.concatenate([rng.normal(size=(n, d)) + means[j] for j, n in enumerate(n_per_class)])
    y = np.concatenate([np.full(n, j) for j, n in enumerate(n_per_class)])
    return Z, y


def test_head_follows_the_paper():
    """W_j = S^-1 mu_j, b_j = -1/2 mu_j S^-1 mu_j + log(N_j / N), with S the total covariance (N - 1), and the
    prediction equals the argmax of the Gaussian log-likelihood with that shared covariance plus log prior."""
    rng = np.random.default_rng(0)
    Z, y = _gaussian_data(rng)
    classes = np.arange(3)
    sums, counts, gram = client_statistics(Z, y, classes)
    W, b, info = fedcgs_head(sums, counts, gram)
    S = np.cov(Z, rowvar=False, ddof=1)                  # total covariance around the global mean
    mu = np.stack([Z[y == j].mean(axis=0) for j in classes])
    prior = counts / counts.sum()
    assert np.allclose(W, np.linalg.solve(S, mu.T).T, rtol=1e-10, atol=1e-12)
    assert np.allclose(b, -0.5 * np.einsum("ij,ij->i", mu, np.linalg.solve(S, mu.T).T) + np.log(prior),
                       rtol=1e-10, atol=1e-12)
    Q = rng.normal(size=(200, Z.shape[1])) * 3
    diff = Q[:, None, :] - mu[None]
    loglik = -0.5 * np.einsum("nkd,de,nke->nk", diff, np.linalg.inv(S), diff) + np.log(prior)
    assert np.array_equal(np.argmax(Q @ W.T + b, axis=1), np.argmax(loglik, axis=1))
    assert info["N"] == len(y) and info["rank"] == Z.shape[1]



def _centred_predict(W_b_info, Q, classes):
    _W, _b, info = W_b_info
    return classes[np.argmax((Q - info["mean"]) @ info["W_centered"].T + info["b_centered"], axis=1)]


def test_centred_form_is_the_same_classifier_and_ignores_a_common_offset():
    """predict() uses the classifier written around the global mean: the same argmax as z.W + b, and (as
    FedCGS itself) unchanged when every feature is shifted by the same vector, with an ill-conditioned
    covariance (condition number ~1e8)."""
    rng = np.random.default_rng(3)
    Z, y = _gaussian_data(rng)
    classes = np.arange(3)
    W, b, info = fedcgs_head(*client_statistics(Z, y, classes))
    Q = rng.normal(size=(300, Z.shape[1])) * 3
    assert np.array_equal(_centred_predict((W, b, info), Q, classes), np.argmax(Q @ W.T + b, axis=1))
    shift = (Q @ W.T + b) - ((Q - info["mean"]) @ info["W_centered"].T + info["b_centered"])
    assert np.allclose(shift, shift[:, :1], rtol=1e-9, atol=1e-9)        # the same for every class

    d = 40
    Qr, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sd = np.logspace(0, -4, d)
    means = (rng.normal(size=(4, d)) * sd) @ Qr.T
    y2 = np.repeat(np.arange(4), 500)
    Z2 = means[y2] + (rng.normal(size=(len(y2), d)) * sd) @ Qr.T
    T2 = means[np.repeat(np.arange(4), 50)] + (rng.normal(size=(200, d)) * sd) @ Qr.T
    offset = rng.normal(size=d) * 10
    ref = _centred_predict(fedcgs_head(*client_statistics(Z2, y2, np.arange(4))), T2, np.arange(4))
    head = fedcgs_head(*client_statistics(Z2 + offset, y2, np.arange(4)))
    assert head[2]["condition_number"] > 1e7
    assert np.array_equal(_centred_predict(head, T2 + offset, np.arange(4)), ref)

def test_statistics_add_over_any_split_of_the_samples():
    rng = np.random.default_rng(1)
    Z, y = _gaussian_data(rng)
    classes = np.arange(3)
    whole = client_statistics(Z, y, classes)
    parts = np.array_split(rng.permutation(len(y)), 4)
    acc = None
    for p in parts:
        acc = client_statistics(Z[p], y[p], classes, *((acc[2], acc[0], acc[1]) if acc else ()))
    for a, b in zip(whole, acc):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_singular_covariance_is_refused_and_gamma_is_explicit():
    rng = np.random.default_rng(2)
    Z, y = _gaussian_data(rng, n_per_class=(3, 3), d=16)        # N - 1 = 5 < d = 16
    sums, counts, gram = client_statistics(Z, y, np.arange(2))
    with pytest.raises(SingularCovarianceError, match="rank 5 < d = 16"):
        fedcgs_head(sums, counts, gram, gamma=0.0)
    W, _b, info = fedcgs_head(sums, counts, gram, gamma=0.1)
    assert np.all(np.isfinite(W)) and info["rank"] == 16


class _Snapshots(FedCGS):
    def init_algorithm(self):
        super().init_algorithm()
        self.heads = []

    def on_end_task(self, t):
        super().on_end_task(t)
        self.heads.append((self.head_classes.copy(), self.W.copy(), self.b.copy()))


def _central_head(sr, classes):
    """FedCGS fitted centrally on every training sample of `classes` (block 3, cls token of the cache)."""
    index = sr["index"]
    feats = np.load(sr["cache_file"], mmap_mode="r")[:, 3, 0, :].astype(np.float64)
    train = (index.split == 0) & np.isin(index.labels, classes)
    Z, y = feats[train], index.labels[train]
    order = np.argsort(classes)
    sums, counts, gram = client_statistics(Z, y, np.asarray(classes)[order])
    inv = np.argsort(order)
    return fedcgs_head(sums[inv], counts[inv], gram)


def test_federated_head_equals_centralized_fedcgs_on_the_seen_data_after_every_task(synthetic_root, tmp_path):
    """The class-incremental adaptation: after task t, the head is official FedCGS on all training data of
    the classes seen so far, whatever the client split (Dirichlet or natural domains)."""
    for i, scen in enumerate((H.DIR_SCENARIO, H.NATURAL_SCENARIO)):
        s = H.run_server(_Snapshots, H.make_cfg(synthetic_root["root"], synthetic_root["results"], "_Snapshots",
                                                scenario=scen), str(tmp_path / f"exp{i}"))
        assert len(s.heads) == s.num_tasks
        for t, (classes, W, b) in enumerate(s.heads):
            assert np.array_equal(classes, s.scenario.seen_classes(t))
            Wc, bc, _ = _central_head(synthetic_root, classes)
            assert np.max(np.abs(W - Wc)) <= 1e-9 * np.max(np.abs(Wc))
            assert np.max(np.abs(b - bc)) <= 1e-9 * np.max(np.abs(bc))


def test_upload_is_the_paper_statistics_and_its_bytes(synthetic_root, tmp_path):
    s = H.build_server(FedCGS, H.make_cfg(synthetic_root["root"], synthetic_root["results"], "FedCGS"),
                       str(tmp_path / "exp"))
    s.init_algorithm()
    c = s.clients[0]
    c._begin_task(0)
    c.train_round(0, 0)
    p = c.upload_payload(0, 0)
    assert sorted(p) == ["classes", "counts", "gram", "sums"] and isinstance(p["gram"], Symmetric)
    m, d = len(p["classes"]), s.store.dim
    assert set(p["classes"].tolist()) <= set(s.scenario.tasks[0].tolist())
    assert wire_nbytes(p) == 4 * d * (d + 1) // 2 + 4 * m * d + 8 * m + 8 * m
    Z, y = c.load_task_data()
    sums, counts, gram = client_statistics(Z, y, np.sort(s.scenario.tasks[0]))
    keep = counts > 0
    assert np.allclose(p["sums"], sums[keep]) and np.array_equal(p["counts"], counts[keep])
    assert np.allclose(p["gram"].matrix, gram)


def test_fedcgs_end_to_end(synthetic_root, tmp_path):
    s = H.run_server(FedCGS, H.make_cfg(synthetic_root["root"], synthetic_root["results"], "FedCGS"),
                     str(tmp_path / "exp"))
    final = H.read_json(os.path.join(s.run_dir, "final.json"))
    assert final["run_flags"] == ["REFERENCE"]
    assert final["metrics"]["acc_final"] > 0.8
    assert s.hp == {"gamma": 0.0}


def test_official_check_finds_the_statistics_block(tmp_path):
    """check_fedcgs_official.py executes the official statements from `cs_l = []` to the gamma loop; here on a
    stand-in file with the same structure (the official code itself is not part of this repository)."""
    import check_fedcgs_official as chk

    (tmp_path / "utils_fl.py").write_text("def eval(*a):\n    return 0\n\ndef test(*a, **k):\n    return 0\n")
    (tmp_path / "main_fedcgs.py").write_text(
        "if __name__ == '__main__':\n"
        "    for dataset in ['x']:\n"
        "        for beta in [0.1]:\n"
        "            model = None\n"
        "            cs_l = []\n"
        "            total = 1 + 1\n"
        "            gamma_list = [0]\n"
        "            for gamma in gamma_list:\n"
        "                seen = gamma\n")
    utils_fl, code, segment = chk.load_official(str(tmp_path))
    assert segment.splitlines()[0] == "cs_l = []" and "for gamma in gamma_list:" in segment
    assert "model = None" not in segment
    ns = {}
    exec(code, ns)
    assert ns["total"] == 2 and ns["seen"] == 0 and callable(utils_fl.eval)
    with pytest.raises(SystemExit, match="no main_fedcgs.py"):
        chk.load_official(str(tmp_path / "missing"))
