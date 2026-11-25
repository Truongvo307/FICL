# Closed-form FCIL baselines: Fed3R, FedCOF, STSA, STSA-E, AFL

This page is the integration guide for five baselines added to FICL: what each one runs, where it comes from
in the official code, how it plugs into the shared pipeline, how to run and check it, and every place where
it departs from the paper or the released code.

Claim labels: **[V]** verified here, by a test or a run whose output is quoted. **[R]** read in the paper or
the official code, not re-run. **[H]** assumption.

Contents:
1. [What this adds](#1-what-this-adds)
2. [How the baselines plug into the pipeline](#2-how-the-baselines-plug-into-the-pipeline)
3. [The methods](#3-the-methods)
4. [Deviations, missing components, unsupported settings](#4-deviations-missing-components-unsupported-settings)
5. [Install, configure, run](#5-install-configure-run)
6. [Verification](#6-verification)
7. [Dependencies and licenses](#7-dependencies-and-licenses)
8. [Status](#8-status)

---

## 1. What this adds

All five are REFERENCE baselines on cached features of a frozen backbone. They use the single-round regime:
one exchange per client per task, and the classifier is computed in closed form. Nothing else in the
repository is changed except five lines in `system/main.py`. No proposed method or existing baseline is
touched.

| File | New / modified | What it holds |
|---|---|---|
| `system/flcore/closedform_utils.py` | new | Shared helpers: class positions, one-hot, class sums, upper-triangle symmetrization, seeds sent on the wire |
| `system/flcore/clients/clientfed3r.py` | new | Fed3R client: `A_k`, `b_k`, and the Fed3R-RF random features map |
| `system/flcore/servers/serverfed3r.py` | new | Fed3R server: accumulates statistics, ridge solve, `normalize_W`, prediction |
| `system/flcore/clients/clientfedcof.py` | new | FedCOF client: class means and counts (M means per class optional) |
| `system/flcore/servers/serverfedcof.py` | new | FedCOF server: unbiased covariance estimate (Eqs. 5–8), `pinv` head, prediction |
| `system/flcore/clients/clientstsa.py` | new | STSA client (`G_k`, `C_k`) and STSA-E client (class sums of `K_D` random groups), random projection |
| `system/flcore/servers/serverstsa.py` | new | STSA and STSA-E servers: spatial and temporal aggregation, Eq. 9 estimator, ridge solve, cosine prediction |
| `system/flcore/clients/clientafl.py` | new | AFL client: `R_k = (X^T X + γI)^-1`, `W_k = R_k X^T Y` |
| `system/flcore/servers/serverafl.py` | new | AFL server: Absolute Aggregation law, Regularization Intermediary, prediction |
| `system/main.py` | **modified** | Five entries in `ALGORITHMS` (nothing else) |
| `system/tests/test_closed_form_baselines.py` | new | 32 CPU tests (section 6.1) |
| `system/check_baselines_official.py` | new | Runs the official code of each method on our features and compares, task by task (section 6.2) |
| `system/configs/baselines/*.yaml` | new | 5 smoke configs (synthetic data) and 5 CIFAR-100 configs with the shared setting |
| `system/run_baselines.sh` | new | Dry-runs, then runs all five configs of one setting and prints one table |
| `system/summarize_runs.py` | new | One table from finished experiments, grouped by scenario and features |
| `system/apply_baselines.py` | new | Registers the five entries in a `main.py` with local changes (FedCGS…) and checks the install |
| `system/baselines_manifest.sha256` | new | SHA-256 of every package file, used by `apply_baselines.py` |
| `docs/BASELINES.md` | new | This guide |

## 2. How the baselines plug into the pipeline

Each baseline is a server class and a client class on the algorithm contract of
`flcore/servers/serverbase.py` and `flcore/clients/clientbase.py`, used exactly as the existing single-round
algorithm (`SmokeClassMean`) uses it. The contract fields are:

- `REGIME = SINGLE_ROUND`
- `INPUT = FEATURES`
- `CLIENT_CLASS`
- `RUN_FLAGS = ("REFERENCE",)`
- `HPARAMS`, with typed and bounded `HParam` entries

`main.py` resolves the name through `ALGORITHMS`. `get_server_class` runs the contract check. Settings come
from the config, and command-line flags override them; `--dry_run` checks the settings and data. The shared
loop then does the same steps for every algorithm:

```
for each scenario seed (run.times runs):
  scenario files  -> partition, task order, client data per task   (identical for every method)
  feature cache   -> FeatureStore (identical for every method)
  for task t:
    every active client: download_payload -> receive -> train_round -> upload_payload   (bytes logged)
    server: aggregate(t, uploads) -> on_end_task(t)  (closed-form head over the classes seen so far)
    shared evaluation: test images of every task 0..t, prediction among the seen classes, no task id
```

The pieces the shared loop owns, so the baselines cannot vary them:

- **Partitioning, task sequence and client data.** These come from the scenario files written by
  `dataset/generate_scenario.py`.
- **Seeds.** The scenario seed is `run.seed + i`, for `i < run.times`. All three runs are reported, never a
  selected one (`summarize_runs.py` reads the mean and SD of `avg/summary.json`).
- **Evaluation.** Accuracy on all seen classes, without task ids, final and average incremental, plus the
  per-client values. This is the code that evaluates every other algorithm.
- **Communication.** Bytes are counted from each payload by the shared rule: floats 4 B, ints 8 B, and a
  `Symmetric` matrix counts its upper triangle.
- **Randomness.** Each method draws from `self.rng(tag, ...)` on the server and `self.rng(t, r)` on the
  client. Both are derived from the run seed, so a run is reproducible and other algorithms' streams are
  untouched.

Client participation: the single-round loop requires every client that holds data of task `t` to upload
once in that task. It raises an error otherwise; see section 4 for partial participation.

## 3. The methods

Equations and constants were checked against the paper and the official code at the commits below. The
server module docstrings repeat them next to the code.

### 3.1 Fed3R and Fed3R-RF

| | |
|---|---|
| Paper | Fanì, Camoriano, Caputo, Ciccone, *Accelerating Heterogeneous Federated Learning with Closed-form Classifiers*, ICML 2024, arXiv:2406.01116 |
| Code | github.com/Erosinho13/Fed3R, commit `c2a8e5d` (2025-06-06), `src/fed3r.py`: `fed3r_step`, `calc_ridge_params`, `normalize_W` |
| Client | `A_k = Σ [z;1][z;1]^T`, `b_k = Σ [z;1] y^T` over the current task's samples (Fed3R-RF: `z → φ(z)`) |
| Server | `A = λI + Σ A_k` over the d+1 entries (bias included), `b = Σ b_k`; `[W; bias] = A^-1 b`; then each column `c` of `W` and `bias_c` are divided by `‖W_c‖` (0 becomes 1e-6) |
| Predict | `argmax_c z·W_c + bias_c` |
| Hyperparameters | `ridge_lambda` 0.01 (paper and configs); `rf_dim` 0, which gives Fed3R; D > 0 gives Fed3R-RF (paper: 5000, 10000); `rf_sigma` 1000 (paper) |

**FCIL use.** The statistics are sums over samples, so after each task the server holds them for all data
of tasks 0..t. The head after task `t` is exactly Fed3R on all seen data, for any client split [V] (test
`test_fed3r_partition_invariance`).

### 3.2 FedCOF

| | |
|---|---|
| Paper | Goswami, Magistri, Wang, Twardowski, Bagdanov, van de Weijer, *Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning*, NeurIPS 2025, arXiv:2412.14326 |
| Code | github.com/dipamgoswami/FedCOF, commit `960014f` (2026-11-03), `federate_classes.py`: `FedCOFClient`, `FedCOFServer.aggregate`, `get_classifier_weights` |
| Client | Class means `μ_{k,c}` and counts `n_{k,c}` of the current task |
| Server | `μ_c = Σ n μ / N_c`; `Σ_c = 1/(K_c−1) Σ_k n_{k,c}(μ_{k,c}−μ_c)(μ_{k,c}−μ_c)^T + γI` (divides by `K_c` when `K_c = 1`, as the code); `G = Σ_c (N_c−1)Σ_c + N μ_g μ_g^T`; `W_c = pinv(G)(N_c μ_c)`, then rows normalized |
| Predict | `argmax_c z·W_c` (no bias) |
| Hyperparameters | `gamma_shrink` 1.0 (ViT-B/16; paper App. L and `bash_for_testmethods/*.sh`); `means_per_client` 1 (the released code) |

**FCIL use.** The server keeps every (mean, count) per class. Tasks hold disjoint classes, so after task `t`
the head is FedCOF over all means of the seen classes. `N` and `μ_g` are taken over the seen classes.

### 3.3 STSA and STSA-E

| | |
|---|---|
| Paper | Guan, Zhu, Zhou, Liu, Wang, Luo, Gu, arXiv:2506.01327, WWW 2026. v1 is titled *STSA: Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation*; v2 and v3 are titled *Enhancing Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation* |
| Code | github.com/Yuqin-G/STSA, commit `aaaa006` (2026-01-17), `methods/STSA.py`: `setup_RP`, `replace_fc1` (STSA), `replace_fc_ours3` and `compute_multiple_sum_by_class` (STSA-E); `convs/linears.py` `CosineLinear` |
| Projection | `h = ReLU(z W_rand)`, `W_rand [d, M] ~ N(0,1)`, the same for every client (the server sends an 8-byte seed); no projection when `M = d` |
| STSA | Clients: `G_k = Σ h h^T`, `C_k = Σ h y^T`. Server: `G ← G + Σ_k G_k`, `C ← C + Σ_k C_k` (Eqs. 1, 2, 4) |
| STSA-E | Clients: class sums `s_{g,i}` and counts `n_{g,i}` of `K_D` random groups of their task data. Server, with `K` = number of groups of the task: `G_t = Σ_i [A_i Σ_g s s^T / n_{g,i} − B_i C_i C_i^T]`, `A_i = (n_i−1)/(K−1)`, `B_i = (n_i−K)/((K−1)n_i)`, each set to 0 where negative (Eq. 9; the official `torch.where`) |
| Head | `W = (G + γI)^-1 C` (Eqs. 8, 10) |
| Predict | Cosine: `argmax_c ⟨h/‖h‖, W_c/‖W_c‖⟩` |
| Hyperparameters | `rp_dim` M = 1250 (paper, ViT; the README uses 1250, the command-line default is 10000); `ridge` γ = 1e6 (paper, ViT; `main.py` `args['ridge']`); `subsets` K_D = 10 (paper, ViT; the released `main.py` sets 5) |

STSA is natively a FCIL method. Its first stage is **not** implemented here (section 4, D1).

### 3.4 AFL

| | |
|---|---|
| Paper | He, Tong, Fang, Sun, Zeng, Li, Chen, Zhuang, *AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models*, CVPR 2025 |
| Code | github.com/RyougiSh1k1/AFL, commit `2a5f7ea` (2025-07-27), `CIFAR-100/afl.py`: `local_update`, `aggregation`, `clean_regularization`, `LinearAnalytic` |
| Client | `R_k = (X_k^T X_k + γI)^-1`, `W_k = R_k X_k^T Y_k` |
| Server | `C_k = R_k^-1`. Absolute Aggregation, one upload at a time, in the official `aggregation()` form (its signs differ from the printed Eqs. 7–10; both give `W = (Σ C_k)^-1 Σ C_k W_k`). The Regularization Intermediary with `K` uploads so far: `W + Kγ (C − KγI)^-1 W` (Eq. 15) |
| Predict | `argmax_c z·W_c` (no bias) |
| Hyperparameters | `rg` γ = 1; `clean_reg` true (official `run.sh`, main experiments) |

**FCIL use.** Each task's uploads are aggregated into the running `(W, C, R)` of earlier tasks, in client
order. The Regularization Intermediary is applied to a copy after each task, matching the official
aggregate-then-clean order. With it, the head equals least squares on all seen data, for any client split
[V] (test `test_afl_equals_least_squares`). The repository's `--continual` option is something different:
one run whose clients hold several class groups.

**Note shared by Fed3R, FedCOF and AFL.** These are one-shot FL methods. Their papers do not define a
class-incremental protocol. The use above is our adaptation: one exchange per task, with the server
accumulating over tasks. It relies only on their statistics being additive over samples (Fed3R, AFL) or
over class means (FedCOF), and it gives exactly the one-shot method on all data seen so far [V].

## 4. Deviations, missing components, unsupported settings

Each run's `run.json` records the hyperparameters. Through `extra_run_info` it also records the Fed3R variant
and RF seed, and STSA's projection, seed and `"first_stage_adaptation": "not implemented ..."` (D1).

| # | Method | Deviation | Effect | Status |
|---|---|---|---|---|
| D1 | STSA, STSA-E | **First-stage adaptation not implemented.** The paper and code first train the backbone with FedAvg on task 1 (AdaptFormer adapters, r = 64, s = 0.1; 10 rounds on CIFAR-100, 2 local epochs, lr 0.2), then freeze it. Here the backbone is the frozen pretrained one, the same as for every other method. | Not "STSA as published"; it is STSA's aggregation and head on frozen features. Expect lower accuracy than the paper. Recorded as `"first_stage_adaptation": "not implemented"`. | **Missing component.** Implementing it needs image input and adapter training, which is outside the frozen-feature protocol. That is a scope decision for you. |
| D2 | Fed3R-RF | Not in the released code. The map is the standard random Fourier features of Rahimi and Recht (2007), which the paper cites: `√(2/D) cos(zω + b)`, `ω ~ N(0, σ^-2 I)`, `b ~ U[0, 2π)`. The bias is appended after the map. | The paper does not print the map; this is the textbook form [H]. | Not checkable against official code; only `rf_dim = 0` is. |
| D3 | FedCOF | `means_per_client > 1` follows the paper's Appendix E (random disjoint subsets per class, at most `n // 2` of them). The released code has only M = 1. | | Default M = 1 is checked against the code; M > 1 is checked against the estimator only. |
| D4 | STSA-E | `K_D` default 10 (paper, ViT) vs 5 in the released `main.py`. | Configurable with `--subsets 5`. | Choice documented. |
| D5 | all | Arithmetic is float64 throughout. The released code is partly float32 (Fed3R statistics and solve; FedCOF `pinv`; STSA's final W; STSA-E's `G_t`; AFL `X^T X`). | At d = 768, float32 alone moves W by 1e-4 to 1e-3 relative, with ≥ 99.9 % identical predictions (section 6.2). | Intended: removes a source of noise. `--as_released` shows the size. |
| D6 | Fed3R, STSA | Only the columns of `b` and `C` for classes the client holds are sent (zero columns omitted), and the symmetric `A` and `G` are sent as their upper triangle. | Identical result; fewer bytes than sending the full matrices. | Communication numbers are therefore lower than a full-matrix count. |
| D7 | AFL | Clients send `R_k` (symmetric) and `W_k`; the server computes `C_k = R_k^-1`. The official clients pass `W_k`, `R_k` and `C_k`. | Same result up to rounding (section 6.2); one d×d matrix fewer on the wire. | |
| D8 | AFL | If `Σ X^T X` is singular (fewer samples than dimensions), the Regularization Intermediary has no defined result: the official `torch.inverse` returns meaningless numbers. Our run stops with `SingularGramError` and says so. `clean_reg false` keeps the K·γ ridge, as the official code without `--clean_reg` does. | Matters for early tasks of small or fine-grained datasets at d = 768. | Guard, not a change of method. |
| D9 | STSA | `rp_dim = d` means no projection (h = z). | Lets STSA run without random projection. | Option; the default is the paper's 1250. |
| D10 | all | Random draws (RF map, `W_rand`, STSA-E groups, FedCOF subsets) come from our seeded numpy streams, not `torch.manual_seed`. | Same distributions, different draws from the official runs. | The official check feeds our draws to the official code. |
| D11 | all | Data, partitions and task order are FICL's scenario files (Dirichlet α per task, `class_per_task`), not each paper's own split code. | This is the point of a shared protocol. | |

**Unsupported settings.** These raise an error or are not offered.
- **Partial client participation within a task.** The single-round loop requires every client holding task
  data to upload once. Methods compared under partial participation would differ in protocol.
- **Image input.** There is no backbone training (this is also why D1 is missing).
- **Task-id (task-incremental) evaluation.** Prediction is always among all seen classes.
- **A seen class with no training sample.** The run stops with "no training samples arrived for classes …".
- **STSA-E with fewer than 2 groups in a task.** The estimator divides by K − 1, so the run stops.

## 5. Install, configure, run

### 5.1 Install

From the repository root, choose one of two cases.

**(a) `system/main.py` is unchanged from GitHub `main`:** unzip the package over the repository.

```bash
cd /path/to/FICL
unzip -o ficl_closed_form_baselines.zip
cd system && python apply_baselines.py --check      # files identical, five entries present, classes resolve
```

**(b) `system/main.py` has local changes** (for example FedCGS registered): keep your `main.py`, and let
the script add the five lines.

```bash
cd /path/to/FICL
unzip -o ficl_closed_form_baselines.zip -x system/main.py
cd system && python apply_baselines.py               # adds the five ALGORITHMS entries before "FedAvg",
                                                     # keeps main.py.before_baselines, then verifies
```

Both paths were tested on clean copies: GitHub `main`, and the FedCGS state `5ea9e98`. `apply_baselines.py`
never edits an existing entry. If one of the five names is already registered with a
different target, it stops without writing.

### 5.2 Data (once per setting)

The CIFAR-100 configs share one setting: CIFAR-100, 10 tasks of 10 classes, 10 clients, Dirichlet 0.5,
ViT-B/16 IN-21K features after the final norm, CLS token, 3 scenario seeds.

```bash
cd dataset
python prepare_data.py CIFAR100 --download
python extract_features.py CIFAR100 --backbone vit_b16_in21k
python generate_scenario.py CIFAR100 dir --num_clients 10 --alpha 0.5 --class_per_task 10 --seed 0 --num_seeds 3
python make_synthetic.py          # only for the smoke configs
```

### 5.3 Run

```bash
cd system
python main.py --list_algorithms | grep -A3 -E "^(Fed3R|FedCOF|STSA|STSA_E|AFL) "   # registered, with hparams

# Sanity: synthetic data, all five, about a minute
bash run_baselines.sh smoke

# Full: CIFAR-100 shared setting, all five, 3 seeds each, then one table
bash run_baselines.sh cifar100

# One method, or other settings (any main.py flag goes to every run)
python main.py -cfg configs/baselines/fedcof_cifar100_vit_in21k_b0.5.yaml --dry_run
python main.py -cfg configs/baselines/fedcof_cifar100_vit_in21k_b0.5.yaml
bash run_baselines.sh cifar100 -alpha 0.1
bash run_baselines.sh cifar100 -bb dinov2_vitb14        # after extracting that cache
python main.py -cfg configs/baselines/fed3r_cifar100_vit_in21k_b0.5.yaml --rf_dim 5000   # Fed3R-RF
python main.py -cfg configs/baselines/stsae_cifar100_vit_in21k_b0.5.yaml --subsets 5     # released K_D

# Table of any finished experiments, grouped by scenario and features
python summarize_runs.py ../results/CIFAR100 --csv baselines.csv
```

Hyperparameter flags are generated from each `HPARAMS` entry: `--ridge_lambda`, `--rf_dim`, `--rf_sigma`,
`--gamma_shrink`, `--means_per_client`, `--rp_dim`, `--ridge`, `--subsets`, `--rg`, `--clean_reg`. Run
`python main.py --list_algorithms` to see them. The configs hold the paper values; none were tuned here,
and no test data enters any choice.

Output goes to the shared layout,
`../results/<dataset>/<scenario>/<algorithm>_REFERENCE_<backbone>-<group>_<goal>_<timestamp>/times_<i>/`.
Each run prints one line per task and `checks: N passed`. The upload and download bytes per task are in the
run's results, as for every algorithm.

## 6. Verification

Every check below compares against something independent: a central computation, a literal transcription of
the official function, or the official code itself.

### 6.1 Unit tests (CPU, no torch needed)

```bash
cd system && python -m pytest tests/test_closed_form_baselines.py -q      # 32 passed
```

What the 32 tests cover:
- **Exactness against central computations,** per task:
  - Fed3R and Fed3R-RF against the ridge solve on all seen data;
  - STSA (M = 40 and M = d) against its central form;
  - AFL against least squares and against ridge with K uploads.
- **Literal transcriptions of official functions,** in a dict-and-loop restatement:
  - FedCOF against `FedCOFServer.aggregate`;
  - STSA-E against `replace_fc_ours3`.
- **Unbiasedness** by simulation: FedCOF Proposition 1 and the STSA-E Eq. 9 estimator.
- **Partition invariance:** Fed3R, STSA and AFL give the same head for any client split.
- **Wire format:** the payload keys, and the 8-byte seed.
- **Helpers:** `split_for_means`, `dummy_groups`, and the RFF map approximating the RBF kernel.
- **End to end:** a `REFERENCE` run through `main.py` on synthetic data.

Full suites after installing: `./run_tests.sh`. The dataset side is unchanged. The system side gains exactly
32 passed tests over your current count, with the deselected count unchanged.
- **GitHub `main`:** the README lists system 102 passed, so 134 is expected. That number is computed, not
  observed, because torch is not available here.
- **A server at the FedCGS state (`5ea9e98`):** add 32 to that state's count.
- **Here, without torch [V]:**
  - GitHub `main` plus the package: system 120 passed and 1 skipped; dataset 52 passed and 5 skipped.
  - `5ea9e98` (`ficl_update_fedcgs.sh` + `ficl_fix_fedcgs_check.sh`) plus the package, installed with
    `apply_baselines.py`: system 131 passed and 1 skipped. FedCGS stays registered, and the smoke table is
    identical to the one from GitHub `main`.

### 6.2 Official code vs ours, on our features

`check_baselines_official.py` runs the official functions unchanged on our cached features, task by task,
and compares the classifier and the predictions.
- **What the script supplies:** stand-ins only for what the functions take from outside. That means a
  "backbone" returning our features, loaders over our rows, our `W_rand` and group permutations, and the
  server-object fields.
- **How the code is loaded:** Fed3R and STSA functions are extracted with `ast`, because their modules import
  wandb, tensorflow and their own packages. The FedCOF and AFL modules are imported.
- **Copying:** nothing of the official code is copied into FICL. The script reads your local clone at run
  time.

Get the official code at the commits checked (with git, or without it):

```bash
mkdir -p ~/official && cd ~/official
git clone https://github.com/Erosinho13/Fed3R      && git -C Fed3R  checkout c2a8e5de59e4b86efaa351c241a4a3357bcdab34
git clone https://github.com/dipamgoswami/FedCOF  && git -C FedCOF checkout 960014f1535b9a63f902b2f97a151b3dc3602294
git clone https://github.com/Yuqin-G/STSA          && git -C STSA   checkout aaaa006e0b63ebc637b16118925b0e19730c7d25
git clone https://github.com/RyougiSh1k1/AFL       && git -C AFL    checkout 2a5f7eaa2fe23af6e1bde3f7b4225f82462e05b9
# without git:
python - <<'EOF'
import io, os, shutil, urllib.request, zipfile
for repo, sha in [("Erosinho13/Fed3R", "c2a8e5de59e4b86efaa351c241a4a3357bcdab34"),
                  ("dipamgoswami/FedCOF", "960014f1535b9a63f902b2f97a151b3dc3602294"),
                  ("Yuqin-G/STSA", "aaaa006e0b63ebc637b16118925b0e19730c7d25"),
                  ("RyougiSh1k1/AFL", "2a5f7eaa2fe23af6e1bde3f7b4225f82462e05b9")]:
    name = repo.split("/")[1]
    data = urllib.request.urlopen(f"https://github.com/{repo}/archive/{sha}.zip").read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(".")
    shutil.rmtree(name, ignore_errors=True); os.rename(f"{name}-{sha}", name); print("ok", name, sha[:7])
EOF
```

Run it (needs torch; `--device cpu` or `cuda:0`):

```bash
cd system
for m in Fed3R:fed3r:Fed3R FedCOF:fedcof:FedCOF STSA:stsa:STSA STSA_E:stsae:STSA AFL:afl:AFL; do
  IFS=: read -r method cfg repo <<< "$m"
  python check_baselines_official.py --method $method --official ~/official/$repo \
      -cfg configs/baselines/${cfg}_cifar100_vit_in21k_b0.5.yaml --device cpu || echo "FAIL $method"
done
# the same with the smoke configs: -cfg configs/baselines/smoke_${cfg}.yaml
```

**Precision modes.**
- **Default (decides PASS/FAIL).** The official statements run as written, but torch's float32 is lifted to
  float64: the default dtype is float64, `Tensor.float()` returns float64, and the features are float64.
  PASS means `max |W_official − W_ours| / max |W_official| ≤ 1e-6`, and the same for the Fed3R bias, at every
  task.
- **`--as_released`.** Keeps the released float32 and only reports the differences and the prediction
  agreement, without a verdict.

**Results here [V].** Caveat: torch cannot be installed in this environment, so these runs executed the
official statements through a small numpy implementation of the torch calls they use. That stand-in is local
only and not part of the package. **The decisive run is yours, with real torch, using the commands above.**

The first table is the default (lifted) mode, as the maximum relative difference in W over all tasks:

| Method | Synthetic d = 16, 4 tasks | Fake CIFAR-100 cache d = 768, 10 tasks | Predictions agree |
|---|---|---|---|
| Fed3R | 0 (bias 0) | ≤ 2.7e-13 (bias ≤ 5.9e-12) | 100 % every task |
| FedCOF | ≤ 3.1e-16 | ≤ 1.0e-13 | 100 % |
| STSA (M = 1250) | ≤ 1.7e-14 | ≤ 3.7e-13 | 100 % |
| STSA-E (K_D = 10) | ≤ 1.2e-13 | ≤ 7.3e-13 | 100 % |
| AFL (RI on) | ≤ 2.6e-12 | ≤ 1.9e-8 | 100 % |

AFL's larger float64 difference comes from the chain of d×d inverses: the AA law makes one per upload, our
server computes `C_k = R_k^-1`, and RI adds one more. It stays 50 times below the tolerance.

The second table is `--as_released`, on the same 768-d cache (official float32, ours float64):

| Method | Max relative difference in W over the tasks | Predictions agree |
|---|---|---|
| Fed3R | ~1e-3 (bias ~9e-3) | ≥ 99.9 % |
| FedCOF | ~3e-5 | 100 % |
| STSA | ~1e-7 | 100 % |
| STSA-E | 7e-5 to 3.5e-4 | ≥ 99.99 % |
| AFL | 2.0e-4 to 4.1e-4 | ≥ 99.9 % |

**Negative controls [V].** Neither check passes something that differs:
- STSA-E with a different grouping than the official code's: FAIL, with a difference of ~1 and 72–88 %
  agreement.
- FedCOF with γ = 0: FAIL at task 0, because `pinv` of a rank-deficient G gives a different head.

The fake 768-d cache has random features. Its accuracies only show that both implementations agree, and say
nothing about the methods.

## 7. Dependencies and licenses

- **Running the baselines:** no new dependency. They use numpy, already required. The configs set
  `device: cpu`, because the solves are d×d (d + 1 for Fed3R, M for STSA) and run in seconds to minutes.
- **The official check:** torch (already in the FICL environment) and the four clones. Nothing else: the
  modules that need wandb and tensorflow are not imported.
- **Licenses of the official code,** at the commits above:
  - Fed3R: **GPL-3.0** (`LICENSE` file).
  - FedCOF, STSA and AFL: **no license file**. Default copyright applies, so they are read-only references.
  - FICL contains no code from any of them. The implementations are written from the papers and equations,
    and the check executes your own clone locally without redistributing it. Do not add their files to this
    repository.
- **Data and features:** nothing is distributed. Each user prepares CIFAR-100 and the feature caches locally
  under their licenses.

## 8. Status

| Baseline | Components implemented | Files | Status | Verification |
|---|---|---|---|---|
| Fed3R | Client `A_k`, `b_k` with bias; server λI start, solve, `normalize_W`; biased linear prediction | `clientfed3r.py`, `serverfed3r.py` | Complete (official code) | Unit: exact vs central ridge, partition invariance. Official: PASS at 1e-6 (d = 16, d = 768, stand-in torch) |
| Fed3R-RF | RFF map with a shared seed, then Fed3R | same | Complete for the paper; map assumed (D2) | Unit: exact vs central, RFF ≈ RBF kernel. No official code exists |
| FedCOF | Client means and counts; Eqs. 1, 5–8 estimator; `pinv` head; linear prediction with normalized rows, no bias; M > 1 option | `clientfedcof.py`, `serverfedcof.py` | Complete (official code); M > 1 from App. E (D3) | Unit: literal transcription, unbiasedness. Official: PASS at 1e-6 (both d) |
| STSA | Random projection with ReLU; spatial and temporal `G`, `C` aggregation; ridge head; cosine prediction | `clientstsa.py`, `serverstsa.py` | **Without first-stage adaptation (D1)** | Unit: exact vs central (M < d and M = d), partition invariance. Official: PASS at 1e-6 (both d) |
| STSA-E | `K_D` random groups; class sums and counts; Eq. 9 estimator with the official clamping; ridge head | same | **Without first-stage adaptation (D1)** | Unit: literal transcription of `replace_fc_ours3`, unbiasedness. Official: PASS at 1e-6 (both d) |
| AFL | Client `R_k`, `W_k`; official AA recursion; RI (Eq. 15); singular guard; linear prediction | `clientafl.py`, `serverafl.py` | Complete (official code) | Unit: equals least squares, equals ridge with K uploads, partition invariance. Official: PASS at 1e-6 (both d) |

No accuracy on real data has been produced yet: that needs the CIFAR-100 feature cache on your server
(`bash run_baselines.sh cifar100`). Paper numbers are not put in any table here. They come from other
protocols: their own splits, STSA with its first-stage adaptation, and AFL and Fed3R as one-shot FL. They
are not comparable to runs under the shared protocol.

Runtime of the official check: AFL's `aggregation()` makes one d×d inverse per upload. It is re-run over
every upload so far at every task, so at d = 768 it is the slowest check.
