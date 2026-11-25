# Checking our baselines against the official repositories and papers

This guide covers `repro/`. It runs the official code of Fed3R, FedCOF, AFL, STSA and STSA-E in its own
setting, then runs our FICL implementations on exactly the same inputs and compares them sample by sample.
It answers two separate questions:

| | Question | How | Verdict |
|---|---|---|---|
| **Q1** | Does our implementation compute the same thing as the official one? | Feed our method the features, client split and task split the official run used. Compare predictions, accuracy and classifier. | MATCH / MISMATCH, by criteria fixed in advance (section 3) |
| **Q2** | Does the official code, run on our server, reproduce the paper? | Compare the official run's accuracy with the paper's table. | No verdict. Differences above 1 point are flagged for investigation. |

Q1 is the one that decides whether we may move on. Q2 tells us whether the server environment reproduces the
published setup. If Q2 differs, the cause lies in data, preprocessing or library versions. It does not lie in
our code, because Q1 compares on identical inputs.

Claim labels: **[V]** verified here. **[R]** read in a paper or repository. **[H]** assumption.

## 1. The settings

Every check uses **CIFAR-100**. All four papers report it, and it is already on the server. Each runs in that
paper's own CIFAR-100 setting.

**FedCOF** (NeurIPS 2025, Table 3) — checks FedCOF and Fed3R.
- **Setting:** ViT-B/16, 100 clients, Dirichlet α 0.1, 5 seeds, γ = 1.
- **Official code:** the repository's `client_accumulate.py` and `server_aggregate.py`, unchanged. It runs FedNCM, Fed3R and FedCOF.
- **Comparisons:**
  - Our FedCOF against official FedCOF.
  - Our Fed3R against the official Fed3R functions (Fed3R repository) on the same features.
  - The FedCOF repository's own Fed3R variant (no bias column) and FedNCM are shown for the paper comparison only.

**AFL** (CVPR 2025, Table 2) — checks AFL.
- **Setting:** ResNet-18 (ImageNet-1k), 100 clients, Dirichlet α 0.1, γ = 1 with RI, 3 runs.
- **Official code:** `CIFAR-100/main.py`, unchanged, with the command line of the "CIFAR-100 (Original)" line in `run.sh`.
- **Comparison:** our AFL against official AFL.

**STSA** (WWW 2026, Table 2) — checks STSA and STSA-E.
- **Setting:** ViT-B/16 IN-21K with adapters, T = 10, K = 5, β 0.5, M 1250, ridge 1e6.
- **Official code:** `main.py`, unchanged, with the README command. It runs in two stages:
  - **adapted:** with the first stage. This is the paper setting.
  - **frozen:** `--com_round 0`. This is the FICL protocol.
- **Comparison:** our STSA and STSA-E against official STSA and STSA-E, on the official features of each stage.

**Fed3R** (ICML 2024, Table 6) — checks Fed3R and Fed3R-RF.
- **Setting:** MobileNetV2 (ImageNet), 100 clients each holding one class (α = 0), λ 0.01, σ 1000.
- **Official code:** the repository's model and checkpoint, its Fed3R functions and its 224-px TF preprocessing functions. The CIFAR-100 reading and the split are reconstructed [H], because the repository has no CIFAR pipeline.
- **Comparisons:**
  - Our Fed3R against the official Fed3R functions.
  - Fed3R-RF (5k, 10k) against the paper only, because there is no official RF code.

What "unchanged" means:
- **No files are edited.** The official files are never modified.
- **FedCOF** runs as its README runs it, in subprocesses.
- **AFL and STSA** run in-process through `runpy`. A few of their functions are wrapped to record what they
  compute: the features that enter the closed form, the labels, the official predictions and the classifier.
  Each wrapper calls the original and returns its result untouched.
- **Random stream:** the extra passes that record embeddings use their own `torch.Generator`, so the
  official random stream is not changed.
- **STSA-E:** the README asks to "set `args["type"] = 1` in main.py". The same value is set on the learner's
  `args` right after `main.py` creates it.
- **Compatibility shims,** recorded in each dump:
  - `numpy.mat = numpy.asmatrix` on numpy ≥ 2, because AFL uses `np.mat`.
  - A `wandb` placeholder when wandb is not installed. It is imported at module level and never called with
    `--wandb 0`.

## 2. How a comparison runs

```
official run (official environment, GPU)            ours (FICL environment, numpy)
------------------------------------------          --------------------------------------------------
official code, paper command line                   import_dump.py: dump -> FICL dataset
  + recording wrappers                                (cl_index, float16 cache, scenario = official split)
  -> dump .npz:                                     run_ours.py: FICL main.py classes, shared loop,
     features (float32, official row order)           hyperparameters of the official run
     client and task of every row                     exact:  the official float32 features are read
     test features, official predictions              cache:  FICL's float16 cache (as all FICL runs)
     official classifier (and W_rand, groups)       -> per task: agreement, accuracy difference,
                                                       relative difference of the classifier, verdict
```

Two replays remove randomness that is not part of the method:
- **STSA and STSA-E** use the official random projection `W_rand`. It has the same distribution as ours.
- **STSA-E** uses the official dummy groups (`compare_mode: replay`). A second entry draws our own groups
  (`own_groups`). It is reported without a verdict, because the estimator is random.

## 3. Pre-declared criteria

These are set in `repro/common.py` and were fixed before any official run.

- **MATCH (Q1), required at every task and every seed:**
  - at least **99.5 %** of the test predictions are identical to the official ones;
  - the accuracies differ by at most **0.2 points**.
- **Why these limits:** the official code is partly float32 and ours is float64. The function-level check of
  `docs/BASELINES.md` (section 6.2) measured at most 0.1 % of predictions changing at d = 768 for that reason.
- **Precision:** only the `exact` precision is judged. `cache` shows the effect of FICL's float16 storage and
  is not judged.
- **Q2 flag:** an official mean over seeds more than **1.0 point** from the paper value is flagged. Nothing is
  tuned to remove a flag. The paper values are [R] and are never copied into FICL result tables
  (`repro/paper_numbers.json`).

## 4. Environment, data, time

**Python packages for the official code:**

| Repository | Packages | Pins in the repository |
|---|---|---|
| FedCOF | torch, torchvision, timm, tqdm, pandas | python 3.8, torch 2.1.2, torchvision 0.16.2 |
| AFL | torch, torchvision, scikit-learn, ujson | `requirements.yaml`: torch 2.0.1 |
| STSA | torch, torchvision, timm, easydict, kornia, scikit-learn, pandas, tensorboard | none |
| Fed3R | torch, torchvision, timm, torchinfo; tensorflow optional (official crop functions) | `requirements.yml`: torch 2.5.1, timm 1.0.14, tensorflow 2.18.0, torchinfo 1.8.0 |

Start with the FICL environment plus the missing packages. Each script checks its imports first and names
what is missing. If an official script fails on a library version, make an environment with that
repository's pins and pass it with `--python` (or as `OFFICIAL_PYTHON` to `run_all.sh`). `run_ours.py` and
`summarize.py` always run in the FICL environment.

**Data and network:**
- **CIFAR-100:** downloaded by torchvision. FedCOF, AFL and Fed3R put it in the folder you give with `--data`.
  STSA's code downloads it into `~/data/cifar100/` (its own fixed path).
- **Pre-trained weights:** timm `vit_base_patch16_224_in21k` (FedCOF, STSA), torchvision `resnet18` (AFL), and
  the MobileNetV2 checkpoint that ships with the Fed3R repository.
- **Network access:** needed to github.com for `fetch_official.py`, and to the weight and dataset hosts.

**Disk [H, estimated from array sizes]:** about 5 GB in total.
- Official FedCOF features: 5 × 0.2 GB.
- Dumps: about 2 GB.
- Imported FICL datasets: about 1 GB (float16).

**Time [H, rough, one recent GPU]:** about 3 GPU-hours in total.

| Run | Estimate |
|---|---|
| FedCOF, 5 seeds | 30 to 45 min |
| AFL, 3 seeds | 20 to 30 min |
| Fed3R (TF preprocessing on the CPU) | 15 to 25 min |
| STSA, 2 variants × 2 stages | 60 to 90 min |
| `run_ours.py` over all dumps, CPU | 10 to 30 min (Fed3R-RF 10k builds a 10001² matrix, about 3 GB of RAM) |

## 5. Commands

From the repository root. Each step skips work whose output exists.

```bash
python repro/fetch_official.py                         # official repos at the checked commits -> repro/official_src/

# official runs (official environment, GPU); each writes repro/work/dumps/*.npz
python repro/run_fedcof_official.py --data /data/cifar --seeds 0 1 2 3 4
python repro/run_afl_official.py   --data /data/cifar --seeds 1 2 3
python repro/run_fed3r_cifar.py    --data /data/cifar --seed 0
python repro/run_stsa_official.py                      # STSA, STSA-E x adapted, frozen

# ours on the same inputs (FICL environment), then one table
for f in repro/work/dumps/*.npz; do python repro/run_ours.py "$f"; done
python repro/summarize.py --md repro/work/REPRO_SUMMARY.md

# or everything:  bash repro/run_all.sh /data/cifar [OFFICIAL_PYTHON] [FICL_PYTHON]
```

**Faster first pass:** use one seed for FedCOF (`--seeds 0`) and the frozen STSA stage
(`--stages frozen --variants STSA`). This exercises every code path before the long runs.

**Self-test of the replay** (FICL environment, no GPU, seconds):

```bash
python -m pytest repro/tests -q             # 10 passed
```

## 6. Reading the result

`run_ours.py` prints one block per dump, for example:

```
fedcof_cifar100_vit_a0.1_K100_seed0: dipamgoswami/FedCOF @ 960014f  (CIFAR-100, ..., 100 clients, 1 task(s))
  ours        vs official  prec   acc official  acc ours  max |diff|  min agree  max W diff  verdict
  FedCOF      FedCOF       exact      ...         ...        ...         ...        ...      MATCH
  paper fedcof_neurips25_t3_vit_fedcof (final): 73.20 [R] | official run, this seed ... | diff ...
```

`summarize.py` groups the seeds of each setting and shows every seed's verdict (MATCH k/n), with mean ± SD.
No seed is dropped.

**Reading `max W diff`:**
- **Rounding:** float32 rounding in the official code gives about 1e-4 to 1e-3 (up to about 1e-2 for Fed3R's
  bias at d = 768, docs/BASELINES.md 6.2).
- **Review:** a MATCH with a clearly larger classifier difference should be reviewed before it is relied on.
  The predictions can survive a small change of the method.

**If a row says MISMATCH:**
- **Look first at the replay itself.** Does `max W diff` show a different classifier (above about 1e-2) or
  rounding (about 1e-4)? Is it every task or one task?
- **Look at the precision.** A low agreement with a tiny W difference points at float32 near-ties.
- **Send the JSON.** Send `repro/work/results/<dump>.json` and the official log in `repro/work/<method>/logs/`.
- **Do not change hyperparameters to make a row pass.**

**If the paper difference is flagged:** compare the library versions with the repository's pins. The flag
says nothing about our implementation.

## 7. What has been verified here, and what is pending

**[V], in this environment (no torch available; the official code ran on a numpy stand-in of the torch
functions it uses, local only and not part of the package):**
- **Replay machinery:** `repro/tests`, 10 tests.
  - Synthetic dumps built from literal numpy restatements of the official computations are reproduced exactly
    through FICL's loop: 100 % agreement, classifier differences ≤ 1e-12.
  - This covers FedCOF, Fed3R, AFL, STSA, STSA-E with replayed groups, and Fed3R-RF without official code.
  - A changed γ on the official FedCOF side gives MISMATCH: 96.7 % agreement and 3.3 points.
  - A changed STSA ridge (1 → 2) leaves the predictions of the small synthetic set unchanged, so it shows as
    MATCH. The classifier difference reveals it: up to 6.6e-2, against ≤ 1e-12 for the true replay.
  - This is why a verdict is never read without the `max W diff` column, and why every dump carries the
    hyperparameters read from the official run into our run.
- **FedCOF dump builder:** fed with fake `client_accumulate.py` outputs (100 clients, d = 768). It ran the
  official `federate_classes.py` and the official Fed3R functions. Replay: FedCOF and Fed3R MATCH, with
  100 % agreement and W difference ≤ 1e-6.
- **AFL hook:** run on the official `afl.py` with a stand-in `main.py` that calls it in the official order
  (100 classes, 6 clients). The recording reproduced the official run, and the replay gave MATCH.
- **STSA hook:** run on the official `replace_fc1`, `replace_fc_ours3` and `compute_multiple_sum_by_class`
  inside a stand-in learner (3 tasks, 3 clients, M = 24), for both stages and both variants.
  - The hook's own checks passed: the reconstructed groups reproduce the official group sums, and the
    recorded inputs reproduce the official predictions.
  - The replay gave MATCH for all four, with W differences ≤ 1e-4 (official float32).
  - This test found a real bug in the hook (`torch.stack` on a tensor), which is fixed.
- **Lint:** ruff, with no errors in the pyflakes or bugbear classes.

**Pending, on the server:**
- **The official runs with real torch, real data and real backbones.** This is the decisive step: Q1 and Q2
  as above.
- **Parts that could not be exercised here:** the dataset loaders, the timm and torchvision models, the
  `--com_round 0` checkpoint path of the real STSA learner, and the TF preprocessing of the Fed3R script.
- **Safety checks:** each official script checks its recordings and stops with a message instead of writing a
  wrong dump:
  - client and sample counts;
  - the STSA-E group sums;
  - the official predictions recomputed from the recorded inputs (STSA);
  - FedCOF's predictions against the accuracy `server_aggregate.py` printed.
- **P1:** confirm the [R] values of `repro/paper_numbers.json` against the PDFs. They were read by automated
  extraction.

## 8. Differences between papers and code that these runs expose (not ours)

- **FedCOF.** The paper says the ViT-B/16 is "pre-trained on ImageNet-1k". The code loads
  `vit_base_patch16_224_in21k` (ImageNet-21k). The run follows the code.
- **FedCOF.** The repository's Fed3R has no bias column and uses `pinv(G + λI)`. The official Fed3R has a bias
  column and uses `solve`. Our Fed3R follows the Fed3R repository, so it is compared with that, and the
  FedCOF-repository variant appears only next to the FedCOF paper's Fed3R value.
- **STSA-E.** The paper says K_D = 10 for ViT. The released `main.py` sets `dummy = 5`. The official run, and
  therefore our replay, use 5. Our FICL configs keep 10 (docs/BASELINES.md D4).
- **STSA.** `main.sh` takes a seed argument but does not pass it to `main.py`, so the official runs use seed 0.
- **AFL.** The active line of `run.sh` is a later "continual" setup (10 clients, 4 tasks of 20 random classes).
  The paper's setting is the commented "Original" line, which is the one used here.
- **Fed3R.** There is no CIFAR-100 code in the repository. The pipeline is reconstructed from the paper's
  Appendix C and the repository's 224-px Landmarks functions [H], so the Q2 comparison is indicative only.

## 9. Files

| File | What it does |
|---|---|
| `repro/fetch_official.py` | Downloads the four repositories at the checked commits (GitHub archives, no git) |
| `repro/run_fedcof_official.py` | FedCOF paper setting: official scripts, then a dump with official FedCOF, FedNCM, the repository's Fed3R variant and the Fed3R-repository functions |
| `repro/run_afl_official.py` | AFL paper setting: official `main.py` with recording wrappers, then a dump |
| `repro/run_stsa_official.py` | STSA / STSA-E, adapted and frozen: official `main.py` with recording wrappers, then dumps |
| `repro/run_fed3r_cifar.py` | Fed3R Table 6 setting with the repository's model, checkpoint, functions and preprocessing |
| `repro/official_common.py` | Subprocess and log helpers, compatibility shims, extraction of official functions |
| `repro/stubs/wandb/` | Placeholder used only when wandb is not installed |
| `repro/common.py` | Dump format, checks, the pre-declared criteria (numpy only) |
| `repro/import_dump.py` | Dump → FICL dataset (cl_index, float16 cache, scenario with the official split) |
| `repro/run_ours.py` | Our method through FICL's loop on the dump; comparison and verdict |
| `repro/summarize.py` | One table over all results, seeds grouped |
| `repro/paper_numbers.json` | [R] values with paper, table and caption |
| `repro/run_all.sh` | All steps in order |
| `repro/tests/` | Replay tests with numpy restatements of the official computations |
| `repro/.gitignore` | Keeps `official_src/` and `work/` out of the repository |

Nothing in `system/` or `dataset/` changes. The official code stays in `repro/official_src/` under its own
licenses (Fed3R GPL-3.0; FedCOF, STSA and AFL have no license file) and is never committed or redistributed.
The same holds for the data and features under `repro/work/`.
