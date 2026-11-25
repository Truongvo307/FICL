# FICL: Federated Class-Incremental Learning

FICL is a simulation library for **federated class-incremental learning**. K clients learn a
sequence of tasks; each task brings new classes, and the clients never share their data. After
every task the model is tested on all classes seen so far, without being told which task a test
image comes from.

The library follows the layout of [PFLlib](https://github.com/TsingZ0/PFLlib) and
[HtFLlib](https://github.com/TsingZ0/HtFLlib): prepare the data once in `dataset/`, then run
experiments from `system/` with `python main.py`. It covers the three families of federated
class-incremental methods, in one loop with one protocol:

1. **Frozen pretrained backbone.** Features are extracted once into a cache; algorithms (closed-form
   heads, or heads trained over communication rounds) run on the cached features.
2. **Pretrained backbone, trained.** A pretrained ViT is trained on the raw images: only its head, its
   last blocks, LoRA factors, prompt tokens, or every weight.
3. **From scratch.** A CNN (ResNet-18 or ResNet-32 with the CIFAR stem) is trained from a random start on
   the raw images.

## What it supports

| | |
|---|---|
| **Learning setting** | Class-incremental: every task brings new classes to all clients at once; testing on all classes seen so far, with no task id |
| **Client data** | Dirichlet (`dir`, with `alpha`), `iid`, or `natural` partitions (one client per domain); each client has its own test shard |
| **Regimes** | *Single-round (closed-form)*: each client uploads once per task. *Multi-round*: R rounds per task, with local epochs, batch size, learning rate and client sampling |
| **Pretrained models** | DINOv2 ViT-B/14, CLIP ViT-B/16, ViT-B/16 ImageNet-21K. Frozen: features from any of the 12 blocks, class token and/or mean patch token, with or without the final norm, optionally randomly projected. Trained: `head`, `last_k` blocks, `lora`, `prompt` (shallow or deep) or `full` |
| **Networks from scratch** | ResNet-18 (CIFAR stem), ResNet-32 (CIFAR), and a tiny CNN for pipeline checks |
| **Datasets** | 14 datasets: general objects, ImageNet distribution shifts, fine-grained, remote sensing, medical (MedMNIST+) and multi-domain (DomainNet); plus a synthetic set for tests |
| **Evaluation** | Accuracy matrix (after task t, on the test set of task i), final and average incremental accuracy, per-client accuracy, bytes per message, wall-clock time |
| **Reproducibility** | Seeded scenarios, one seed per run, a sha256 for every data file, every setting saved with where it came from, and automatic checks of the result files |
| **Algorithms included** | `FedAvg` (sequential fine-tuning on images) and `FedAvgJoint` (its joint-training upper bound) for families 2 and 3; two pipeline-test algorithms on cached features. A new algorithm is two files |

---

## Contents

1. [Setup](#1-setup)
2. [Datasets](#2-datasets)
3. [How a dataset is split into a FICL scenario](#3-how-a-dataset-is-split-into-a-ficl-scenario)
4. [Dataset generation commands](#4-dataset-generation-commands)
5. [Running FICL experiments](#5-running-ficl-experiments)
6. [Hyperparameters](#6-hyperparameters)
7. [Models](#7-models)
8. [Results](#8-results)
9. [Writing a new algorithm](#9-writing-a-new-algorithm)
10. [Tests](#10-tests)
11. [Repository layout](#11-repository-layout)
12. [Acknowledgements and license](#12-acknowledgements-and-license)

---

## 1. Setup

**Requirements:** Linux, Python 3.11, PyTorch 2.0.1 with CUDA 11.8, and one GPU for feature
extraction and for training networks (the reference machine has one RTX 4090 and 126 GB RAM).
Algorithms on cached features run in numpy on the CPU.

**Step 1. Get the code.** Clone the repository, or copy its folder to your machine, and go into it:

```bash
git clone <repository-url> FICL
cd FICL
```

**Step 2. Create the environment** (Python 3.11, NumPy 1.x, PyTorch 2.0.1 + CUDA 11.8, timm,
open_clip, medmnist, pytest):

```bash
conda env create -f env_cuda.yaml
conda activate ficl
```

**Step 3. Check it:**

```bash
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
python -m pip check
```

The output should show NumPy `1.26.x` and torch `2.0.1+cu118`, and `pip check` should report no conflicts.

**Step 4. Run the tests** (from the repository root):

```bash
(cd dataset && python -m pytest tests -q -m "not gpu and not data")
(cd system && python -m pytest tests -q -m "not gpu")
```

**Step 5. Run a first experiment on synthetic data** (CPU, about a minute):

```bash
(cd dataset && python make_synthetic.py)
cd system
python main.py -cfg configs/smoke_single.yaml          # frozen features, closed-form: one upload per client per task
python main.py -cfg configs/smoke_multi.yaml           # frozen features, multi-round: 20 rounds per task
python main.py -cfg configs/smoke_multi_images.yaml    # images: FedAvg trains a tiny CNN from scratch
```

Each run prints the protocol, one line per task, the final values over 3 seeds and `checks: N passed,
0 failed`.

**Step 6. Record the exact versions** once: `python -m pip freeze > requirements.lock`.

Disk and memory: a feature cache takes about 40 KB per image and backbone with the default
extraction config (section 4, step 2), and a d × d statistics matrix in fp64 takes 8·d² bytes.
Pretrained weights download on first use into the torch.hub, timm (Hugging Face) and open_clip caches.
If a step fails, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

---

## 2. Datasets

`python prepare_data.py --info` (in `dataset/`) prints this list, with the licence of each dataset.
Image counts are the published totals; image size is the size of the original files (every backbone
resizes to 224 × 224).

**General objects**

| Dataset (`-data`) | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `CIFAR100` | 100 | 60,000 | 32 × 32 | Small natural images of everyday objects and animals | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| `TinyImageNet` | 200 | 110,000 | 64 × 64 | A 200-class subset of ImageNet at low resolution | [Stanford CS231n](http://cs231n.stanford.edu/) |

**Distribution shift (ImageNet variants)**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `ImageNetR` | 200 | 30,000 | variable | Renditions (art, cartoons, sketches, toys, …) of 200 ImageNet classes | [ImageNet-R](https://github.com/hendrycks/imagenet-r) |
| `ImageNetA` | 200 | 7,500 | variable | Natural adversarial examples: real photos that ImageNet classifiers get wrong | [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) |

**Fine-grained**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `CUB200` | 200 | 11,788 | variable | Photos of 200 bird species (CUB-200-2011) | [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) |

**Remote sensing**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `EuroSAT` | 10 | 27,000 | 64 × 64 | Sentinel-2 satellite patches, land use and land cover (RGB) | [EuroSAT](https://github.com/phelber/EuroSAT) |
| `RESISC45` | 45 | 31,500 | 256 × 256 | Aerial scene images (NWPU-RESISC45) | [Cheng et al., 2017](https://arxiv.org/abs/1703.00121) |

**Medical (MedMNIST+, used at 224 × 224)**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `TissueMNIST` | 8 | 236,386 | 224 × 224, gray | Kidney cortex cells, microscopy | [MedMNIST](https://medmnist.com/) |
| `OCTMNIST` | 4 | 109,309 | 224 × 224, gray | Retinal optical coherence tomography | MedMNIST |
| `OrganCMNIST` | 11 | 23,583 | 224 × 224, gray | Abdominal CT, coronal view, 11 organs | MedMNIST |
| `DermaMNIST` | 7 | 10,015 | 224 × 224, RGB | Dermatoscope images of skin lesions | MedMNIST |
| `PathMNIST` | 9 | 107,180 | 224 × 224, RGB | Colon pathology tissue patches | MedMNIST |
| `BloodMNIST` | 8 | 17,092 | 224 × 224, RGB | Blood cells, microscopy | MedMNIST |

**Multi-domain (domain shift between clients)**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `DomainNet` | 345 | about 0.59 M | variable | Six domains (clipart, infograph, painting, quickdraw, real, sketch); with `natural`, each domain is one client | [DomainNet](http://ai.bu.edu/M3SDA/) |

**Synthetic (tests)**

| Dataset | Classes | #Samples | Image size | Description | Source |
|---|---|---|---|---|---|
| `Synthetic` | 8 | 640 | 32 × 32 images; features 4 blocks × 2 tokens × 16 | Class-structured synthetic features and images (a coloured square whose place and colour give the class) in 3 domains, for smoke tests | `dataset/make_synthetic.py` |

**Train / test split.** MedMNIST, DomainNet, CIFAR100, TinyImageNet and CUB200 use their official
splits. TinyImageNet's labelled validation set serves as its test set. ImageNetR, ImageNetA, EuroSAT and
RESISC45 have no official split, so the index uses a seeded, stratified split: per class, 20% of the
images (at least one) go to test, chosen with seed 0. The rule is saved in the index.

**Licences.** Every dataset keeps its own terms (`prepare_data.py --info` lists them). DermaMNIST and
CUB200 are non-commercial; TinyImageNet follows ImageNet's terms. Never redistribute the data or its
features.

**Overlap with pretraining data.** CUB's page warns that its images overlap with ImageNet, and
TinyImageNet is cut from ImageNet, so an ImageNet-pretrained backbone may have seen their test
images. Report this next to numbers on those two datasets.

**Disk per backbone** with the default extraction config:

| Dataset | Disk | Dataset | Disk |
|---|---|---|---|
| TissueMNIST | 9.4 GB | ImageNetR | 1.2 GB |
| OCTMNIST | 4.4 GB | ImageNetA | 0.3 GB |
| OrganCMNIST | 0.9 GB | CUB200 | 0.5 GB |
| DermaMNIST | 0.4 GB | EuroSAT | 1.1 GB |
| PathMNIST | 4.3 GB | RESISC45 | 1.3 GB |
| BloodMNIST | 0.7 GB | CIFAR100 | 2.4 GB |
| DomainNet | about 23 GB | TinyImageNet | 4.4 GB |

---

## 3. How a dataset is split into a FICL scenario

A **scenario** fixes who holds which images, and when. It is written once by `generate_scenario.py`
and read by every experiment, so all algorithms see exactly the same data.

1. **Tasks.** The class order is shuffled with the seed and cut into tasks of `class_per_task` classes:
   tasks = ceil(classes / class_per_task), and the last task may be smaller. Each class belongs to
   exactly one task.
2. **Clients.** The training images of every class are shared among the clients:
   - `dir`: Dirichlet proportions per class with concentration `alpha`. Small `alpha` (e.g. 0.1) means
     each client holds few classes of a task; large `alpha` (e.g. 10) is close to even.
   - `iid`: an even random split.
   - `natural`: client k is domain k, so the number of clients is the number of domains (6 for DomainNet).
3. **Arrival.** At the start of task t, every client receives its share of task t's classes. During
   task t a client can read only that data; data of earlier tasks are gone.
4. **Test data.** The global test set of task t is the official test images of task t's classes. Each
   client also gets a test shard that follows its own training mix: per class, test images in
   proportion to its training images (with `natural`, the domain's own test images). Together the
   shards are exactly the global test set.
5. **Evaluation.** After task t, the model is tested on the test sets of tasks 0..t (all classes seen
   so far), predicting among the seen classes only.

**Options:**
- `--use_val merge`: put an official validation split into training (default `ignore`).
- `--min_train_per_client m`: redraw a task until every client has at least m training images (not for
  `natural`).
- `--class_order sorted`: tasks take the classes in label-id order (0..C−1) for every seed, as
  fed-mammoth and most benchmark code do; the seed then changes only the partition. The default,
  `shuffled`, is a seeded permutation. Folder names get `_sorted`.
- `--num_seeds n`: write seeds seed .. seed+n−1, one folder each; run i of an experiment reads seed
  `seed + i`.

Nothing is dropped or duplicated: within a task, the client shards are disjoint and cover every
training image of the task's classes. Folder names encode the settings:
`cl_<K>clients_dir_<alpha>_cpt<c>_seed<s>`, `cl_<K>clients_iid_cpt<c>_seed<s>`,
`cl_natural_cpt<c>_seed<s>`, plus `_valmerge` or `_min<m>` when those options are used.

**Example.** CIFAR100 with `--class_per_task 10 --num_clients 10 --alpha 0.1` gives 10 tasks of 10
classes. In task 0 the 5,000 training images of 10 classes are shared among 10 clients by a skewed
Dirichlet draw, and the 1,000 test images of those classes form the first global test set. After task
9 the model is tested on all 10,000 test images.

**Suggested class increments** (10 tasks where the class count allows): CIFAR100 10; ImageNetR,
ImageNetA, CUB200 and TinyImageNet 20; DomainNet 35; RESISC45 5 (9 tasks); EuroSAT 2 (5 tasks);
MedMNIST 2.

**Not supported yet:** clients that join late or leave, different class subsets per client within a
task, task-incremental evaluation (task id given) and domain-incremental streams.

---

## 4. Dataset generation commands

Run these from `dataset/`. Each step writes into `dataset/<Name>/` and never overwrites silently: a
different index needs `--force`, an existing cache must be deleted before it is rebuilt, and an
existing scenario is kept only if the new one is identical.

| Family (section 5.4) | Step 1: index | Step 2: feature cache | Step 3: scenarios |
|---|---|---|---|
| 1. Frozen pretrained (cached features) | yes | yes, once per backbone | yes |
| 2. Pretrained, trained on images | yes | no | yes |
| 3. From scratch on images | yes | no | yes |

Families 2 and 3 read the raw images in `<Name>/rawdata/` through the same adapter that built the index.
Before every run the index is rebuilt from the raw files and its sha256 compared, so every row still
means the same image (MedMNIST: the package's files fix the rows, and the sample counts are compared).

**Step 1. Sample index** (and the raw files):

```bash
cd dataset
python prepare_data.py --info                       # every dataset: size, split, download, licence
python prepare_data.py TissueMNIST --download       # MedMNIST+ at 224 px (the medmnist package downloads)
python prepare_data.py ImageNetR --download         # benchmark archives into ImageNetR/rawdata/
python prepare_data.py DomainNet --download         # six domains and the official split files (large)
python prepare_data.py RESISC45                     # after extracting NWPU-RESISC45.rar into RESISC45/rawdata/
python prepare_data.py --list                       # datasets that already have an index
```

| Flag | Default | Meaning |
|---|---|---|
| `--download` | off | download missing raw files into `<Name>/rawdata/` (all datasets except RESISC45) |
| `--size` | 224 | MedMNIST image size (64, 128 or 224) |
| `--test_fraction`, `--split_seed` | 0.2, 0 | only for datasets without an official split |
| `--force` | off | replace an index with a different one (its caches and scenarios become invalid) |
| `--dataset_root` | `dataset/` | folder that holds the dataset folders |

**Step 2. Feature cache** (GPU), once per backbone. Try a small extraction first:

```bash
python extract_features.py ImageNetR --backbone dinov2_vitb14 --limit 512    # quick check, into <group>_limit512
python extract_features.py ImageNetR --backbone dinov2_vitb14                # the full cache
python extract_features.py ImageNetR --list                                  # finished caches
```

| Flag | Default | Meaning |
|---|---|---|
| `--backbone` | — | `dinov2_vitb14`, `clip_vitb16` or `vit_b16_in21k` (section 7) |
| `--config` | `configs/extract_default.yaml` | which blocks and tokens to store, in groups |
| `--batch_size`, `--num_workers` | 256, 8 | data loading |
| `--device_id` | 0 | the GPU (`CUDA_VISIBLE_DEVICES`) |
| `--limit N` | 0 (all) | extract only the first N images, into `<group>_limitN` |

The default config stores two groups: `pre` (all 12 blocks, class token and mean patch token, before
the final norm) and `last_ln` (the last block after the final norm, i.e. the model's own output).

**Step 3. Scenarios**, one folder per seed:

```bash
python generate_scenario.py ImageNetR dir --num_clients 10 --alpha 0.1 --class_per_task 20 --seed 0 --num_seeds 3
python generate_scenario.py CIFAR100 iid --num_clients 10 --class_per_task 10 --num_seeds 3
python generate_scenario.py DomainNet natural --class_per_task 35 --num_seeds 3
python generate_scenario.py ImageNetR --list
```

| Flag | Default | Meaning |
|---|---|---|
| `dataset partition` | — | e.g. `ImageNetR dir`; partition `dir`, `iid` or `natural` |
| `--num_clients` | 10 | clients (not for `natural`) |
| `--alpha` | 0.1 | Dirichlet concentration (`dir` only) |
| `--class_per_task` | 2 | classes per task |
| `--seed`, `--num_seeds` | 0, 1 | first seed and number of seeds |
| `--use_val` | `ignore` | `merge` adds an official validation split to training |
| `--min_train_per_client` | 0 (off) | minimum training images per client and task |
| `--class_order` | `shuffled` | `sorted`: classes in label-id order for every seed |

---

## 5. Running FICL experiments

### 5.1 The protocol

Every algorithm runs inside the same loop, so every algorithm follows the same protocol:

```
for t in tasks:                       # task t: its classes arrive at the clients
    begin_task(t)
    for r in rounds of task t:        # single-round: exactly one exchange; multi-round: rounds_per_task
        select clients -> server sends -> clients train -> clients upload -> server aggregates
    end_task(t)                       # e.g. the server broadcasts the model for inference
    evaluate                          # all classes seen so far: global test set and every client's shard
```

| Regime | Inside a task | Schedule settings |
|---|---|---|
| `single_round` (closed-form) | every client uploads once | none; round settings are refused |
| `multi_round` | `rounds_per_task` rounds | `rounds_per_task`, `local_epochs`, `batch_size`, `learning_rate` (required); `join_ratio`, `eval_gap` (optional) |

The regime is a property of the algorithm. Total rounds are tasks × `rounds_per_task`; there is no
separate global-round setting. Only final values are reported (no best round, no early stopping on
test data). Algorithms that train a network on images are multi-round.

### 5.2 Running with a config file

A config file has five sections: `scenario`, then `features` (algorithms on cached features) or
`model` (algorithms that train a network on images), then `schedule`, `algorithm` and `run`. This is
`system/configs/template_single.yaml`:

```yaml
scenario:
  dataset: TissueMNIST
  partition: dir             # dir | iid | natural
  num_clients: 10
  alpha: 0.1                 # Dirichlet alpha, dir only
  class_per_task: 2          # the class increment: 8 classes -> 4 tasks
features:
  backbone: dinov2_vitb14    # the frozen pretrained model of the cache
  group: pre                 # cache group: pre (before the final norm) or last_ln
  blocks: [11]               # which blocks (0..11); several blocks are concatenated
  tokens: [cls]              # cls, mean or both
schedule:
  regime: single_round       # optional statement; must match the algorithm
algorithm:
  name: SmokeClassMean
run:
  seed: 0                    # run i reads the scenario with seed 0 + i
  times: 3                   # number of runs (seeds)
  goal: pipeline_check       # a free label in the result folder name
  device: cuda
```

```bash
cd system
python main.py -cfg configs/template_single.yaml --dry_run    # check the settings and every seed's data; run nothing
python main.py -cfg configs/template_single.yaml              # run
python main.py -cfg configs/template_single.yaml -t 1 -go try # flags override the file
```

The dry run prints the protocol: algorithm and regime, scenario, tasks, the scenario folder of every
seed, features or model, schedule, hyperparameters and evaluation. If a scenario or cache is missing, it
says which, lists what exists and prints the command that makes it. For image algorithms it also
checks the raw images against the index and decodes one of them. The other templates are
`template_multi.yaml` (cached features, multi-round), `template_multi_pretrained.yaml` (family 2) and
`template_multi_scratch.yaml` (family 3).

### 5.3 Running with flags only

```bash
python main.py -data ImageNetR -partition dir -nc 10 -alpha 0.1 -cpt 20 \
    -bb dinov2_vitb14 -fg pre -blocks 11 -tokens cls \
    -algo SmokeClassMean -t 3 -seed 0
python main.py -data ImageNetR -partition dir -nc 10 -alpha 0.1 -cpt 20 \
    -bb dinov2_vitb14 -fg pre -blocks 11 -tokens cls \
    -algo SmokeFedAvgLinear -rpt 50 -ls 1 -lbs 64 -lr 0.01 -jr 0.5 -eg 10 --weight_decay 0.0001 -t 3
```

For long runs, keep the console output: `mkdir -p logs && nohup python main.py -cfg configs/my.yaml >
logs/my.out 2>&1 &`. Each run also writes its own log file (section 8). `run.sh` (dry run, then run)
and `smoke.sh` (all smoke configs) are optional shortcuts.

Exit codes: 0 all runs completed and every check passed; 1 a run or a check failed; 2 the settings or
the data cannot be used (nothing is written).

### 5.4 The three model families: how to run each

All three families run in the same loop, on the same scenario files, with the same evaluation, result
files, checks and byte counting. They differ in what the clients read and what they train:

| Family | What the clients train | Input | Settings section | Needs in `dataset/` | Algorithms here |
|---|---|---|---|---|---|
| 1. Frozen pretrained | nothing in the backbone: statistics or a head on fixed features | cached features | `features` | index, feature cache, scenarios | `FedCGS`; `SmokeClassMean`, `SmokeFedAvgLinear` (pipeline checks) |
| 2. Pretrained, trained | a pretrained ViT: `head`, `last_k`, `lora`, `prompt` or `full`, plus a linear head | raw images | `model` | index, scenarios | `FedAvg`, `FedAvgJoint` |
| 3. From scratch | a CNN from a random start, plus a linear head | raw images | `model` | index, scenarios | `FedAvg`, `FedAvgJoint` |

An algorithm reads either cached features or images; `python main.py --list_algorithms` shows which.
Settings of the other kind are refused (for example `-bb` with `FedAvg`, or `-m` with `SmokeClassMean`).
A pretrained ViT with `trainable: head` is the frozen backbone trained through the images (with
augmentation): its head reads the same final-norm class token that a cache stores for the last block.

**Family 1: frozen pretrained features.**

```bash
cd dataset
python prepare_data.py ImageNetR --download
python extract_features.py ImageNetR --backbone dinov2_vitb14
python generate_scenario.py ImageNetR dir --num_clients 10 --alpha 0.1 --class_per_task 20 --seed 0 --num_seeds 3
cd ../system
python main.py -data ImageNetR -partition dir -nc 10 -alpha 0.1 -cpt 20 \
    -bb dinov2_vitb14 -fg pre -blocks 11 -tokens cls -algo SmokeClassMean -t 3 --dry_run
```

What to choose: the backbone and cache group, which blocks and tokens to read, and an optional random
projection (section 6.2, Features). The FedCGS baseline runs the same way, with `-algo FedCGS` (configs in
`system/configs/fedcgs/`, see [reports/FEDCGS.md](reports/FEDCGS.md)).

**Family 2: a pretrained ViT trained on the images.** No feature extraction; the weights download on
first use.

```bash
cd dataset
python prepare_data.py ImageNetR --download
python generate_scenario.py ImageNetR dir --num_clients 10 --alpha 1.0 --class_per_task 20 --seed 0 --num_seeds 3
cd ../system
python main.py -data ImageNetR -partition dir -nc 10 -alpha 1.0 -cpt 20 \
    -algo FedAvg -m vit_b16_in21k -tm lora -lrank 8 -ltgt q,v \
    -rpt 10 -ls 1 -lbs 32 -lr 0.01 --momentum 0.9 -t 3 --dry_run
```

Choose the backbone with `-m` (`vit_b16_in21k`, `dinov2_vitb14`, `clip_vitb16`; 224 px) and what
trains with `-tm`; each mode has its own options, and options of another mode are refused:

| `-tm` | Its options | What is trained |
|---|---|---|
| `head` | none | the linear head; the backbone stays frozen |
| `last_k` | `-lastk K` (required, 1 to 12) | the last K transformer blocks, the final norm and the head |
| `lora` | `-lrank R` (8), `-lalpha A` (= R), `-ltgt q,v` (from `q,k,v,proj,fc1,fc2`) | LoRA factors on those weights of every block, and the head |
| `prompt` | `-plen L` (10), `-pdeep true/false` (false) | L prompt tokens (before the first block, or new ones before every block), and the head |
| `full` | none | every weight the class token depends on, and the head |

**Family 3: a CNN trained from scratch on the images.**

```bash
cd dataset
python prepare_data.py CIFAR100 --download
python generate_scenario.py CIFAR100 dir --num_clients 10 --alpha 1.0 --class_per_task 10 --seed 0 --num_seeds 3
cd ../system
python main.py -data CIFAR100 -partition dir -nc 10 -alpha 1.0 -cpt 10 \
    -algo FedAvg -m resnet18_cifar -aug crop_flip \
    -rpt 50 -ls 2 -lbs 64 -lr 0.05 --momentum 0.9 --weight_decay 0.0005 -t 3 --dry_run
```

Choose the network with `-m` (`resnet18_cifar`, `resnet32`, or `tiny_cnn` for pipeline checks) and the
input size with `-isize` (default 32; e.g. 64 for TinyImageNet). A network from scratch always trains in
full, and the CNNs have no pretrained weights here, so `-pre true` and other `-tm` values are refused.

**The same run with a config file.** `configs/template_multi_pretrained.yaml` and
`configs/template_multi_scratch.yaml` hold every setting of families 2 and 3 with comments; copy one,
set the values from your protocol (the numbers in the templates only make the file run), then:

```bash
python main.py -cfg configs/my_run.yaml --dry_run
python main.py -cfg configs/my_run.yaml
```

**Upper bound and centralised training.**
- `-algo FedAvgJoint` with the same flags trains, in every task, on the images of all tasks so far
  (joint training; its results carry the `UPPER_BOUND` flag).
- Centralised training is one client holding all the data: generate
  `python generate_scenario.py <Name> iid --num_clients 1 --class_per_task C --num_seeds 3` and run with
  `-partition iid -nc 1`. FedAvg over one client is ordinary training.

**The numbers in these commands are examples.** Rounds per task, local epochs, batch size, learning
rate, momentum, weight decay, LoRA rank and prompt length are protocol choices: the code defaults are
neutral (plain SGD, no weight decay, constant learning rate), not tuned recipes. Every value and where it
came from is saved in `experiment.json`.

---

## 6. Hyperparameters

### 6.1 Where they live and how to see them

| What | Defined in | See it with |
|---|---|---|
| General settings (scenario, features or model, schedule, run) | `system/utils/config_utils.py`, table `OPTIONS` | `python main.py --help`, `python main.py --list_settings` |
| The networks (names, sizes, which modes exist) | `system/utils/model_catalog.py` | `python main.py --list_settings` (under `model`) |
| An algorithm's own hyperparameters | `HPARAMS` in the algorithm's server class | `python main.py -algo NAME --help`, `python main.py --list_algorithms` |

Values come from three layers, later wins: built-in default, then the YAML file (`-cfg`), then the
command line. Where every value came from is saved in `experiment.json`. A setting that has no meaning
for the chosen algorithm or partition is refused with an error; it is never silently ignored.

### 6.2 Common FICL settings

**Scenario: which data, how many clients and tasks**

| YAML key | Flag | Default | Meaning |
|---|---|---|---|
| `dataset` | `-data` | — | dataset folder name, e.g. `ImageNetR` |
| `partition` | `-partition` | — | `dir`, `iid` or `natural` |
| `num_clients` | `-nc` | — | number of clients (`dir`, `iid`); `natural` takes it from the scenario |
| `alpha` | `-alpha` | — | Dirichlet concentration (`dir` only); smaller is more skewed |
| `class_per_task` | `-cpt` | — | classes per task; tasks = ceil(classes / cpt) |
| `use_val` | `-useval` | `ignore` | official validation split: `ignore` or `merge` |
| `min_train_per_client` | `-mintr` | 0 | as given to `generate_scenario.py` |
| `class_order` | `-corder` | `shuffled` | as given to `generate_scenario.py`: `shuffled` or `sorted` |
| `dataset_root` | `-droot` | `../dataset` | folder that holds the dataset folders |

**Features: the frozen pretrained model and what to read from it**

| YAML key | Flag | Default | Meaning |
|---|---|---|---|
| `backbone` | `-bb` | — | `dinov2_vitb14`, `clip_vitb16` or `vit_b16_in21k` (must be extracted) |
| `group` | `-fg` | — | cache group: `pre` (before the final norm) or `last_ln` (after it) |
| `blocks` | `-blocks` | all stored | blocks to read, e.g. `11` or `6,7,8,9,10,11`; several are concatenated |
| `tokens` | `-tokens` | all stored | `cls`, `mean` or `cls,mean` |
| `proj_rank` | `-prank` | — | random projection of every block to this many dimensions |
| `proj_seed` | `-pseed` | from the run seed | seed of the projection |

**Model: the network an image algorithm trains** (families 2 and 3; refused for algorithms on cached features)

| YAML key | Flag | Default | Meaning |
|---|---|---|---|
| `backbone` | `-m` | — | `tiny_cnn`, `resnet18_cifar`, `resnet32` (from scratch); `vit_b16_in21k`, `dinov2_vitb14`, `clip_vitb16` (pretrained) |
| `pretrained` | `-pre` | ViTs `true`, CNNs `false` | the ViTs are used with their weights only; the CNNs have no weights here |
| `trainable` | `-tm` | CNNs `full`; required for ViTs | `full`, `head`, `last_k`, `lora` or `prompt` (section 7.3) |
| `last_k` | `-lastk` | required with `last_k` | how many of the last blocks train (1 to 12) |
| `lora_rank` | `-lrank` | 8 | `lora` only: rank r |
| `lora_alpha` | `-lalpha` | = rank | `lora` only: the update is W + (alpha / r)·B·A |
| `lora_targets` | `-ltgt` | `q,v` | `lora` only: weights adapted in every block, from `q,k,v,proj,fc1,fc2` |
| `prompt_length` | `-plen` | 10 | `prompt` only: number of prompt tokens |
| `prompt_deep` | `-pdeep` | `false` | `prompt` only: new prompt tokens before every block (deep) or before the first only |
| `image_size` | `-isize` | CNNs 32; ViTs 224 (fixed) | input size in pixels |
| `augment` | `-aug` | CNNs `crop_flip`, ViTs `rrc_flip` | training augmentation: `none`, `flip`, `crop_flip`, `rrc_flip` (section 7.4) |

**Schedule: time inside a task**

| YAML key | Flag | Default | Meaning |
|---|---|---|---|
| `regime` | `-regime` | from the algorithm | optional: `single_round` or `multi_round`, refused if it differs |
| `rounds_per_task` | `-rpt` | — | multi-round: rounds per task (a new task every rpt rounds) |
| `local_epochs` | `-ls` | — | multi-round: local epochs per round |
| `batch_size` | `-lbs` | — | multi-round: local batch size |
| `learning_rate` | `-lr` | — | multi-round: local learning rate |
| `join_ratio` | `-jr` | 1.0 | multi-round: share of clients per round, at least one, seeded per round |
| `eval_gap` | `-eg` | 0 (off) | multi-round: test accuracy every eg rounds, for curves only |

**Run: repetitions and output**

| YAML key | Flag | Default | Meaning |
|---|---|---|---|
| `seed` | `-seed` | 0 | first seed; run i uses the scenario with seed `seed + i` |
| `times` | `-t` | 1 | number of runs; every seed's scenario must exist |
| `goal` | `-go` | `test` | label in the result folder name |
| `device`, `device_id` | `-dev`, `-did` | `cuda`, 0 | used by algorithms that run torch |
| `results_root` | `-rroot` | `../results` | root of the result folders |
| `exp_dir` | `-exp` | automatic | explicit result folder; must not exist yet |
| `num_workers` | `-nw` | 4 | image algorithms: processes that decode images; results do not depend on it |
| `eval_batch_size` | `-ebs` | 256 | image algorithms: images per evaluation forward pass |

Control flags: `-cfg FILE`, `--dry_run`, `--verify_cache` (recompute the cache sha256),
`--list_algorithms`, `--list_settings`.

### 6.3 Algorithm hyperparameters

An algorithm declares its own hyperparameters, and each becomes a flag `--NAME VALUE`:

```bash
python main.py -algo SmokeFedAvgLinear --help        # the general settings and SmokeFedAvgLinear's --weight_decay
python main.py -cfg configs/smoke_multi.yaml --weight_decay 0.001
```

In a YAML file they go under `algorithm.hparams`:

```yaml
algorithm:
  name: SmokeFedAvgLinear
  hparams:
    weight_decay: 0.001
```

| Algorithm | Regime, input | Hyperparameters |
|---|---|---|
| `SmokeClassMean` | single-round, features | none |
| `SmokeFedAvgLinear` | multi-round, features | `--weight_decay` (float, default 0.0, ≥ 0): L2 penalty in the local SGD gradient |
| `FedAvg` | multi-round, images | `--optimizer` (`sgd`, `adam`, `adamw`; default `sgd`), `--momentum` (default 0.0, 0 to 1; `sgd` only), `--weight_decay` (default 0.0; decoupled for `adamw`), `--lr_schedule` (`constant` or `cosine` over the rounds of each task; default `constant`), `--loss_classes` (`seen`: every class so far, or `task`: the current task's classes only; default `seen`) |
| `FedAvgJoint` | multi-round, images | the same, without `--loss_classes` (it always uses every class seen so far) |

The local optimizer of `FedAvg` and `FedAvgJoint` is created anew in every round, so no optimizer state
(momentum, Adam moments) carries over between rounds. `--momentum` with `adam` or `adamw` is refused.

List values are comma-separated (e.g. `--grid 0.1,1,10` for a list hyperparameter), booleans `true`
or `false`. A value outside the declared range or choices, or an unknown name, stops with an error.
`-hp NAME=VALUE` is accepted as well.

### 6.4 Checklist for a new experiment

1. **Scenario:** `dataset`, `partition`, `num_clients`, `alpha` (dir) and `class_per_task` must match a
   folder made by `generate_scenario.py`, for every seed.
2. **Features** (algorithms on cached features): `backbone`, `group`, `blocks` and `tokens` must exist in
   the cache (`python extract_features.py <Name> --list`).
   **Model** (algorithms on images): `backbone`, `trainable` and that mode's options, `augment`; the raw
   images must be in `<Name>/rawdata/`.
3. **Schedule:** nothing for a closed-form algorithm; `rounds_per_task`, `local_epochs`, `batch_size`
   and `learning_rate` for a multi-round one.
4. **Algorithm:** `name` and its hyperparameters (`python main.py -algo NAME --help`).
5. **Run:** `seed`, `times` (3 or more for results) and `goal`; for images also `device: cuda` and
   `num_workers`.
6. **Check:** `python main.py -cfg <file> --dry_run`.

---

## 7. Models

### 7.1 Pretrained backbones

All three are ViT-B models (12 transformer blocks, 768-dimensional tokens, about 86 M parameters). They
are used in two ways: frozen, to write feature caches (family 1), or trained on images (family 2).

| `--backbone` / `-m` | Model | Pretraining | Patch, tokens at 224 px | Loaded from | Preprocessing |
|---|---|---|---|---|---|
| `dinov2_vitb14` | DINOv2 ViT-B/14 | self-supervised (DINOv2) on LVD-142M, 142 M curated images | 14 px, 256 patches + class token | torch.hub `facebookresearch/dinov2` | resize 256 (bicubic), center crop 224, ImageNet mean and std |
| `clip_vitb16` | CLIP ViT-B/16, image tower | contrastive image–text pretraining (OpenAI, 400 M pairs) | 16 px, 196 patches + class token | open_clip `ViT-B-16`, `openai` weights | open_clip's own transform |
| `vit_b16_in21k` | ViT-B/16 | supervised on ImageNet-21K (AugReg) | 16 px, 196 patches + class token | timm `vit_base_patch16_224.augreg_in21k` | timm's data config for the model |

**What is extracted** (family 1). For every chosen block, the block's output tokens are pooled in two
ways:
- `cls`: the class token;
- `mean`: the mean of the patch tokens (class and register tokens left out).

Either way, the pooling is done on the raw block output (`final_ln: false`) or after the backbone's own
final norm (`final_ln: true`). The last block with the final norm is exactly what the model itself
outputs: every extracted batch checks this, and a test checks it on 10 images per backbone. For CLIP,
the features are taken before its 768 → 512 output projection. Each block gives a 768-dimensional
vector per token; with several blocks or tokens the vectors are concatenated, and `proj_rank` can
project each block to fewer dimensions with a fixed random matrix.

**What the trained networks read** (family 2). The backbone is loaded by the same code as for
extraction (`dataset/utils/backbone_utils.py`) and evaluated with the same preprocessing. Its output is
the class token of the last block after the final norm (for CLIP, before the projection): the vector a
cache stores for the last block with `final_ln: true`. A linear head over all classes of the dataset
reads it. Before training, the network therefore computes the features that extraction computes (in
fp32; the cache stores them in fp16), and a GPU test checks this for all three backbones.

### 7.2 Networks trained from scratch

| `-m` | Structure | Parameters (without the head) | Features |
|---|---|---|---|
| `resnet18_cifar` | ResNet-18 with the CIFAR stem: 3×3 conv with stride 1 and no max pool, four stages of two basic blocks (64, 128, 256, 512 channels), 1×1 projection shortcuts | 11,168,832 | 512 |
| `resnet32` | CIFAR ResNet-32 (He et al., 2016): 3×3 conv with 16 channels, three stages of five basic blocks (16, 32, 64 channels), parameter-free shortcuts (subsampling and zero channels) | 463,504 | 64 |
| `tiny_cnn` | two 3×3 conv layers with batch norm; for pipeline checks only, its runs carry the `SMOKE_TEST` flag | 5,136 | 32 |

Convolutions start from Kaiming-normal weights (fan out), batch norm from weight 1 and bias 0, as in
torchvision's ResNets. Global pooling is the mean over positions, so any input size works (`-isize`).

### 7.3 What `trainable` trains

| `trainable` | Trained | How |
|---|---|---|
| `full` | every weight and the head | CNNs from scratch; for the ViTs every weight the class token depends on (CLIP's output projection and DINOv2's mask token are never used, so they stay frozen) |
| `head` | the linear head | the backbone stays frozen in evaluation mode |
| `last_k` | the last K blocks, the final norm and the head | the other blocks stay frozen |
| `lora` | LoRA factors and the head | each chosen weight W of every block is read as W + (alpha / r)·B·A, with A of shape r × in (Kaiming-uniform start) and B of shape out × r (zero start), so training starts from the pretrained network. `q`, `k`, `v` are the row blocks of the fused input projection, each with its own pair (A, B); `proj` is the attention output, `fc1` and `fc2` the MLP. Built with `torch.nn.utils.parametrize`, which works the same for timm and DINOv2 (`nn.Linear` layers) and for CLIP (`nn.MultiheadAttention`, whose forward reads its weights directly). The pretrained W never changes |
| `prompt` | prompt tokens and the head | Visual Prompt Tuning (Jia et al., 2022): L learnable tokens (start N(0, 0.02²)) inserted after the class token before the first block; with `prompt_deep: true` they are replaced by new ones before every block. No position embedding is added to them, and the class token stays first |

### 7.4 Training and evaluation of image algorithms

- **One network on the device.** The clients of a round train one after another on one network object:
  each loads the model it received, trains, and keeps its result on the CPU until the upload. A ViT-B is
  therefore on the GPU once, not once per client.
- **Local training.** `local_epochs` epochs of minibatches of `batch_size`, with the optimizer of
  `--optimizer` created anew every round, at `local_learning_rate` (constant, or cosine over the rounds
  of the task with `--lr_schedule cosine`). The loss is the cross-entropy over the classes of
  `--loss_classes`: the logits of every other class are set to −∞, so they get no probability and no
  gradient.
- **FedAvg.** The server averages the uploaded states weighted by the clients' training samples.
  Batch-norm running statistics of trained layers are averaged with them.
- **Messages.** A download is the global state of the trained tensors (and batch-norm statistics); an
  upload adds the number of training samples. The head travels only for the classes seen so far. Bytes
  follow the same rule as everything else (4 per float, 8 per integer): LoRA of rank 8 on `q,v` of a
  ViT-B sends 294,912 floats (about 1.2 MB) plus the head per message, full fine-tuning about 340 MB.
- **Evaluation** after every task: ViTs use the backbone's own preprocessing (the one feature
  extraction uses), CNNs resize the shorter side to `image_size`, center crop and normalize with the
  ImageNet mean and std. Prediction is among the classes seen so far.
- **Augmentation** (`augment`, training only):

  | Value | What it does |
  |---|---|
  | `none` | the evaluation transform |
  | `flip` | the evaluation transform, then a horizontal flip with probability 1/2 |
  | `crop_flip` | CNNs: zero padding of `image_size / 8` pixels (4 at 32 px) and a random crop back to `image_size`; ViTs: the evaluation resize, then a random crop instead of the center crop. Then a random flip |
  | `rrc_flip` | a random resized crop of the original image (area 8 % to 100 %, aspect ratio 3/4 to 4/3, the torchvision algorithm), then a random flip |

- **Randomness.** The order of a client's batches and each image's augmentation come from generators
  seeded by (algorithm seed, client, task, round, epoch, image row), so results do not depend on
  `num_workers` or on which client ran first; initial weights, LoRA factors, prompts and the head come
  from the run seed. On the GPU, cuDNN runs in deterministic mode with TF32 off, attention uses the
  plain (math) kernel, the loss is written with operations that have deterministic GPU versions, and
  torch's deterministic algorithms are switched on (an operation without a deterministic version
  prints a warning instead of stopping). The GPU tests check that training steps repeat exactly.
- **What is not here yet:** first-session adaptation followed by frozen arms (adapt on task 0, then
  extract a new cache), the re-extraction oracle, and the PEFT reference methods (e.g. LoRM, PILoRA).
  The design is in [`reports/MODEL_MODES.md`](reports/MODEL_MODES.md).

### 7.5 Algorithms included

| Algorithm | Family | Regime | What it does |
|---|---|---|---|
| `SmokeClassMean` | 1 | single-round | Each client uploads per-class feature sums and counts once per task; the server keeps the totals and predicts the nearest class mean. Because sums add up, the federated model equals the centralized one for any client split. Flag `SMOKE_TEST`. |
| `FedCGS` | 1 | single-round | FedCGS (Guan et al., AAAI 2025). Clients upload class sums, class counts and the uncentred second moment of their current task's features; the server adds them over clients and tasks. It then predicts with a Gaussian discriminant: class means, the total covariance around the global mean and class priors. One hyperparameter, `gamma` (default 0, the official value). After every task it equals FedCGS fitted centrally on all data seen so far. Flag `REFERENCE`. Details, differences from the official code and the official-code check: [reports/FEDCGS.md](reports/FEDCGS.md). |
| `SmokeFedAvgLinear` | 1 | multi-round | FedAvg on a linear softmax head over the cached features. Every round the selected clients run local SGD on their current-task data, and the server averages the heads weighted by sample counts. There is no replay, so earlier classes are forgotten, as expected. Flag `SMOKE_TEST`. |
| `FedAvg` | 2, 3 | multi-round | FedAvg of the network of the `model` section on the current task's images only: sequential fine-tuning, the lower reference of class-incremental learning (earlier classes are forgotten). Flag `REFERENCE`. |
| `FedAvgJoint` | 2, 3 | multi-round | The same on the images of every task so far: joint training, an upper bound. Flag `UPPER_BOUND`. |

Section 9 shows how to add a method.

---

## 8. Results

Each experiment gets a new folder, and each run (seed) a subfolder:

```
results/<dataset>/<scenario without seed>/<algorithm>[_FLAGS]_<backbone>-<group>_<goal>_<timestamp>/
                       (image algorithms: <algorithm>[_FLAGS]_<network>-<trainable>_<goal>_<timestamp>/)
    experiment.json        every setting and where it came from, the protocol text, versions, git commit
    times_<i>/             run i (scenario seed seed + i)
        run.json           seeds, scenario (name and sha256 values), features (cache sha256) or model
                           (settings, the network built, trainable and total parameters, the
                           pretrained weights' sha256, the transforms), schedule
        tasks.csv          per task: accuracy on all seen classes and on the new task, client mean and
                           minimum, bytes up and down, seconds (multi-round also: rounds)
        acc_global.csv     accuracy matrix: row = after task t, column = test set of task i (i <= t)
        acc_clients.csv    per task and client: accuracy on the client's own test shards
        clients/client_<k>.csv   the rows of one client
        comm.csv           every message: task, (round,) client, direction, bytes, keys
        rounds.csv         multi-round only: per round, the selected clients, train loss, curve accuracy
        final.json         the final values of the run; written last, so a run is complete when it exists
        <dataset>_<algorithm>_<goal>_run<i>.log   the run's log (with the traceback if it failed)
    avg/summary.json       mean and SD (ddof = 1) over all complete runs, and the status of every run
    avg/tasks.csv          accuracy on seen classes per task, mean and SD over runs
```

| Final value | Meaning |
|---|---|
| `acc_final` | accuracy on all test images after the last task |
| `acc_avg_incremental` | mean over tasks of the accuracy on the classes seen so far |
| `client_acc_final_mean`, `_min` | mean and lowest client accuracy on their own test shards after the last task |
| `bytes_up_total`, `bytes_down_total` | bytes sent by the clients and by the server |
| `seconds_total` | wall-clock time of the run |

Bytes count 4 per float (fp32 on the wire), 8 per integer, 1 per boolean, and only the upper triangle
of a symmetric matrix. Downloads include the model sent to the clients at the end of every task. For
image algorithms a message holds the trained tensors only (section 7.4). A
failed run has no `final.json`: its log holds the error, `avg/summary.json` marks it failed, and the
remaining runs are not started. After every experiment, `check_run.py` checks the files. Among other
things, it verifies that nothing was evaluated before its task arrived, that the accuracies are
whole-number counts over the scenario's test sets, and that no file reports a best round. You can
also run it yourself: `python check_run.py <experiment folder>`.

---

## 9. Writing a new algorithm

An algorithm is a server class and a client class. The shared loop calls their hooks; an algorithm
never writes its own loop.

```python
# system/flcore/servers/servermyalgo.py
import numpy as np
from flcore.clients.clientmyalgo import clientMyAlgo
from flcore.servers.serverbase import Server
from utils.config_utils import SINGLE_ROUND, HParam


class MyAlgo(Server):
    """MyAlgo: one line that --list_algorithms shows."""
    REGIME = SINGLE_ROUND                       # or MULTI_ROUND
    CLIENT_CLASS = clientMyAlgo
    HPARAMS = {                                 # each becomes --NAME on the command line
        "lam": HParam("float", 1.0, "ridge penalty", low=0.0),
        "solver": HParam("str", "cholesky", "how to solve the system", choices=("cholesky", "eigh")),
    }

    def init_algorithm(self):                   # self.store.dim, self.num_classes, self.hp are known
        self.lam = self.hp["lam"]

    def aggregate(self, t, r, uploads):         # required: uploads = [(client_id, payload), ...]
        ...

    def final_broadcast_payload(self, t):       # optional: the model sent to every client after task t
        return None

    def predict(self, Z, seen):                 # required: class ids among `seen` for features Z [n, d]
        ...


# system/flcore/clients/clientmyalgo.py
from flcore.clients.clientbase import Client
from utils.comm_utils import Symmetric


class clientMyAlgo(Client):
    def train_round(self, t, r):                # required: one statistics pass, or local SGD epochs
        Z, y = self.load_task_data()            # the current task's training rows: float64 [n, d], int64 [n]
        self._payload = {"G": Symmetric(Z.T @ Z), "n": len(y)}

    def upload_payload(self, t, r):
        return self._payload
```

Then register it in `system/main.py`:

```python
ALGORITHMS = {
    ...,
    "MyAlgo": "flcore.servers.servermyalgo:MyAlgo",
}
```

**Hyperparameters:** `HParam(kind, default, help, choices, low, high)`.
- `kind` is `int`, `float`, `str`, `bool`, `int_list`, `float_list` or `str_list`.
- The default must satisfy the choices and range.
- Names are lower case and must not reuse a general setting or flag (e.g. `seed`, `lr`).
- Read them as `self.hp["lam"]`, on the server and on every client.
- `python main.py -algo MyAlgo --help` shows them with their help text, type, default, choices and range.

**An algorithm that trains a network on images** subclasses `ImageServer` and `ImageClient`, which
already build the network from the `model` section, run the local epochs, average the states (FedAvg)
and evaluate. It changes only what differs:

```python
# system/flcore/servers/servermyft.py
from flcore.clients.clientimage import ImageClient
from flcore.servers.serverimage import TRAIN_HPARAMS, ImageServer
from utils.config_utils import HParam


class clientMyFT(ImageClient):
    def loss(self, logits, y, x):               # logits [b, classes], -inf outside the loss classes
        return super().loss(logits, y, x) + self.hp["beta"] * my_extra_term(self.bundle.net, x)


class MyFT(ImageServer):
    """MyFT: one line that --list_algorithms shows."""
    CLIENT_CLASS = clientMyFT
    HPARAMS = {**TRAIN_HPARAMS, "beta": HParam("float", 0.1, "weight of the extra term", low=0.0)}
    # aggregate(), on_end_task(), download_payload() can be overridden as well
```

Other client hooks for images: `training_rows()` (which rows to train on) and `loss_classes()`.

**Other hooks:**
- `on_begin_task(t)` and `on_end_task(t)` on both sides; the server may call `self.send(client, t, payload)`.
- `download_payload(t, r, client)` and `client.receive(t, r, payload)`.
- `client.round_report()`: the train loss for `rounds.csv`.
- `Client.PERSONALIZED = True` with `client.predict` for personalized models.

**What the base classes do for you:**
- task gating of client data and client selection;
- seeded random generators (`self.rng(...)`, `client.rng(t, r)`);
- byte counting of every message (wrap a symmetric matrix in `Symmetric`);
- evaluation, result files, logs and checks.

**What they refuse:**
- a class that overrides the loop;
- a `predict` that returns classes not seen yet;
- reading test features or test images outside evaluation (`TestDataAccessError`);
- unknown run flags and badly named hyperparameters.

---

## 10. Tests

From the repository root:

```bash
(cd dataset && python -m pytest tests -q -m "not gpu and not data")
(cd system && python -m pytest tests -q -m "not gpu")
(cd dataset && python -m pytest tests -q -m gpu)     # hooked features vs the models' own outputs (GPU, downloads weights)
(cd system && python -m pytest tests -q -m gpu)      # trained encoders vs the cached features, gradients, repeats on the GPU
(cd dataset && python -m pytest tests -q -m data)    # MedMNIST counts vs the package metadata (needs the data)
```

The two sides run separately, because `dataset/utils` would shadow `system/utils` if both folders were
on Python's path. `./run_tests.sh` runs both CPU sides in one go and first prints the Python, NumPy,
torch, torchvision, timm and open_clip versions. Expected: dataset 64 passed and 8 deselected; system
113 passed and 19 deselected. The GPU tests download the pretrained weights on first use.

The tests cover:
- **Scenarios:** every class in one task; disjoint client shards that cover the data; test shards equal
  to the official test set; byte-identical files for the same seed.
- **Benchmark layouts:** every supported layout, read from small fake copies.
- **Extraction:** pooling and the reference check.
- **Loop:** hook order in both regimes; clients see only the current task; no test data while training.
- **Image path:** the model settings and their refusals; the image store reads the index's rows and
  refuses test images outside evaluation; the CNN sizes; the transforms; LoRA and prompts on small timm
  and open_clip ViTs (unchanged output at the start, only the chosen rows move); FedAvg end to end on
  the synthetic images (learns, passes every check, sends exactly the trained tensors); identical results
  with and without loader workers; the joint upper bound trains on every task and forgets less.
- **Exactness:** federated class sums equal centralized ones.
- **Results:** no round concept in single-round files; the curve points; no best-round field.
- **Settings:** the settings and hyperparameter rules.
- **Entry point:** `main.py` end to end.

### 10.1 Checking the code on real datasets

The tests run on synthetic data. Two further checks run on the real datasets; neither produces results
for the experiment (`-go validate` marks the folders).

**Step A: every dataset's data path** (downloads, index, published counts, scenarios, the raw images
against the index). No training. Run from `dataset/`:

```bash
cd dataset
declare -A CPT=([CIFAR100]=10 [TinyImageNet]=20 [ImageNetR]=20 [ImageNetA]=20 [CUB200]=20 [EuroSAT]=2 \
                [RESISC45]=5 [TissueMNIST]=2 [OCTMNIST]=2 [OrganCMNIST]=2 [DermaMNIST]=2 [PathMNIST]=2 [BloodMNIST]=2)
for d in "${!CPT[@]}"; do
  echo "######## $d"
  python prepare_data.py "$d" --download || continue          # RESISC45 is a manual download
  python generate_scenario.py "$d" dir --num_clients 10 --alpha 1.0 --class_per_task "${CPT[$d]}" --seed 0 --num_seeds 3
  (cd ../system && python main.py -data "$d" -partition dir -nc 10 -alpha 1.0 -cpt "${CPT[$d]}" -t 3 \
       -algo FedAvg -m resnet32 -rpt 1 -ls 1 -lbs 32 -lr 0.01 --dry_run | tail -3)
done
python -m pytest tests -q -m data                               # MedMNIST counts vs the package metadata
```

A dataset passes when `prepare_data.py` prints no `WARNING` about the published totals and the dry run
ends with `settings and data are consistent for all 3 seed(s)`. DomainNet (large) works the same way,
with `generate_scenario.py DomainNet natural --class_per_task 35` and `-partition natural`.

**Step B: short runs per family, on two datasets each.** Each run must end with `checks: N passed,
0 failed`. Run from `system/`:

```bash
cd system
# families 1 and 2 on EuroSAT (remote sensing) and DermaMNIST (medical), with DINOv2:
# frozen features (nearest class mean), then the same backbone trained on the images (head, LoRA)
for d in EuroSAT DermaMNIST; do
  (cd ../dataset && python extract_features.py $d --backbone dinov2_vitb14)
  python main.py -data $d -partition dir -nc 10 -alpha 1.0 -cpt 2 -bb dinov2_vitb14 -fg last_ln -tokens cls \
      -algo SmokeClassMean -t 3 -go validate
  for tm in head lora; do
    python main.py -data $d -partition dir -nc 10 -alpha 1.0 -cpt 2 -algo FedAvg -m dinov2_vitb14 -tm $tm \
        -aug none -rpt 3 -ls 1 -lbs 32 -lr 0.01 --momentum 0.9 -t 1 -go validate
  done
done
# family 3 on CIFAR100 (32 px) and TinyImageNet (64 px), ResNet-32 from scratch, 5 tasks each,
# sequential (FedAvg) and joint (FedAvgJoint)
(cd ../dataset && python generate_scenario.py CIFAR100 dir --num_clients 10 --alpha 1.0 --class_per_task 20 --seed 0 \
               && python generate_scenario.py TinyImageNet dir --num_clients 10 --alpha 1.0 --class_per_task 40 --seed 0)
for spec in "CIFAR100 20 32" "TinyImageNet 40 64"; do
  read -r d cpt size <<< "$spec"
  for algo in FedAvg FedAvgJoint; do
    python main.py -data $d -partition dir -nc 10 -alpha 1.0 -cpt $cpt -algo $algo -m resnet32 -isize $size \
        -rpt 10 -ls 1 -lbs 64 -lr 0.05 --momentum 0.9 -t 1 -go validate
  done
done
```

What to look for, beyond the checks: the accuracy on each new task (`acc_task_new`, one line per task)
well above chance (1 / classes per task); `FedAvgJoint`'s final accuracy above `FedAvg`'s, which forgets
the earlier tasks; and, on each of EuroSAT and DermaMNIST, the frozen nearest class mean and the head
trained on the same frozen DINOv2 features in the same range. These are sanity checks of the pipeline with short schedules, not
reference numbers.

### 10.2 Checking against a published FCIL benchmark

This check reproduces the fed-mammoth protocol of LoRM (ICLR 2025) and HGP (TMLR 2025): CIFAR-100 in
10 tasks, 10 clients, Dirichlet β, timm `vit_base_patch16_224.augreg_in21k` trained in full with AdamW,
5 rounds × 5 epochs × batch 16. It runs FedAvg fine-tuning and the Joint bound three ways:

1. our code (`system/configs/reference/*.yaml`);
2. fed-mammoth's own code, re-run with the same flags;
3. the published numbers [R].

`compare_reference.py` puts the three side by side and applies the gap rule (max of 2 points and
2 × SD). The full protocol, the known differences, the commands and how to read the outcome are in
[reports/REFERENCE_CHECK.md](reports/REFERENCE_CHECK.md). The published numbers live in
`reports/reference_numbers.json` and never go into our result tables.

```bash
cd dataset
python generate_scenario.py CIFAR100 dir --num_clients 10 --alpha 0.05 --class_per_task 10 \
    --class_order sorted --min_train_per_client 6 --seed 0 --num_seeds 3
cd ../system
python main.py -cfg configs/reference/ref_cifar100_fedavg_b0.05.yaml
python compare_reference.py configs/reference/ref_cifar100_fedavg_b0.05.yaml <EXPERIMENT_DIR> \
    --mammoth <fed-mammoth logs of the 3 seeds>
```

---

## 11. Repository layout

```
.
├── README.md, LICENSE, env_cuda.yaml, run_tests.sh
├── docs/TROUBLESHOOTING.md
├── reports/                      research protocol, design notes and the changelog
├── dataset/                      data preparation (run from this folder)
│   ├── prepare_data.py           step 1: sample index (and downloads)
│   ├── extract_features.py       step 2: frozen-backbone features -> fp16 cache (GPU)
│   ├── generate_scenario.py      step 3: class-incremental client splits, one folder per seed
│   ├── make_synthetic.py         the synthetic dataset for smoke tests
│   ├── configs/extract_default.yaml
│   ├── utils/                    raw_utils (index, dataset adapters), scenario_utils, cache_utils,
│   │                             backbone_utils (pretrained backbones), synthetic_utils
│   ├── tests/
│   └── <Name>/                   created per dataset: rawdata/, cl_index/, features/, cl_<scenario>/
└── system/                       experiments (run from this folder)
    ├── main.py                   entry point
    ├── check_run.py              checks a result folder (runs after every experiment)
    ├── configs/                  smoke_*.yaml, template_single.yaml, template_multi.yaml,
    │                             template_multi_pretrained.yaml, template_multi_scratch.yaml
    ├── flcore/servers/           serverbase.py (the shared loop), serverimage.py (base of image
    │                             algorithms), one server per algorithm
    ├── flcore/clients/           clientbase.py, clientimage.py, one client per algorithm
    ├── flcore/models/            networks of the image path: cnn.py, vit.py (pretrained ViTs, LoRA,
    │                             prompts), transforms.py, network.py (head, trainable parts, messages)
    ├── utils/                    config_utils (settings), model_catalog (the networks), data_utils,
    │                             image_utils (ImageStore), dataset_code, result_utils, comm_utils, log_utils
    ├── run.sh, smoke.sh          optional shortcuts
    └── tests/
```

Results go to `results/` at the repository root. Data, caches, scenarios, results and logs are never
committed.

---

## 12. Acknowledgements and license

The structure and workflow follow HtFLlib and PFLlib:

```
@inproceedings{Zhang2025htfllib,
  author={Zhang, Jianqing and Wu, Xinghao and Zhou, Yanbing and Sun, Xiaoting and Cai, Qiqi and Liu, Yang and Hua, Yang and Zheng, Zhenzhe and Cao, Jian and Yang, Qiang},
  title = {HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark},
  year = {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining}
}
@article{zhang2025pfllib,
  title={PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={Journal of Machine Learning Research},
  volume={26}, number={50}, pages={1--10}, year={2025}
}
```

The code is under Apache-2.0 (`LICENSE`, inherited from HtFLlib). Datasets and pretrained weights keep
their own licences; check them before any use beyond research. The research protocol and the history
of changes are in [`reports/`](reports/).
