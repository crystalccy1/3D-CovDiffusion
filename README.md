# 3D-CovDiffusion: 3D-Aware Diffusion Policy for Coverage Path Planning

Official implementation of **3D-CovDiffusion**, a geometry-conditioned
diffusion policy for ordered 6-DoF coverage trajectories from 3D point clouds.

**Accepted at IROS 2026.**

Chenyuan Chen · Haoran Ding · Ran Ding · Tianyu Liu · Zewen He · Anqing Duan<sup>*</sup> · Yoshihiko Nakamura

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb?style=for-the-badge&logo=githubpages&logoColor=white)](https://crystalccy1.github.io/3D-CovDiffusion/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.03011-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.03011)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://crystalccy1.github.io/3D-CovDiffusion/3d-covdiffusion-paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-Train--Ready-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ChenyuanC/3D-CovDiffusion-Train-Ready)
[![Models](https://img.shields.io/badge/Models-Pretrained-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/ChenyuanC/3D-CovDiffusion)

![3D-CovDiffusion overview](https://crystalccy1.github.io/3D-CovDiffusion/media/teaser.jpg)

## Table of Contents

- [To-Do List](#to-do-list)
- [Getting Started](#-getting-started)
  - [Reproduction entry point](#reproduction-entry-point)
- [Inference with Released Checkpoints](#-inference-with-released-checkpoints)
  - [Prepare the data](#1-prepare-the-data)
  - [Download the checkpoints](#2-download-the-checkpoints)
  - [Run inference](#3-run-inference)
- [Train from Scratch](#-train-from-scratch)
- [Categories and Checkpoints](#categories-and-checkpoints)
- [Outputs](#outputs)
- [Repository Structure](#repository-structure)
- [Fixed Protocol](#fixed-protocol)
- [Citation](#citation)
- [Acknowledgements and License](#acknowledgements-and-license)

## To-Do List

- [x] Release the train-ready `v2` datasets for all four categories.
- [x] Release the pretrained EMA checkpoints for all four categories.
- [x] Release the inference and evaluation pipeline.
- [x] Release the training pipeline.
- [x] Publish the project page and supplementary videos.
- [x] Replace the legacy loader, configuration, and metric wrappers with
      release-owned implementations.
- [x] Publish the audited source tree with clean Git history.

## 🚀 Getting Started

The validated platform is Linux x86_64 with Python 3.10.18, PyTorch
1.13.1+cu117, CUDA 11.7, and an NVIDIA GPU. Install Git, Conda, `curl`, `unzip`,
and `md5sum` first.

```bash
git clone https://github.com/crystalccy1/3D-CovDiffusion.git
cd 3D-CovDiffusion

conda create -n 3dcov python=3.10.18 pip -y
conda activate 3dcov
python -m pip install -r requirements.txt

python reproduce.py doctor
```

This is the recommended setup for normal training and inference. `doctor`
checks the core runtime imports and an available CUDA device.

For an exact hash-locked environment, use the optional strict installer:

```bash
bash scripts/create_locked_environment.sh 3dcov-cu117
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3dcov-cu117
export PYTHONNOUSERSITE=1
python reproduce.py doctor
```

### Reproduction entry point

`reproduce.py` is the recommended reproducibility entry point. It validates
the requested artifacts and launches the underlying training or evaluation
implementation with the locked category profile:

```text
reproduce.py     unified entry point and reproduction workflow
├── train.py     optimization and checkpointing
├── evaluate.py  rollout, inference, and metrics
├── configs/     model and category configurations
└── scripts/     data/checkpoint download and validation
```

The model and training logic remain in the underlying modules; the wrapper
keeps their data paths, configs, seeds, batches, and checkpoint variants
consistent across machines.

## 🤖 Inference with Released Checkpoints

After installing the environment, prepare the category data, download its
released best EMA checkpoint, and run inference.

The commands below download pinned artifacts from these public Hugging Face
repositories:

- **Train-ready dataset:**
  [ChenyuanC/3D-CovDiffusion-Train-Ready](https://huggingface.co/datasets/ChenyuanC/3D-CovDiffusion-Train-Ready)
- **Pretrained checkpoints:**
  [ChenyuanC/3D-CovDiffusion](https://huggingface.co/ChenyuanC/3D-CovDiffusion)

### 1. Prepare the data

This downloads and validates the train-ready v2 dataset and self-contained
evaluation-ready inputs for the selected category:

```bash
# Windows
python reproduce.py prepare windows --data-only

# Cuboids
python reproduce.py prepare cuboids --data-only

# Shelves
python reproduce.py prepare shelves --data-only

# Containers
python reproduce.py prepare containers --data-only
```

### 2. Download the checkpoints

Each command downloads and validates the released best EMA checkpoint for one
category:

```bash
# Windows
python reproduce.py prepare windows --checkpoint-only

# Cuboids
python reproduce.py prepare cuboids --checkpoint-only

# Shelves
python reproduce.py prepare shelves --checkpoint-only

# Containers
python reproduce.py prepare containers --checkpoint-only
```

### 3. Run inference

```bash
# Windows
python reproduce.py evaluate windows

# Cuboids
python reproduce.py evaluate cuboids

# Shelves
python reproduce.py evaluate shelves

# Containers
python reproduce.py evaluate containers
```

## 💪 Train from Scratch

Each category is trained independently from its train-ready `v2` dataset. Before
launching a run, download and validate that category once with
`python reproduce.py prepare <category> --data-only`, where `<category>` is
`windows`, `cuboids`, `shelves`, or `containers`.

The commands below reproduce the full seed-42 training protocol. `train.py`
handles optimization, EMA updates, rollout evaluation, and checkpointing. Run
them from the repository root and select the GPU with `CUDA_VISIBLE_DEVICES`.
Each run writes its resolved configuration and checkpoints to
`artifacts/runs/<category>/seed42`; `checkpoints/latest.ckpt` can be used to
resume training, while `checkpoints/best.json` records the selected best EMA
checkpoint.

> **Training time:** each command runs 4,800 epochs and is intended as a full
> reproduction experiment rather than a quick execution check.

### Windows

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  'config=[covdiffusion,windows]' \
  seed=42 \
  batch_size=512 \
  workers=0 \
  training.replay_epoch_sampling=true \
  checkpoint.save_ckpt=true \
  processed_data_root="${PWD}/artifacts/dataset" \
  evaluation_cache_root="${PWD}/artifacts/dataset/evaluation-cache" \
  epochs=4800 \
  output_dir="${PWD}/artifacts/runs/windows/seed42"
```

### Cuboids

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  'config=[covdiffusion,cuboids]' \
  seed=42 \
  batch_size=512 \
  workers=0 \
  training.replay_epoch_sampling=true \
  checkpoint.save_ckpt=true \
  processed_data_root="${PWD}/artifacts/dataset" \
  evaluation_cache_root="${PWD}/artifacts/dataset/evaluation-cache" \
  epochs=4800 \
  output_dir="${PWD}/artifacts/runs/cuboids/seed42"
```

### Shelves

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  'config=[covdiffusion,shelves]' \
  seed=42 \
  batch_size=512 \
  workers=0 \
  training.replay_epoch_sampling=true \
  checkpoint.save_ckpt=true \
  processed_data_root="${PWD}/artifacts/dataset" \
  evaluation_cache_root="${PWD}/artifacts/dataset/evaluation-cache" \
  epochs=4800 \
  output_dir="${PWD}/artifacts/runs/shelves/seed42"
```

### Containers

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  'config=[covdiffusion,containers]' \
  seed=42 \
  batch_size=128 \
  workers=0 \
  training.replay_epoch_sampling=true \
  checkpoint.save_ckpt=true \
  processed_data_root="${PWD}/artifacts/dataset" \
  evaluation_cache_root="${PWD}/artifacts/dataset/evaluation-cache" \
  epochs=4800 \
  output_dir="${PWD}/artifacts/runs/containers/seed42"
```

## Categories and Checkpoints

| Category | Dataset / config | Batch | Released best EMA checkpoint | Local training run |
|:--|:--|--:|:--|:--|
| Windows | `windows-v2` / `windows` | 512 | [Download](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/windows/model.safetensors?download=true) | `artifacts/runs/windows/seed42` |
| Cuboids | `cuboids-v2` / `cuboids` | 512 | [Download](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/cuboids/model.safetensors?download=true) | `artifacts/runs/cuboids/seed42` |
| Shelves | `shelves-v2` / `shelves` | 512 | [Download](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/shelves/model.safetensors?download=true) | `artifacts/runs/shelves/seed42` |
| Containers | `containers-v2` / `containers` | 128 | [Download](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/containers/model.safetensors?download=true) | `artifacts/runs/containers/seed42` |

The model links are pinned to Hugging Face revision `a2d93ff`. Exact saved
epoch/global-step metadata is in
[`docs/TRAINING.md`](docs/TRAINING.md#released-checkpoint-metadata).

Checkpoint roles are intentionally separate:

- `artifacts/models/<category>/model.safetensors`: released best EMA weights
  for normal fixed-test inference with `evaluate`;
- `artifacts/models/raw/<category>/model.safetensors`: released non-EMA weights
  for the locked selected-case replay with `infer`;
- `latest.ckpt`: model, EMA, optimizer, RNG, and backend state for resuming a
  local training run;
- `checkpoints/best.json`: exact training-to-evaluation handoff for the best
  local EMA checkpoint.

Training `.ckpt` files use Python serialization and must only be loaded from a
trusted source. The released inference weights use the safer `safetensors`
format.

## Outputs

```text
artifacts/
├── dataset/data/<category>-v2/train.zarr/
├── dataset/evaluation-cache/<category>-v2/*.npz
├── models/<category>/                  # released EMA safetensors
├── models/raw/<category>/              # selected-case non-EMA safetensors
└── runs/<category>/seed42/
    ├── config.yaml
    └── checkpoints/
        ├── latest.ckpt
        ├── best.json
        └── epoch=...ckpt

outputs/
├── evaluation/<category>-s42/test_results.json
└── selected/selected_<category>-v2_s42_ep<N>/test_results.json
```

Expected terminal signals are `Environment check passed`, a preparation message
ending in `validated: <category>`, `TRAIN_STEP_OK`, `Training complete`, and
`Evaluation complete`. Locked `infer` additionally writes `"verified": true`.

## Repository Structure

```text
reproduce.py                    # recommended reproducibility CLI
train.py / evaluate.py          # training and rollout execution
configs/                        # four fixed category profiles and manifests
covdiffusion/ / models/ / utils/ # model, data loaders, metrics, checkpointing
scripts/                        # environment, artifact download, and validation
requirements/                   # validated runtime and visualization locks
tests/                          # data, model, metric, CLI, and resume checks
docs/                           # detailed protocol and provenance
```

The current source tree excludes project-page media and private
artifact-publication utilities. The independently hosted
[project page](https://crystalccy1.github.io/3D-CovDiffusion/) remains the
canonical website.

## Fixed Protocol

- Input: a 5,120-point XYZ cloud and the previous 24-D token, representing four
  ordered 6-DoF poses.
- Training: seed 42, 4,800 epochs, Adam at `1e-4`, EMA, and fixed rollout every
  five epochs.
- Diffusion: DDIM, 100 training timesteps, `prediction_type: sample`, and 10
  sampling steps.
- Data: processed training tensors and evaluation-ready test inputs from
  [Hugging Face](https://huggingface.co/datasets/ChenyuanC/3D-CovDiffusion-Train-Ready),
  pinned to revision `38bf84c`. The evaluation-ready v2 release embeds the
  model input, trajectory, ground truth, stroke IDs, and mesh geometry for all
  618 fixed-test samples.

Preparing all categories downloads the train-ready tensors, evaluation-ready
records, and released checkpoints without a separate raw-data extraction step.
A training run needs up to about **12.25 GB** of temporary checkpoint space.
See the [data](docs/DATA.md), [training](docs/TRAINING.md),
[evaluation/inference](docs/INFERENCE.md), and
[reproducibility](docs/REPRODUCIBILITY.md) guides for details.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{chen2026_3dcovdiffusion,
  title  = {{3D-CovDiffusion}: 3D-Aware Diffusion Policy for Coverage Path Planning},
  author = {Chen, Chenyuan and Ding, Haoran and Ding, Ran and Liu, Tianyu
            and He, Zewen and Duan, Anqing and Nakamura, Yoshihiko},
  year   = {2026},
  eprint = {2510.03011},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url    = {https://arxiv.org/abs/2510.03011},
  note   = {Accepted at IROS 2026}
}
```

## Acknowledgements and License

Built on [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy),
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy). See the
[third-party notices](THIRD_PARTY_NOTICES.md) and
[file-level provenance record](docs/CODE_PROVENANCE.md).

Original source-code portions are available under the scoped
[MIT license](LICENSE). Released model weights use
[CC BY 4.0](MODEL_LICENSE.md), and the train-ready dataset uses CC BY 4.0.
Third-party components remain subject to their original terms.
