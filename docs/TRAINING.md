# Training

The public training path reproduces the checkpoint-producing seed-42 v2
protocol. Its machine-readable profile is
[`configs/training/seed42_v2.json`](../configs/training/seed42_v2.json).

## 1. Prepare once

Create the validated Python 3.10.18 / CUDA 11.7 environment, then:

```bash
bash scripts/create_locked_environment.sh 3dcov-cu117
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3dcov-cu117
export PYTHONNOUSERSITE=1
python reproduce.py prepare all
```

The creation script installs the checked-in SHA-256 lock and refuses to alter
an existing environment. `requirements.txt` remains only a convenience path
for an already compatible environment.

`prepare` places every required input below one `--artifact-root` (default
`artifacts/`): processed training Zarr, all self-contained evaluation-ready v2
records, and released tensor-only models. The evaluation records embed the
ground truth and mesh geometry required by rollout metrics.

Training has one data path:

- optimization reads `dataset/data/<category>-v2/train.zarr`;
- rollout/top-k selection reads
  `dataset/evaluation-cache/<category>-v2/*.npz` under the same artifact root.

The public CLI does not expose separate training-data or evaluation-data roots.
For training subprocesses it defaults
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`, which prevents reserved-memory
fragmentation at the formal Windows batch size without changing the model,
loss, batch, or optimizer.

## 2. Train and evaluate

```bash
python reproduce.py train windows
python reproduce.py evaluate windows \
  --run-dir artifacts/runs/windows/seed42
```

Use `cuboids`, `shelves`, or `containers` for one other category. `train all`
runs all four sequentially. A newly trained run is always evaluated one
category at a time because `--run-dir` identifies one `best.json`.

Each category has a stable run directory:

```text
artifacts/runs/<category>/seed42/
├── config.yaml
└── checkpoints/
    ├── latest.ckpt
    ├── best.json
    └── ...
```

The command refuses to overwrite a non-empty run. Resume one category with:

```bash
python reproduce.py train windows \
  --resume-from artifacts/runs/windows/seed42
```

`latest.ckpt` preserves the policy, EMA policy, optimizer, epoch/global step,
Python/NumPy/Torch RNG state, and Torch backend switches that affect CUDA
numerics (including the cuDNN deterministic/benchmark and TF32 settings).
Restoring both RNG and backend state is necessary to make a checkpoint boundary
equivalent to the uninterrupted locked-stack run. `best.json` records the
full-precision historical selection score, checkpoint filename, epoch, and
global step. It is the only training-to-evaluation handoff; `latest.ckpt` is
for resume.

Resume also compares the checkpoint's saved training configuration with the
current command before loading any model or optimizer state. Relocating the run,
processed-data, or evaluation-cache paths is allowed; changing the category,
seed, batch size, model/diffusion profile, or training protocol is rejected.

During a formal run, `latest.ckpt` is replaced on the five-epoch rollout and
checkpoint cadence, and once more after normal completion. Resume therefore
continues from the most recent saved boundary; a crash can discard up to five
epochs since that boundary. A measured checkpoint is 4,084,193,075 bytes
(about 4.084 GB), so `latest` plus the top-1 `best` occupies about 8.17 GB.
Atomic temporary-file replacement can briefly require about 12.25 GB. Reserve
that headroom before a long run.

For a bounded execution check:

```bash
python reproduce.py train windows --smoke
```

This deliberately forces batch size 2, one epoch capped at one optimizer/EMA
step, one fixed-test rollout episode, no checkpoint writes, and no archived
end-of-epoch replay sampling. It tests code/data/rollout execution, not
convergence or the GPU memory required by the formal category batch size.

### Released checkpoint metadata

The README keeps the release table focused on direct downloads. For
traceability, the released seed-42 EMA checkpoints were saved at the following
training boundaries. Epochs are zero-based, and every link is pinned to model
revision `a2d93ff`.

| Category | Saved epoch | Global step | EMA checkpoint |
|:--|--:|--:|:--|
| Windows | 165 | 48,472 | [model.safetensors](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/windows/model.safetensors?download=true) |
| Cuboids | 40 | 38,007 | [model.safetensors](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/cuboids/model.safetensors?download=true) |
| Shelves | 45 | 40,204 | [model.safetensors](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/shelves/model.safetensors?download=true) |
| Containers | 500 | 141,783 | [model.safetensors](https://huggingface.co/ChenyuanC/3D-CovDiffusion/resolve/a2d93ff2dfef9217882904a2d6ef7260a23f91b7/containers/model.safetensors?download=true) |

## 3. Fixed profile

The merged configuration is:

```text
default → covdiffusion → <category>
```

All categories use:

- seed 42, 4,800 epochs, zero data-loader workers, and Adam at `1e-4`;
- a 5,120-point XYZ observation and one previous 24-D action token;
- a horizon of 16 action tokens, where every token concatenates four ordered
  6-DoF poses;
- the effective 128-D point encoder used by the released checkpoints (the
  archived config field says 256, but the historical constructor fixed 128);
- DDIM with 100 training timesteps, `prediction_type: sample`, and 10 sampling
  steps;
- EMA after every optimizer step and deterministic end-of-epoch sampling to
  retain the archived RNG order;
- EMA rollout every five epochs in `GT_Cond → Pred_Cond` order.

| Category | Config | Batch | Training sequences | Fixed-test samples |
|:--|:--|--:|--:|--:|
| Windows | `windows` | 512 | 149,436 | 200 |
| Cuboids | `cuboids` | 512 | 474,169 | 200 |
| Shelves | `shelves` | 512 | 447,439 | 200 |
| Containers | `containers` | 128 | 36,125 | 18 |

These batch sizes come from the machine-readable public wrapper profile:
`python reproduce.py train <category>` injects 512 for Windows, Cuboids, and
Shelves, and 128 for Containers. Prefer that wrapper. An old direct command
such as `python train.py config=[...] seed=42 overfit=false` does not establish
the public profile by itself; it must explicitly add `batch_size=512` for the
three main categories or `batch_size=128` for Containers, along with the
prepared data/evaluation roots.

### Historical selection score vs final PCD

Top-k selection minimizes the prediction-conditioned historical **6-D pose
Chamfer**: symmetric squared nearest-neighbor distance over every generated and
target waypoint, using XYZ plus orientation normal scaled by 0.25, then
multiplied by `1e4`.

This is not the paper's final PCD. Final evaluation uses symmetric nearest
squared distance over XYZ only. The code and manifests keep the names separate
to prevent the historical selector from being reported as test PCD.

The CLI intentionally omits alternate encoders, partial observations,
multi-dataset training, and trajectory-loss ablations. They are not required to
reproduce the released model path.

## 4. Validation and claim boundary

### Paper-reported protocol vs released-checkpoint protocol

The manuscript and the current public artifacts describe two distinct protocol
records. Keep them separate when reproducing the code or citing model details:

| Item | Paper / arXiv v2 | Current public release |
|:--|:--|:--|
| Model scope | Reports one jointly trained model across benchmark categories | Provides four separately trained category checkpoints |
| Point-cloud feature | 64-D | Effective 128-D |
| History feature | `24 → 128 → 64` | `24 → 128 → 128` |
| Global condition | 128-D | 256-D |
| Prediction target | Added noise (`epsilon`) | Clean trajectory sample (`x0`) |
| Epoch schedule | 200 epochs | Configured for 4,800 epochs |
| Batch size | 128 | 512 for Windows/Cuboids/Shelves; 128 for Containers |
| Diffusion schedule | 100 DDIM inference steps | 100 training timesteps; 10 DDIM inference steps |
| Public artifact | Paper description and aggregate results | Category configs, per-category run metadata, and four checkpoint hashes |

The historical source supported a multi-dataset training option, but the public
release does not include the joint checkpoint, its resolved configuration, or a
manifest tying one checkpoint hash to every category result. Therefore the
paper's joint-model claim is reported here as a paper claim, not as a capability
verified by the current public artifacts. The release-owned, machine-readable
configuration remains authoritative for the downloadable checkpoints.

The release audit on a validated RTX 4090 machine completed:

- one forward/backward/Adam/EMA step for all four categories;
- finite first-step losses:

  | Category | Loss |
  |:--|--:|
  | Windows | 0.45874408 |
  | Cuboids | 0.56946135 |
  | Shelves | 0.45461127 |
  | Containers | 0.35581228 |

- fixed-test rollout through both `GT_Cond` and `Pred_Cond`;
- save/resume from epoch 0/global step 1 to epoch 1/global step 2, including
  RNG restoration;
- `best.json` resolution and evaluation of its selected EMA checkpoint;
- dataset/cache validation and strict released-checkpoint loading.

The formal Windows batch size was also checked independently on that machine:
one batch-512 optimizer step, EMA update, archived end-of-epoch replay sample,
and one `GT_Cond → Pred_Cond` rollout completed on an RTX 4090, with a
measured peak of **20,786 MiB**. After `prepare windows`, the bounded preflight
can be repeated without starting the 4,800-epoch job:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train.py \
  "config=[covdiffusion,windows]" \
  seed=42 batch_size=512 workers=0 epochs=1 \
  training.max_train_steps=1 training.replay_epoch_sampling=true \
  training.eval_episodes=1 checkpoint.save_ckpt=false \
  processed_data_root="$PWD/artifacts/dataset" \
  evaluation_cache_root="$PWD/artifacts/dataset/evaluation-cache" \
  output_dir="$PWD/outputs/preflight/windows-batch512"
```

The 20,786 MiB value is evidence from that audited run, not a
universal capacity guarantee; framework versions and other GPU processes can
change the peak. In the fresh hash-locked environment, the same batch-512
preflight first exposed allocator fragmentation and then passed with the
wrapper's `max_split_size_mb:128` default. The complete 4,800-epoch job was
subsequently launched separately; this bounded command is still the recommended
preflight.

The complete 4,800-epoch jobs were not rerun during the release audit. The
release therefore validates the full execution chain and convergence recipe,
not a new converged checkpoint.

The manuscript describes 200 epochs, epsilon prediction, and 100 DDIM steps,
while the checkpoint-producing saved configs record 4,800 epochs,
`prediction_type: sample`, and 10 DDIM steps. The public recipe follows the
saved configs tied to the released models.

Byte-identical historical retraining is not claimed: the complete historical
source objects for Windows, Cuboids, and Shelves are unavailable, the original
software/GPU stack differs, and long GPU optimization can vary across kernels.
Use the released checkpoints for exact selected-case comparison; use training
to reproduce the released algorithm and experiment protocol.
