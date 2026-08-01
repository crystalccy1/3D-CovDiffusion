# Reproducibility

The public interface has four commands and two explicit paths:

```text
                              prepare
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        processed training data          released weights
                 │                               │
               train                    evaluate / infer
                 │
        evaluate --run-dir
```

Run from the repository root on Linux x86_64 with an NVIDIA GPU. The exact
validated stack is Python 3.10.18 and PyTorch 1.13.1+cu117:

```bash
bash scripts/create_locked_environment.sh 3dcov-cu117
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3dcov-cu117
export PYTHONNOUSERSITE=1
```

Pass `--visualization` to the creation script when byte-level PLY rendering is
needed. The installer uses pinned build tools followed by hash-checked runtime
locks with build isolation disabled, which is required by the locked
`antlr4-python3-runtime` source distribution. It refuses to change an existing
environment. The shorter `pip install -r requirements.txt` is a convenience
path, not the exact transitive lock. PyTorch3D and external experiment logging
are not part of either path.

The lock target is Linux x86_64 with glibc 2.31 or newer. Regeneration requires
exactly uv 0.10.0 and is encoded in `scripts/lock_requirements.sh`; changing a
constraint requires rerunning the complete validation suite before accepting
the new lock.

## 1. Shared preparation

```bash
python reproduce.py prepare all
```

`prepare` obtains and validates:

- processed Zarr tensors for all four training categories;
- all 618 self-contained evaluation-ready v2 records, which lock preprocessed
  model inputs, trajectories, ground truth, stroke IDs, and mesh geometry for
  every fixed-test case;
- released EMA and non-EMA tensor-only checkpoint variants.

`--artifact-root` is the only public input-artifact root. Use the same value for
all stages; the default is `artifacts/`.

`all` is supported by all four commands, with these boundaries:

- `prepare all`: prepare all categories;
- `train all`: train all four sequentially from new run directories;
- `evaluate all`: evaluate all **released EMA** models;
- `infer all`: replay all four locked selected cases;
- resume and `evaluate --run-dir` operate on one category/run only.

## 2. Retraining path

```bash
python reproduce.py train windows
python reproduce.py evaluate windows \
  --run-dir artifacts/runs/windows/seed42
```

Training uses processed Hugging Face tensors for optimization and fixed
rollout evaluation for top-k selection. `latest.ckpt` is the resumable state;
`checkpoints/best.json` records the selected checkpoint, full-precision
historical 6-D pose-Chamfer score, epoch, and global step. Evaluation resolves
that manifest rather than guessing from a filename.

Resume restores the saved Python/NumPy/Torch RNG streams and Torch backend
switches together. This matters because the historical rollout changes cuDNN
to deterministic mode after the first training epoch; restoring RNG alone
would not reproduce the uninterrupted execution state.

The public wrapper injects the formal batch size: 512 for Windows, Cuboids,
and Shelves, and 128 for Containers. Old direct `train.py` commands must add
the corresponding `batch_size` explicitly. The wrapper defaults
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`; the fresh locked environment
needed this allocator split to avoid reservation fragmentation at Windows
batch 512. During training, `latest.ckpt` is updated on the five-epoch
checkpoint cadence (and at normal completion), so a crash resumes from that
saved boundary and can lose up to five epochs. One checkpoint is about 4.084
GB; `latest` plus top-1 `best` is about 8.17 GB, with about 12.25 GB temporary
peak space during atomic replacement.

A bounded execution check is available:

```bash
python reproduce.py train windows --smoke
```

`--smoke` forces batch size 2, one epoch/one optimizer step, checkpointing off,
and end-of-epoch replay sampling off. It validates bounded execution, not the
formal batch-size memory footprint.

See [`TRAINING.md`](TRAINING.md) for the fixed profile, save/resume protocol,
the repeatable formal-batch preflight command, and why the historical 6-D
selector is distinct from final XYZ-only PCD.

## 3. Released-model path

Full fixed-test metrics with the released EMA policy:

```bash
python reproduce.py evaluate windows
python reproduce.py evaluate all
```

Locked selected-case replay with the archived non-EMA (`raw` weight-variant)
policy:

```bash
python reproduce.py infer windows
python reproduce.py infer all
```

The term `raw` refers only to policy weights before EMA, not to raw input data.
Locked inference preserves `GT_Cond → Pred_Cond` RNG order and verifies the
evaluation-ready input, checkpoint, evaluator, provenance, and numeric result.

Numeric execution is the primary regression. Add `--render` only after using
the visualization lock (or creating the environment with `--visualization`)
for an optional locked-environment PLY hash; rendered bytes are not required
for numeric reproduction.

See [`INFERENCE.md`](INFERENCE.md) for selected cases and metric definitions.

## 4. Evidence and claim boundary

| Claim | Release evidence |
|:--|:--|
| Processed data is consumable | Zarr schemas, finite values, and logical hashes validated |
| Fixed-test inputs are deterministic | all 618 evaluation-ready files are ordered by the manifest and bound to SHA-256 |
| Rollout inputs are complete | every record embeds the model input, ground truth, stroke IDs, and mesh geometry |
| Models match the code | all EMA/non-EMA safetensors strict-load |
| Optimization executes | forward/backward/Adam/EMA step passed for all four categories |
| Formal Windows batch fits the audited GPU | batch 512 plus EMA replay and `GT_Cond → Pred_Cond` rollout passed on the validated RTX 4090 machine; measured peak 20,786 MiB |
| Complete train/eval handoff executes | fixed rollout, top-k manifest, save/resume, and `--run-dir` evaluation passed |
| Selected inference is reproducible | four cases locked by data/model/code/numeric checks |

The release does **not** claim:

- a new full 4,800-epoch convergence run during the audit;
- byte-identical historical checkpoints or the same selected epoch across a
  different software/GPU stack;
- reconstruction of the paper's three-seed mean from one public seed-42 model
  per category;
- reproduction of the full baseline table, whose checkpoints and entry points
  are outside this minimal path.

The manuscript and saved run configs disagree on duration and diffusion
settings. This repository follows the checkpoint-producing configs: 4,800
epochs, `prediction_type: sample`, and 10 DDIM inference steps.
