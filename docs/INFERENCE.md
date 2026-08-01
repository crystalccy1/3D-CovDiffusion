# Evaluation and inference

The release exposes two test paths:

- `evaluate`: fixed-test rollout with a released EMA checkpoint, or with the
  EMA checkpoint selected by a new training run;
- `infer`: one locked seed-42 visualization case per category using the
  archived non-EMA (`raw` weight-variant) export.

Here `raw` describes the policy weights before EMA. It does **not** mean raw
dataset input or a Python/pickle checkpoint; all public model files are
tensor-only safetensors.

## 1. Prepare artifacts

```bash
bash scripts/create_locked_environment.sh 3dcov-cu117
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3dcov-cu117
export PYTHONNOUSERSITE=1
python reproduce.py prepare all
```

One `--artifact-root` (default `artifacts/`) contains the released models,
processed dataset, and all 618 self-contained evaluation-ready v2 records.
Each record locks the preprocessed 5,120-point model input together with the
trajectory, ground truth, stroke IDs, and mesh geometry used by rollout and
metrics.

## 2. Full fixed-test evaluation

Evaluate the released EMA policy:

```bash
python reproduce.py evaluate windows
```

The default `--episodes 0` means the complete available split. Evaluation uses
prediction-conditioned history, rollout seed 42, area-weighted coverage at
spray radius 0.1, final XYZ-only PCD, and translational jerk.

Useful bounded or multi-category runs are:

```bash
python reproduce.py evaluate windows --episodes 1
python reproduce.py evaluate all
```

`evaluate all` applies only to released EMA weights. To evaluate a newly
trained model, resolve one run's exact best EMA entry:

```bash
python reproduce.py evaluate windows \
  --run-dir artifacts/runs/windows/seed42
```

Results go to `outputs/evaluation/` unless `--output-dir` is supplied.
Training seed (stored with the checkpoint) and rollout seed are reported
separately; changing a rollout seed does not create an independently trained
model or reproduce a three-seed table.

### Released EMA full-split references

The following deterministic references were produced on the validated Linux /
RTX 4090 stack with the released EMA tensors at model revision `a2d93ff`, the
pinned evaluation-ready v2 records, rollout seed 42, area-weighted coverage, and
`metrics_version: 3dcov-paper-v1`:

| Category | Episodes | Mean PCD | Mean translational jerk | Mean coverage |
|:--|--:|--:|--:|--:|
| Windows | 200 | 9.0791322 | 0.04119577 | 99.993584% |
| Cuboids | 200 | 9.1523125 | 0.03620493 | 99.943004% |
| Shelves | 200 | 9.1087111 | 0.05182516 | 99.985299% |
| Containers | 18 | 138.2134399 | 0.03346441 | 88.710092% |

These are single released seed-42 checkpoint results, not the paper's
three-seed mean. Minor floating-point differences can occur on another GPU or
software stack; the locked selected-case path below provides the stricter
automatic numerical regression.

## 3. Locked selected-case inference

The cases are fixed in
[`configs/inference/seed42_selected_episodes.json`](../configs/inference/seed42_selected_episodes.json):

| Category | Test index | Sample ID | Policy weights |
|:--|--:|:--|:--|
| Windows | 5 | `810_wr1fr_1` | non-EMA (`raw`) |
| Cuboids | 3 | `669_cube_1001_1285_1263` | non-EMA (`raw`) |
| Shelves | 4 | `box_h620_w500_d220.0_sh1.0_sv2.0` | non-EMA (`raw`) |
| Containers | 1 | `spoegcr3gv` | non-EMA (`raw`) |

Run one case or all four:

```bash
python reproduce.py infer windows
python reproduce.py infer all
```

`infer`:

1. validates safetensors/config/metadata and the selected evaluation-ready NPZ;
2. validates the numerical evaluator by critical-source SHA-256;
3. requires a clean Git checkout;
4. seeds once and runs `GT_Cond` before `Pred_Cond`, matching the archived RNG
   order;
5. writes source/model/data/evaluator provenance;
6. verifies the numeric `Pred_Cond` result automatically.

The first condition mode is retained only because the historical runner did
not reset RNG between modes. `Pred_Cond` is the reported model output.

Outputs use a fresh directory and refuse to reuse stale results:

```text
outputs/selected/selected_windows-v2_s42_ep5/
├── episode_005_GT_Cond/
├── episode_005_Pred_Cond/
├── test_results.json
└── test_scene_metrics.txt
```

## 4. Optional rendering regression

Numeric inference does not need Open3D, PyVista, VTK, or PyTorch3D. To add the
centered prediction PLY:

```bash
python -m pip install --no-build-isolation --require-hashes \
  -r requirements/requirements-visualization-linux-py310-cu117.lock.txt
python reproduce.py infer windows --render
```

`--render` adds a byte-level PLY check for the locked validation environment.
It is optional: CUDA kernels, floating-point serialization, or Open3D versions
can change PLY bytes even when trajectory metrics agree. Without `--render`,
the model inputs, checkpoint, evaluator, provenance, and numeric references are
still checked.

## 5. Selected-case numeric references

These values use the public final-metric implementation for one selected
visualization, not the paper's full test-set mean.

| Category | PCD | Translational jerk | Face-count coverage |
|:--|--:|--:|--:|
| Windows | 8.0002249 | 0.02754691 | 100.0000% |
| Cuboids | 36.7216839 | 0.03619373 | 100.0000% |
| Shelves | 15.3278743 | 0.05712805 | 99.9997% |
| Containers | 45.6353581 | 0.02282970 | 99.0824% |

Metric definitions:

- **PCD:** symmetric mean nearest squared XYZ distance, multiplied by `1e4`.
- **Translational jerk:** mean squared third finite difference over the ordered
  translation sequence.
- **Coverage:** mesh-face centroid within `spray_radius` of an ordered
  trajectory segment. Full evaluation is area weighted; the archived selected
  protocol uses face count.

The selected-case files also retain a legacy second-difference smoothness value
for regression provenance. It is not the final-paper jerk metric. Likewise,
the historical 6-D pose Chamfer used to select training checkpoints is not the
XYZ-only PCD reported here.

## 6. Troubleshooting

- Missing or corrupt artifact: rerun `python reproduce.py prepare <category>`.
- Manifest/data mismatch: remove the affected prepared category and rerun
  `prepare`; do not edit the manifest or an NPZ.
- Evaluator mismatch or dirty checkout: restore or commit the exact evaluator
  before locked `infer`.
- Missing Open3D: omit `--render`, or install the checked-in visualization
  lock shown in section 4.
- Out of memory: evaluate one category at a time and lower `--workers`.
