# Data

The public pipeline downloads one pinned, train-ready Hugging Face release:

- [ChenyuanC/3D-CovDiffusion-Train-Ready](https://huggingface.co/datasets/ChenyuanC/3D-CovDiffusion-Train-Ready)
- revision `38bf84c4c8a34bc497167b21b2b68182e0099dc8`

It contains the processed training tensors and the self-contained
evaluation-ready v2 records needed by the public training and inference paths.
No separate raw-data archive or source-repository checkout is required at
runtime.

> **Data attribution.** This work uses modified data derived from
> *MaskPlanner: Learning-Based Object-Centric Motion Generation from 3D Point
> Clouds* by Gabriele Tiboni, [Zenodo record
> 14967945](https://zenodo.org/records/14967945), licensed under CC BY 4.0.

## Prepare

Prepare one category or all four:

```bash
python reproduce.py prepare windows --data-only
python reproduce.py prepare all --data-only
```

`prepare` downloads only the selected category files, then validates the
dataset manifest, Zarr schema, ordered evaluation sample IDs, per-file SHA-256
digests, NPZ schema, array shapes, and finite values. Use the same
`--artifact-root` for every stage; its default is `artifacts/`.

```text
artifacts/
└── dataset/
    ├── dataset_manifest.json
    ├── evaluation_cache_manifest.json
    ├── data/
    │   ├── windows-v2/train.zarr/
    │   ├── cuboids-v2/train.zarr/
    │   ├── shelves-v2/train.zarr/
    │   └── containers-v2/train.zarr/
    └── evaluation-cache/
        ├── windows-v2/*.npz
        ├── cuboids-v2/*.npz
        ├── shelves-v2/*.npz
        └── containers-v2/*.npz
```

To relocate the artifacts:

```bash
python reproduce.py prepare all --data-only --artifact-root /absolute/path
python reproduce.py train windows --artifact-root /absolute/path
python reproduce.py evaluate windows --artifact-root /absolute/path
```

## Processed training tensors

Each category has one `train.zarr`:

```text
train.zarr/
├── meta/episode_ends       int64   [episodes]
├── data/action             float32 [steps, 24]
├── data/stroke_ids         float32 [steps]
└── obs/point_cloud         float32 [episodes, 5120, 3]
```

One 24-D action token concatenates **four ordered 6-DoF poses**. The training
loader builds horizon-16 windows from those tokens and stores each static point
cloud once per episode.

## Self-contained evaluation-ready v2

The release contains all **618** fixed-test samples:

| Category | Samples |
|:--|--:|
| Windows | 200 |
| Cuboids | 200 |
| Shelves | 200 |
| Containers | 18 |

Every NPZ is a complete evaluator input with these fields:

| Field | Shape | Role |
|:--|:--|:--|
| `point_cloud` | `[5120, 3]` | deterministic model observation |
| `trajectory` | `[T, 24]` | ordered model trajectory chunks |
| `gt_trajectory` | `[P, 6]` | metric ground truth |
| `stroke_ids` | `[P]` | ground-truth stroke grouping |
| `mesh_vertices` | `[V, 3]` | coverage geometry |
| `mesh_faces` | `[F, 3]` | coverage triangles |

The NPZ also stores scalar `schema_version` and `sample_id` values. The adjacent
manifest fixes the test order and SHA-256 digest of every file. Evaluation reads
these arrays directly; it performs no mesh sampling, trajectory text parsing,
or runtime reconstruction of evaluator inputs.

Do not reorder files, edit the manifest, or merge a new download into a
corrupted prepared directory. Remove the affected category and rerun
`prepare`. Do not commit Zarr buffers, evaluation NPZ files, checkpoints, or
experiment outputs to Git.
