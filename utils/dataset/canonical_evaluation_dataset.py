"""Validated reader for the evaluation-ready 3D-CovDiffusion release.

The loader deliberately performs no geometry sampling, trajectory parsing, or
trajectory chunking.  Every array consumed by evaluation is stored in one NPZ
per fixed-test sample and is validated before it is returned.

Evaluation-ready NPZ schema (``3dcov-evaluation-v2``):

``schema_version``
    Scalar Unicode string equal to :data:`SCHEMA_VERSION`.
``sample_id``
    Scalar Unicode string equal to the file's manifest key.
``point_cloud``
    Finite floating-point array with shape ``[5120, 3]``.
``trajectory``
    Finite floating-point model trajectory with shape ``[T, 24]``.  Each row
    contains four ordered six-dimensional poses and contains no padding.
``gt_trajectory``
    Finite floating-point expert trajectory with shape ``[P, 6]``.
``stroke_ids``
    Integer array with shape ``[P]`` aligned with ``gt_trajectory``.
``mesh_vertices``
    Finite floating-point vertex array with shape ``[V, 3]``.
``mesh_faces``
    Integer triangle-index array with shape ``[F, 3]``.

The adjacent ``evaluation_cache_manifest.json`` supplies the exact sample
order and SHA-256 hashes.  This makes the NPZ release self-contained for
rollout and metrics; raw meshes or trajectory text files are not read at
runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


SCHEMA_VERSION = "3dcov-evaluation-v2"
MANIFEST_VERSION = "3dcov-evaluation-ready-v2"
MANIFEST_FILENAME = "evaluation_cache_manifest.json"
POINT_CLOUD_SHAPE = (5120, 3)
REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "point_cloud",
        "trajectory",
        "gt_trajectory",
        "stroke_ids",
        "mesh_vertices",
        "mesh_faces",
    }
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar Unicode string")
    text = str(array.item())
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _floating_matrix(
    value: np.ndarray,
    *,
    name: str,
    columns: int,
    exact_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    array = np.asarray(value)
    expected = f"{exact_shape}" if exact_shape is not None else f"[N, {columns}]"
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}")
    if exact_shape is not None and tuple(array.shape) != exact_shape:
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must have a floating-point dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _integer_matrix(value: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N, {columns}]; got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    return array


def validate_evaluation_sample(
    arrays: Mapping[str, np.ndarray], *, expected_sample_id: Optional[str] = None
) -> dict[str, object]:
    """Validate and normalize one already-loaded evaluation NPZ mapping."""

    keys = set(arrays)
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ValueError("invalid evaluation NPZ keys: " + "; ".join(details))

    schema_version = _scalar_text(arrays["schema_version"], name="schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evaluation schema {schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    sample_id = _scalar_text(arrays["sample_id"], name="sample_id")
    if Path(sample_id).name != sample_id:
        raise ValueError(f"sample_id must be one safe path component: {sample_id!r}")
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise ValueError(
            f"NPZ sample_id {sample_id!r} does not match manifest key "
            f"{expected_sample_id!r}"
        )

    point_cloud = _floating_matrix(
        arrays["point_cloud"],
        name="point_cloud",
        columns=3,
        exact_shape=POINT_CLOUD_SHAPE,
    )
    trajectory = _floating_matrix(
        arrays["trajectory"], name="trajectory", columns=24
    )
    if np.any(trajectory == -100.0):
        raise ValueError(
            "trajectory must contain complete 24-D chunks without padding"
        )
    gt_trajectory = _floating_matrix(
        arrays["gt_trajectory"], name="gt_trajectory", columns=6
    )
    if np.any(gt_trajectory == -100.0):
        raise ValueError("gt_trajectory must not contain padding")

    stroke_ids = np.asarray(arrays["stroke_ids"])
    if stroke_ids.ndim != 1 or stroke_ids.shape[0] != gt_trajectory.shape[0]:
        raise ValueError(
            "stroke_ids must have shape [P] aligned with gt_trajectory; "
            f"got {stroke_ids.shape} and {gt_trajectory.shape}"
        )
    if not np.issubdtype(stroke_ids.dtype, np.integer):
        raise ValueError("stroke_ids must have an integer dtype")

    mesh_vertices = _floating_matrix(
        arrays["mesh_vertices"], name="mesh_vertices", columns=3
    )
    mesh_faces = _integer_matrix(arrays["mesh_faces"], name="mesh_faces", columns=3)
    if np.min(mesh_faces) < 0 or np.max(mesh_faces) >= mesh_vertices.shape[0]:
        raise ValueError("mesh_faces contains an out-of-range vertex index")

    # Preserve released numeric values.  Only index arrays are normalized to
    # int64 so downstream PyTorch collation has one stable representation.
    return {
        "schema_version": schema_version,
        "sample_id": sample_id,
        "point_cloud": point_cloud,
        "trajectory": trajectory,
        "gt_trajectory": gt_trajectory,
        "stroke_ids": stroke_ids.astype(np.int64, copy=False),
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces.astype(np.int64, copy=False),
    }


class CanonicalEvaluationDataset:
    """Read a fixed category split from self-contained canonical NPZ files."""

    def __init__(
        self,
        root,
        category,
        *,
        split: str = "test",
        manifest_path=None,
        verify_hashes: bool = False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.category = str(category)
        self.split = str(split)
        self.verify_hashes = bool(verify_hashes)

        if self.split != "test":
            raise ValueError("The canonical evaluation release contains only test splits")
        if Path(self.category).name != self.category or not self.category:
            raise ValueError(f"invalid evaluation category: {self.category!r}")
        if not self.root.is_dir():
            raise FileNotFoundError(f"evaluation cache root not found: {self.root}")

        self.category_root = self.root / self.category
        if not self.category_root.is_dir():
            raise FileNotFoundError(
                f"evaluation category directory not found: {self.category_root}"
            )

        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.root.parent / MANIFEST_FILENAME
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"evaluation-ready manifest not found: {self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported evaluation manifest format "
                f"{manifest.get('format_version')!r}; expected {MANIFEST_VERSION!r}. "
                "Regenerate the Hugging Face evaluation-ready artifact: the v2 "
                "release must embed chunked trajectories and mesh geometry."
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version must be {SCHEMA_VERSION!r}"
            )

        entry = (manifest.get("categories") or {}).get(self.category)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest has no category {self.category!r}")
        if entry.get("split") != self.split:
            raise ValueError(
                f"manifest split for {self.category!r} is {entry.get('split')!r}; "
                f"requested {self.split!r}"
            )
        sample_ids = entry.get("sample_ids")
        files = entry.get("files")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("manifest sample_ids must be a non-empty ordered list")
        if not isinstance(files, dict):
            raise ValueError("manifest files must map sample IDs to SHA-256 digests")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("manifest sample_ids contains duplicates")
        if set(files) != set(sample_ids):
            raise ValueError("manifest files must exactly cover ordered sample_ids")

        for sample_id in sample_ids:
            if not isinstance(sample_id, str) or Path(sample_id).name != sample_id:
                raise ValueError(f"invalid manifest sample ID: {sample_id!r}")
            digest = files[sample_id]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid SHA-256 for sample {sample_id!r}")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    f"invalid SHA-256 for sample {sample_id!r}"
                ) from error

        self.sample_ids = tuple(sample_ids)
        self.files = dict(files)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        path = self.category_root / f"{sample_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"evaluation sample not found: {path}")
        if self.verify_hashes:
            actual = _sha256_file(path)
            expected = self.files[sample_id]
            if actual != expected:
                raise ValueError(
                    f"evaluation sample SHA-256 mismatch for {sample_id}: "
                    f"{actual}; expected {expected}"
                )

        try:
            with np.load(path, allow_pickle=False) as arrays:
                return validate_evaluation_sample(
                    arrays, expected_sample_id=sample_id
                )
        except Exception as error:
            raise RuntimeError(f"invalid evaluation sample {sample_id!r}: {path}") from error
