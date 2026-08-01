"""Small adapter around the metrics defined by the 3D-CovDiffusion paper.

The rollout runner works with batched PyTorch tensors while
``paper_metrics`` deliberately exposes NumPy functions for one episode.  This
module contains only the conversion and aggregation needed to connect those
two APIs.  Metric definitions live exclusively in ``paper_metrics``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from covdiffusion.evaluation.paper_metrics import (
    coverage_percentage,
    pointwise_chamfer_distance,
    translational_jerk,
)


METRIC_NAMES = ("pcd", "smoothness", "coverage", "inference_time", "latency")


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors without making PyTorch a dependency of this module."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _prediction_episodes(value: Any) -> list[np.ndarray]:
    array = _as_numpy(value)
    if array.ndim == 2:
        return [array]
    if array.ndim in (3, 4):
        return [array[index] for index in range(array.shape[0])]
    raise ValueError(
        "y_pred must have shape [T, D], [B, T, D], or [B, T, K, 6]; "
        f"got {array.shape}"
    )


def _reference_episodes(value: Any, batch_size: int) -> list[np.ndarray]:
    array = _as_numpy(value)
    if array.ndim == 2:
        if batch_size != 1:
            raise ValueError(
                "An unbatched reference can only be paired with one prediction"
            )
        return [array]
    if array.ndim in (3, 4):
        if array.shape[0] != batch_size:
            raise ValueError(
                f"Reference batch size {array.shape[0]} does not match "
                f"prediction batch size {batch_size}"
            )
        return [array[index] for index in range(batch_size)]
    raise ValueError(
        "reference must have shape [N, D], [B, N, D], or [B, T, K, 6]; "
        f"got {array.shape}"
    )


def _xyz_reference_as_6d(reference: np.ndarray) -> np.ndarray:
    """Turn XYZ reference points into poses accepted by ``paper_metrics``.

    A point whose three coordinates are all ``-100`` stays a complete padded
    six-vector.  Orientation values are zero for every valid point because the
    final-paper PCD uses translation only.
    """

    reference = np.asarray(reference)
    if reference.shape[-1] != 3:
        return reference
    converted = np.full(reference.shape[:-1] + (6,), -100.0, dtype=np.float64)
    valid = ~np.all(reference == -100.0, axis=-1)
    converted[..., :3][valid] = reference[valid]
    converted[..., 3:][valid] = 0.0
    return converted


def _select_batch_value(value: Any, index: int, batch_size: int, name: str) -> np.ndarray:
    """Select one mesh from either a batch, a list, or one shared mesh."""

    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(
                f"{name} list length {len(value)} does not match batch size {batch_size}"
            )
        return _as_numpy(value[index])

    array = _as_numpy(value)
    if array.ndim == 2:
        if batch_size != 1:
            raise ValueError(f"Unbatched {name} can only be used with batch size 1")
        return array
    if array.ndim == 3 and array.shape[0] == batch_size:
        return array[index]
    raise ValueError(
        f"{name} must be [N, 3], [B, N, 3], or a length-B sequence; got {array.shape}"
    )


def _mesh_episode(
    batch_data: Mapping[str, Any], index: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        vertices_value = batch_data["mesh_vertices"]
        faces_value = batch_data["mesh_faces"]
    except KeyError as error:
        raise KeyError(
            "coverage requires batch_data['mesh_vertices'] and "
            "batch_data['mesh_faces']"
        ) from error

    vertices = _select_batch_value(
        vertices_value, index, batch_size, "mesh_vertices"
    )
    faces = _select_batch_value(faces_value, index, batch_size, "mesh_faces")

    # Canonical batched artifacts may pad the face table with negative rows.
    if faces.ndim == 2 and faces.shape[1] == 3:
        padded = np.all(faces < 0, axis=1)
        if np.any((faces < 0) & ~padded[:, None]):
            raise ValueError("mesh_faces contains a partially padded face row")
        faces = faces[~padded]
    return vertices, faces


def _flat_mean(values: Any, name: str) -> float:
    if values is None:
        raise ValueError(f"{name} values are required")
    if isinstance(values, np.ndarray) and values.dtype != object:
        flattened = values.astype(np.float64, copy=False).reshape(-1)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        parts = [np.asarray(part, dtype=np.float64).reshape(-1) for part in values]
        flattened = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    else:
        flattened = np.asarray([values], dtype=np.float64)
    if flattened.size == 0:
        raise ValueError(f"{name} values must not be empty")
    if not np.all(np.isfinite(flattened)):
        raise ValueError(f"{name} values must be finite")
    return float(np.mean(flattened))


def _finite_or_nan_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.nanmean(array))


class MetricSuite:
    """Compute the five rollout outputs in their established public order."""

    output_metrics_names = (
        ("point-wise chamfer distance",),
        ("translational jerk",),
        ("paint coverage %",),
        ("inference time (ms)",),
        ("latency (ms)",),
    )
    metric_index = {name: index for index, name in enumerate(METRIC_NAMES)}

    def __init__(self, config: Any = None, metrics: Sequence[str] | None = None):
        selected = tuple(metrics or METRIC_NAMES)
        unknown = set(selected) - set(METRIC_NAMES)
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        if len(set(selected)) != len(selected):
            raise ValueError("metrics must not contain duplicates")
        self.config = config
        self.metrics = list(selected)

    def tot_num_of_metrics(self) -> int:
        return sum(
            len(self.output_metrics_names[self.metric_index[metric]])
            for metric in self.metrics
        )

    def get_eval_metric(
        self,
        metric: str,
        *,
        y_pred: Any,
        y: Any = None,
        traj_as_pc: Any = None,
        batch_data: Mapping[str, Any] | None = None,
        inference_times: Any = None,
        latencies: Any = None,
    ) -> list[float]:
        """Return one metric's outputs as a list for adapter compatibility."""

        if metric not in self.metric_index:
            raise ValueError(f"Unsupported metric: {metric!r}")

        if metric == "inference_time":
            return [_flat_mean(inference_times, "inference_time")]
        if metric == "latency":
            return [_flat_mean(latencies, "latency")]

        predictions = _prediction_episodes(y_pred)
        batch_size = len(predictions)

        if metric == "smoothness":
            return [
                _finite_or_nan_mean(
                    [translational_jerk(prediction) for prediction in predictions]
                )
            ]

        if metric == "pcd":
            reference_value = traj_as_pc if traj_as_pc is not None else y
            if reference_value is None:
                raise ValueError("pcd requires traj_as_pc or y")
            references = _reference_episodes(reference_value, batch_size)
            values = [
                pointwise_chamfer_distance(
                    prediction, _xyz_reference_as_6d(reference)
                )
                for prediction, reference in zip(predictions, references)
            ]
            return [_finite_or_nan_mean(values)]

        if batch_data is None:
            raise ValueError("coverage requires batch_data")
        spray_radius = float(
            _config_value(self.config, "coverage_spray_radius", 0.1)
        )
        area_weighted = bool(
            _config_value(self.config, "coverage_area_weighted", False)
        )
        keypoint_indices = _config_value(
            self.config, "coverage_keypoint_indices", None
        )
        values = []
        for index, prediction in enumerate(predictions):
            vertices, faces = _mesh_episode(batch_data, index, batch_size)
            values.append(
                coverage_percentage(
                    prediction,
                    vertices,
                    faces,
                    spray_radius,
                    keypoint_indices=keypoint_indices,
                    area_weighted=area_weighted,
                )
            )
        return [_finite_or_nan_mean(values)]

    def compute(self, **kwargs: Any) -> np.ndarray:
        values: list[float] = []
        for metric in self.metrics:
            values.extend(self.get_eval_metric(metric, **kwargs))
        return np.asarray(values, dtype=np.float64)

    def pprint(self, values: Any, prefix: str = "Metrics:") -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.tot_num_of_metrics():
            raise ValueError(
                f"Expected {self.tot_num_of_metrics()} metric values, got {array.size}"
            )
        print(prefix)
        value_index = 0
        for metric in self.metrics:
            names = self.output_metrics_names[self.metric_index[metric]]
            for name in names:
                print(f"  {name}: {array[value_index]:.6f}")
                value_index += 1


__all__ = ["METRIC_NAMES", "MetricSuite"]
