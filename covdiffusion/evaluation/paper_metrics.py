"""NumPy implementations of the metrics reported in the final paper.

The training and rollout code has historically depended on PyTorch3D, SciPy,
and scikit-learn for evaluation.  This module intentionally depends only on
NumPy so that released trajectories can be checked independently of the model
runtime.

All trajectory inputs represent one episode and have shape ``[T, D]``, where
``D`` is a positive multiple of six.  Each six-vector is ``(x, y, z, r1, r2,
r3)``.  A six-vector whose values are all exactly ``-100`` is padding.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


PADDING_VALUE = -100.0


def _trajectory_6d(trajectory: np.ndarray, name: str = "trajectory") -> np.ndarray:
    """Validate and reshape one trajectory to ``[T, K, 6]``."""

    array = np.asarray(trajectory, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3 and array.shape[-1] == 6:
        reshaped = array
    elif array.ndim == 2:
        if array.shape[1] <= 0 or array.shape[1] % 6 != 0:
            raise ValueError(
                "{} must have a positive feature dimension divisible by 6; got {}".format(
                    name, array.shape
                )
            )
        reshaped = array.reshape(array.shape[0], array.shape[1] // 6, 6)
    else:
        raise ValueError(
            "{} must have shape [T, D] (D divisible by 6) or [T, K, 6]; got {}".format(
                name, array.shape
            )
        )

    valid = ~np.all(reshaped == PADDING_VALUE, axis=-1)
    if np.any(valid) and not np.all(np.isfinite(reshaped[valid])):
        raise ValueError("{} contains a non-finite value in a non-padding pose".format(name))
    return reshaped


def _valid_pose_mask(trajectory_6d: np.ndarray) -> np.ndarray:
    return ~np.all(trajectory_6d == PADDING_VALUE, axis=-1)


def trajectory_positions(trajectory: np.ndarray) -> np.ndarray:
    """Return valid translational XYZ positions in time-major order.

    Multiple keypoints are flattened after reshaping the final dimension into
    six-vectors.  Orientation is deliberately excluded from every paper metric.
    """

    trajectory_6d = _trajectory_6d(trajectory)
    valid = _valid_pose_mask(trajectory_6d)
    return trajectory_6d[..., :3][valid]


def _nearest_squared_distances(
    source: np.ndarray,
    target: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Compute nearest squared distances without a SciPy/sklearn dependency."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    output = np.empty(source.shape[0], dtype=np.float64)
    for source_start in range(0, source.shape[0], block_size):
        source_block = source[source_start : source_start + block_size]
        best = np.full(source_block.shape[0], np.inf, dtype=np.float64)
        for target_start in range(0, target.shape[0], block_size):
            target_block = target[target_start : target_start + block_size]
            difference = source_block[:, None, :] - target_block[None, :, :]
            distances_squared = np.einsum(
                "ijk,ijk->ij", difference, difference, optimize=True
            )
            best = np.minimum(best, np.min(distances_squared, axis=1))
        output[source_start : source_start + source_block.shape[0]] = best
    return output


def pointwise_chamfer_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    block_size: int = 1024,
) -> float:
    """Compute the final-paper point-wise Chamfer distance (PCD).

    The exact reported formula is::

        1e4 * (mean min ||pred - gt||^2 + mean min ||gt - pred||^2)

    Padding six-vectors are removed and only translational XYZ is used.
    """

    predicted_positions = trajectory_positions(prediction)
    ground_truth_positions = trajectory_positions(ground_truth)
    if predicted_positions.shape[0] == 0:
        raise ValueError("prediction has no valid, non-padding poses")
    if ground_truth_positions.shape[0] == 0:
        raise ValueError("ground_truth has no valid, non-padding poses")

    pred_to_gt = _nearest_squared_distances(
        predicted_positions, ground_truth_positions, block_size
    )
    gt_to_pred = _nearest_squared_distances(
        ground_truth_positions, predicted_positions, block_size
    )
    return float(1.0e4 * (np.mean(pred_to_gt) + np.mean(gt_to_pred)))


def checkpoint_selection_pose_chamfer_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    block_size: int = 1024,
) -> float:
    """Compute the historical 6-D Chamfer used to select checkpoints.

    This is intentionally separate from :func:`pointwise_chamfer_distance`.
    The final paper metric compares XYZ only, while the archived training code
    flattened every 24-D chunk into four weighted ``(XYZ, orientation)`` poses
    and ran symmetric squared-distance Chamfer in all six dimensions. The
    formula is preserved here without the old PyTorch3D dependency.
    """

    predicted = _trajectory_6d(prediction, name="prediction")
    reference = _trajectory_6d(ground_truth, name="ground_truth")
    predicted_poses = predicted[_valid_pose_mask(predicted)]
    reference_poses = reference[_valid_pose_mask(reference)]
    if predicted_poses.shape[0] == 0:
        raise ValueError("prediction has no valid, non-padding poses")
    if reference_poses.shape[0] == 0:
        raise ValueError("ground_truth has no valid, non-padding poses")

    pred_to_gt = _nearest_squared_distances(
        predicted_poses, reference_poses, block_size
    )
    gt_to_pred = _nearest_squared_distances(
        reference_poses, predicted_poses, block_size
    )
    return float(1.0e4 * (np.mean(pred_to_gt) + np.mean(gt_to_pred)))


def translational_jerk(prediction: np.ndarray) -> float:
    """Compute mean squared translational jerk over consecutive valid poses.

    The ``[T, K, 6]`` representation is flattened in time-major order to the
    actual pose sequence before differencing.  For every four-pose valid window
    this computes
    ``||p[t+2] - 3 p[t+1] + 3 p[t] - p[t-1]||^2`` and returns the mean.  Windows
    that contain padding are ignored, so padding gaps are never bridged.  If an
    episode contains no valid four-step window, the metric is undefined and
    ``nan`` is returned.
    """

    trajectory_6d = _trajectory_6d(prediction, name="prediction")
    positions = trajectory_6d[..., :3].reshape(-1, 3)
    valid = _valid_pose_mask(trajectory_6d).reshape(-1)
    if positions.shape[0] < 4:
        return float("nan")

    valid_windows = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
    if not np.any(valid_windows):
        return float("nan")

    third_difference = (
        positions[3:]
        - 3.0 * positions[2:-1]
        + 3.0 * positions[1:-2]
        - positions[:-3]
    )
    squared_norm = np.sum(third_difference * third_difference, axis=-1)
    return float(np.mean(squared_norm[valid_windows]))


def _validated_mesh(
    mesh_vertices: np.ndarray, mesh_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    faces_input = np.asarray(mesh_faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh_vertices must have shape [V, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh_vertices contains a non-finite value")
    if faces_input.ndim != 2 or faces_input.shape[1] != 3:
        raise ValueError("mesh_faces must have shape [F, 3]")
    if not np.issubdtype(faces_input.dtype, np.integer):
        if not np.all(np.isfinite(faces_input)) or not np.all(
            faces_input == np.round(faces_input)
        ):
            raise ValueError("mesh_faces must contain integer vertex indices")
    faces = faces_input.astype(np.int64, copy=False)
    if faces.size and (np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]):
        raise ValueError("mesh_faces contains an out-of-range vertex index")
    return vertices, faces


def coverage_mask(
    prediction: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    spray_radius: float,
    *,
    keypoint_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return which mesh faces are covered by consecutive trajectory segments.

    A face is covered when its centroid lies at most ``spray_radius`` from any
    segment joining two consecutive valid predicted positions.  By default the
    ``[T, K, 6]`` chunks are flattened in time-major order into the full pose
    trajectory.  Supplying ``keypoint_indices`` is a diagnostic mode that
    evaluates the selected keypoint tracks separately.
    """

    if not np.isfinite(spray_radius) or spray_radius < 0:
        raise ValueError("spray_radius must be a finite, non-negative number")

    trajectory_6d = _trajectory_6d(prediction, name="prediction")
    vertices, faces = _validated_mesh(mesh_vertices, mesh_faces)
    covered = np.zeros(faces.shape[0], dtype=bool)
    if faces.shape[0] == 0:
        return covered

    keypoint_count = trajectory_6d.shape[1]
    if keypoint_indices is None:
        paths = [
            (
                trajectory_6d[..., :3].reshape(-1, 3),
                _valid_pose_mask(trajectory_6d).reshape(-1),
            )
        ]
    else:
        selected_keypoints = []
        for raw_index in keypoint_indices:
            index = int(raw_index)
            if index != raw_index or index < 0 or index >= keypoint_count:
                raise ValueError(
                    "keypoint index {} is outside [0, {})".format(
                        raw_index, keypoint_count
                    )
                )
            selected_keypoints.append(index)
        paths = [
            (
                trajectory_6d[:, index, :3],
                _valid_pose_mask(trajectory_6d)[:, index],
            )
            for index in selected_keypoints
        ]

    centroids = vertices[faces].mean(axis=1)
    radius_squared = float(spray_radius) ** 2
    for path_positions, path_valid in paths:
        consecutive = path_valid[:-1] & path_valid[1:]
        segment_indices = np.flatnonzero(consecutive)
        for start_index in segment_indices:
            start = path_positions[start_index]
            end = path_positions[start_index + 1]
            segment = end - start
            length_squared = float(np.dot(segment, segment))
            if length_squared == 0.0:
                distance_squared = np.sum((centroids - start) ** 2, axis=1)
            else:
                relative = centroids - start
                fraction = np.clip((relative @ segment) / length_squared, 0.0, 1.0)
                closest = start + fraction[:, None] * segment
                distance_squared = np.sum((centroids - closest) ** 2, axis=1)
            covered |= distance_squared <= radius_squared
            if np.all(covered):
                return covered
    return covered


def coverage_percentage(
    prediction: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    spray_radius: float,
    *,
    keypoint_indices: Optional[Sequence[int]] = None,
    area_weighted: bool = False,
) -> float:
    """Compute covered-face percentage, optionally weighted by triangle area."""

    vertices, faces = _validated_mesh(mesh_vertices, mesh_faces)
    if faces.shape[0] == 0:
        return float("nan")
    covered = coverage_mask(
        prediction,
        vertices,
        faces,
        spray_radius,
        keypoint_indices=keypoint_indices,
    )
    if not area_weighted:
        return float(100.0 * np.mean(covered))

    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    total_area = float(np.sum(doubled_areas))
    if total_area == 0.0:
        return float("nan")
    return float(100.0 * np.sum(doubled_areas[covered]) / total_area)


def compute_paper_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    spray_radius: float,
    *,
    keypoint_indices: Optional[Sequence[int]] = None,
    area_weighted_coverage: bool = False,
    block_size: int = 1024,
) -> Dict[str, float]:
    """Compute the three final-paper metrics for one predicted episode."""

    return {
        "pcd": pointwise_chamfer_distance(
            prediction, ground_truth, block_size=block_size
        ),
        "jerk": translational_jerk(prediction),
        "coverage": coverage_percentage(
            prediction,
            mesh_vertices,
            mesh_faces,
            spray_radius,
            keypoint_indices=keypoint_indices,
            area_weighted=area_weighted_coverage,
        ),
    }
