import math
import unittest

import numpy as np

from covdiffusion.evaluation.paper_metrics import (
    checkpoint_selection_pose_chamfer_distance,
    coverage_mask,
    coverage_percentage,
    pointwise_chamfer_distance,
    trajectory_positions,
    translational_jerk,
)


def pose(x, y=0.0, z=0.0):
    return [x, y, z, 7.0, 8.0, 9.0]


class PaperMetricsTest(unittest.TestCase):
    def test_pcd_uses_xyz_squared_distance_and_ignores_padding(self):
        prediction = np.array(
            [pose(0.0), pose(1.0), [-100.0] * 6], dtype=np.float64
        )
        ground_truth = np.array([pose(0.0), pose(2.0)], dtype=np.float64)

        # Each directed mean is (0^2 + 1^2) / 2, so their sum is 1.
        self.assertAlmostEqual(
            pointwise_chamfer_distance(prediction, ground_truth), 10_000.0
        )

        # Orientation does not participate in PCD.
        ground_truth[:, 3:] = 1_000_000.0
        self.assertAlmostEqual(
            pointwise_chamfer_distance(prediction, ground_truth), 10_000.0
        )

    def test_checkpoint_selection_chamfer_uses_all_six_pose_values(self):
        prediction = np.array([pose(0.0, 0.0, 0.0)], dtype=np.float64)
        prediction[0, 3:] = 0.0
        ground_truth = prediction.copy()
        ground_truth[0, 3] = 2.0

        self.assertEqual(pointwise_chamfer_distance(prediction, ground_truth), 0.0)
        self.assertEqual(
            checkpoint_selection_pose_chamfer_distance(prediction, ground_truth),
            80_000.0,
        )

    def test_positions_reshape_multiple_keypoints(self):
        trajectory = np.array(
            [
                pose(1.0) + pose(2.0),
                pose(3.0) + [-100.0] * 6,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            trajectory_positions(trajectory),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        )

    def test_translational_jerk_is_third_finite_difference(self):
        # x=t^3 has constant third finite difference 6, whose squared norm is 36.
        trajectory = np.array([pose(float(t**3)) for t in range(6)])
        self.assertAlmostEqual(translational_jerk(trajectory), 36.0)

    def test_jerk_flattens_ordered_pose_chunks(self):
        # Two 2-pose chunks encode the sequence x=[0, 1, 8, 27].
        trajectory = np.array(
            [pose(0.0) + pose(1.0), pose(8.0) + pose(27.0)]
        )
        self.assertAlmostEqual(translational_jerk(trajectory), 36.0)

    def test_jerk_does_not_bridge_padding_gap(self):
        trajectory = np.array(
            [pose(0.0), pose(1.0), [-100.0] * 6, pose(8.0), pose(27.0)]
        )
        self.assertTrue(math.isnan(translational_jerk(trajectory)))

    def test_coverage_uses_consecutive_segments_and_supports_area_weighting(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [10.0, 2.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        trajectory = np.array([pose(0.0, 1.0 / 3.0), pose(1.0, 1.0 / 3.0)])

        np.testing.assert_array_equal(
            coverage_mask(trajectory, vertices, faces, spray_radius=0.05),
            np.array([True, False]),
        )
        self.assertAlmostEqual(
            coverage_percentage(trajectory, vertices, faces, spray_radius=0.05),
            50.0,
        )
        # Triangle areas are 0.5 and 2.0, respectively.
        self.assertAlmostEqual(
            coverage_percentage(
                trajectory,
                vertices,
                faces,
                spray_radius=0.05,
                area_weighted=True,
            ),
            20.0,
        )

    def test_coverage_does_not_connect_across_padding(self):
        vertices = np.array(
            [[4.0, -1.0, 0.0], [6.0, -1.0, 0.0], [5.0, 2.0, 0.0]]
        )
        faces = np.array([[0, 1, 2]])  # Centroid is [5, 0, 0].
        trajectory = np.array([pose(0.0), [-100.0] * 6, pose(10.0)])
        self.assertFalse(
            coverage_mask(trajectory, vertices, faces, spray_radius=0.1)[0]
        )

    def test_single_chunk_coverage_uses_multiple_ordered_poses(self):
        vertices = np.array(
            [[4.0, -1.0, 0.0], [6.0, -1.0, 0.0], [5.0, 2.0, 0.0]]
        )
        faces = np.array([[0, 1, 2]])
        trajectory = np.array([pose(0.0) + pose(10.0)])
        self.assertTrue(
            coverage_mask(trajectory, vertices, faces, spray_radius=0.1)[0]
        )

    def test_invalid_feature_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by 6"):
            pointwise_chamfer_distance(np.zeros((2, 5)), np.zeros((2, 6)))


if __name__ == "__main__":
    unittest.main()
