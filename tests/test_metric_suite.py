import contextlib
import io
import unittest

import numpy as np

from covdiffusion.evaluation.metric_suite import METRIC_NAMES, MetricSuite
from covdiffusion.evaluation.paper_metrics import pointwise_chamfer_distance


def pose(x, y=0.0, z=0.0):
    return [x, y, z, 0.0, 0.0, 0.0]


class MetricSuiteTest(unittest.TestCase):
    def test_public_metric_order_and_count(self):
        suite = MetricSuite()

        self.assertEqual(
            METRIC_NAMES,
            ("pcd", "smoothness", "coverage", "inference_time", "latency"),
        )
        self.assertEqual(suite.metrics, list(METRIC_NAMES))
        self.assertEqual(
            suite.output_metrics_names,
            (
                ("point-wise chamfer distance",),
                ("translational jerk",),
                ("paint coverage %",),
                ("inference time (ms)",),
                ("latency (ms)",),
            ),
        )
        self.assertEqual(suite.tot_num_of_metrics(), 5)

    def test_compute_supports_batches_xyz_reference_padding_and_mesh_padding(self):
        predictions = np.array(
            [
                [pose(0.0), pose(1.0), pose(8.0), pose(27.0)],
                [pose(0.0), pose(1.0), pose(2.0), pose(3.0)],
            ],
            dtype=np.float64,
        )
        xyz_reference = np.array(
            [
                [[0.0, 0.0, 0.0], [27.0, 0.0, 0.0], [-100.0] * 3],
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-100.0] * 3],
            ],
            dtype=np.float64,
        )
        vertices = np.array(
            [
                [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 2.0, 0.0]],
                [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 2.0, 0.0]],
            ]
        )
        faces = np.array(
            [
                [[0, 1, 2], [-1, -1, -1]],
                [[0, 1, 2], [-1, -1, -1]],
            ]
        )

        values = MetricSuite(
            config={"coverage_spray_radius": 1.0}
        ).compute(
            y_pred=predictions,
            y=np.full_like(predictions, 123.0),
            traj_as_pc=xyz_reference,
            batch_data={"mesh_vertices": vertices, "mesh_faces": faces},
            inference_times=[1.0, 3.0],
            latencies=[[2.0, 4.0], [6.0]],
        )

        reference_6d = []
        for episode in xyz_reference:
            valid = episode[~np.all(episode == -100.0, axis=-1)]
            reference_6d.append(
                np.concatenate([valid, np.zeros_like(valid)], axis=-1)
            )
        expected_pcd = np.mean(
            [
                pointwise_chamfer_distance(predictions[index], reference_6d[index])
                for index in range(2)
            ]
        )
        self.assertAlmostEqual(values[0], expected_pcd)
        self.assertAlmostEqual(values[1], 18.0)  # mean of cubic 36 and linear 0
        self.assertAlmostEqual(values[2], 100.0)
        self.assertAlmostEqual(values[3], 2.0)
        self.assertAlmostEqual(values[4], 4.0)

    def test_coverage_honors_area_weighting_and_sequence_mesh_batches(self):
        prediction = np.array([[pose(0.0, 1.0 / 3.0), pose(1.0, 1.0 / 3.0)]])
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
        suite = MetricSuite(
            config={
                "coverage_spray_radius": 0.05,
                "coverage_area_weighted": True,
            },
            metrics=["coverage"],
        )

        result = suite.get_eval_metric(
            "coverage",
            y_pred=prediction,
            batch_data={"mesh_vertices": [vertices], "mesh_faces": [faces]},
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 20.0)

    def test_pprint_uses_selected_metric_order(self):
        suite = MetricSuite(metrics=["coverage", "latency"])
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            suite.pprint([75.0, 2.5], prefix="Summary")

        self.assertEqual(
            output.getvalue(),
            "Summary\n"
            "  paint coverage %: 75.000000\n"
            "  latency (ms): 2.500000\n",
        )

    def test_invalid_metric_and_partially_padded_faces_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported metrics"):
            MetricSuite(metrics=["not-a-metric"])

        suite = MetricSuite(metrics=["coverage"])
        with self.assertRaisesRegex(ValueError, "partially padded"):
            suite.compute(
                y_pred=np.array([[pose(0.0), pose(1.0)]]),
                batch_data={
                    "mesh_vertices": np.zeros((1, 3, 3)),
                    "mesh_faces": np.array([[[0, 1, -1]]]),
                },
            )


if __name__ == "__main__":
    unittest.main()
