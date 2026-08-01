import copy
import json
from pathlib import Path
import unittest

from scripts.verify_selected_inference import verify_metric_reference


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/inference/seed42_selected_episodes.json"


class SelectedInferenceVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        cls.tolerances = cls.profile["protocol"]["metric_tolerances"]

    def test_validated_cross_driver_measurements_match(self):
        measurements = {
            "windows": [
                (8.000443293903636, 0.027545481080820902, 100.0),
                (8.000443293903636, 0.027545481080820902, 100.0),
            ],
            "cuboids": [
                (36.721153211687856, 0.036193281440118734, 100.0),
                (36.721153211687856, 0.036193281440118734, 100.0),
            ],
            "shelves": [
                (15.327845522726685, 0.05712825559437023, 99.99968475738929),
                (15.327845522726685, 0.05712825559437023, 99.99968475738929),
            ],
            "containers": [
                (45.634531961716434, 0.02282900826035007, 99.08239120334521),
                (45.634531961716434, 0.02282900826035007, 99.08239120334521),
            ],
        }
        for category, runs in measurements.items():
            self.assertEqual(runs[0], runs[1])
            for run_index, values in enumerate(runs, start=1):
                record = dict(zip(("pcd", "jerk", "coverage"), values))
                with self.subTest(category=category, run=run_index):
                    verify_metric_reference(
                        Path("test_results.json"),
                        record,
                        self.profile["categories"][category][
                            "public_metric_reference"
                        ],
                        self.tolerances,
                    )

    def test_each_metric_regression_outside_tolerance_fails(self):
        reference = self.profile["categories"]["windows"][
            "public_metric_reference"
        ]
        for key in ("pcd", "jerk", "coverage"):
            record = dict(reference)
            record[key] += self.tolerances[key]["abs"] * 1.01
            with self.subTest(metric=key), self.assertRaisesRegex(
                ValueError, "within abs="
            ):
                verify_metric_reference(
                    Path("test_results.json"),
                    record,
                    reference,
                    self.tolerances,
                )

    def test_profile_cannot_relax_verifier_owned_caps(self):
        reference = self.profile["categories"]["windows"][
            "public_metric_reference"
        ]
        invalid_values = (float("inf"), float("nan"), 1.01e-3, -1.0)
        for value in invalid_values:
            tolerances = copy.deepcopy(self.tolerances)
            tolerances["pcd"]["abs"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_metric_reference(
                    Path("test_results.json"),
                    reference,
                    reference,
                    tolerances,
                )

        tolerances = copy.deepcopy(self.tolerances)
        tolerances["pcd"]["rel"] = 1e-12
        with self.assertRaisesRegex(ValueError, "verifier-owned"):
            verify_metric_reference(
                Path("test_results.json"),
                reference,
                reference,
                tolerances,
            )

        tolerances = copy.deepcopy(self.tolerances)
        del tolerances["pcd"]
        with self.assertRaisesRegex(ValueError, "no pcd metric tolerance"):
            verify_metric_reference(
                Path("test_results.json"),
                reference,
                reference,
                tolerances,
            )

    def test_non_finite_metric_fails(self):
        reference = self.profile["categories"]["windows"][
            "public_metric_reference"
        ]
        for value in (float("inf"), float("nan")):
            record = dict(reference)
            record["pcd"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "non-finite pcd metric"
            ):
                verify_metric_reference(
                    Path("test_results.json"),
                    record,
                    reference,
                    self.tolerances,
                )


if __name__ == "__main__":
    unittest.main()
