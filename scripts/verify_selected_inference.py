#!/usr/bin/env python3
"""Verify one selected-episode inference output against its release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs/inference/seed42_selected_episodes.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce import DATASET_REPO, DATASET_REVISION, evaluator_fingerprint


MAX_METRIC_TOLERANCES = {
    "pcd": {"abs": 1e-3, "rel": 0.0},
    "jerk": {"abs": 2e-6, "rel": 0.0},
    "coverage": {"abs": 1e-4, "rel": 0.0},
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        required=True,
        choices=("windows", "cuboids", "shelves", "containers"),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--skip-ply-hash",
        action="store_true",
        help=(
            "Validate locked numeric results and provenance without requiring "
            "an exported PLY file."
        ),
    )
    return parser.parse_args()


def verify_metric_reference(
    result_path: Path,
    record: dict,
    reference: dict,
    tolerances: dict,
) -> None:
    """Check published metrics with explicit cross-driver float tolerances."""

    for key in ("pcd", "jerk", "coverage"):
        maximum = MAX_METRIC_TOLERANCES[key]
        tolerance = tolerances.get(key)
        if not isinstance(tolerance, dict):
            raise ValueError(f"Inference profile has no {key} metric tolerance")
        try:
            absolute = float(tolerance["abs"])
            relative = float(tolerance.get("rel", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Inference profile has an invalid {key} metric tolerance"
            ) from error
        if not math.isfinite(absolute) or not math.isfinite(relative):
            raise ValueError(f"Inference profile has a non-finite {key} tolerance")
        if absolute < 0.0 or relative < 0.0:
            raise ValueError(f"Inference profile has a negative {key} tolerance")
        if absolute > maximum["abs"] or relative > maximum["rel"]:
            raise ValueError(
                f"Inference profile exceeds the verifier-owned {key} "
                f"tolerance cap"
            )
        if key not in reference:
            raise ValueError(f"Inference profile has no {key} metric reference")
        actual_value = float(record[key])
        reference_value = float(reference[key])
        if not math.isfinite(actual_value) or not math.isfinite(reference_value):
            raise ValueError(f"{result_path}: non-finite {key} metric")
        if not math.isclose(
            actual_value,
            reference_value,
            rel_tol=relative,
            abs_tol=absolute,
        ):
            raise ValueError(
                f"{result_path}: {key}={record[key]!r}, expected "
                f"{reference[key]!r} within abs={absolute}, rel={relative}"
            )


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    case = profile["categories"][args.category]
    run_dir = args.run_dir.expanduser().resolve()
    result_path = run_dir / "test_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    expected = {
        "dataset": case["dataset"],
        "training_seed": case["training_seed"],
        "rollout_seed": case["rollout_seed"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"{result_path}: {key}={result.get(key)!r}, expected {value!r}"
            )
    if result.get("summary", {}).get("n_episodes") != 1:
        raise ValueError(f"{result_path} is not a one-episode inference result")
    records = result.get("per_episode", [])
    if len(records) != 1:
        raise ValueError(f"{result_path} must contain exactly one episode record")
    source_index = int(case["test_split_index"])
    if int(records[0].get("episode", -1)) != source_index:
        raise ValueError(
            f"Result episode={records[0].get('episode')!r}, expected fixed split "
            f"index {source_index}"
        )

    protocol = result.get("protocol")
    expected_protocol = {
        "metrics_version": "3dcov-paper-v1",
        "coverage": profile["protocol"]["coverage_mode"],
        "coverage_spray_radius": profile["protocol"]["spray_radius"],
        "condition_mode": profile["protocol"]["reported_condition_mode"],
        "rollout_seed": case["rollout_seed"],
    }
    if not isinstance(protocol, dict):
        raise ValueError(f"{result_path} has no protocol block")
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"{result_path}: protocol {key}={protocol.get(key)!r}, "
                f"expected {value!r}"
            )

    provenance = result.get("release_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{result_path} has no release_provenance block")
    expected_provenance = {
        "model_repository": profile["model_repository"],
        "model_revision": profile["model_revision"],
        "evaluation_cache_repository": DATASET_REPO,
        "evaluation_cache_revision": DATASET_REVISION,
        "evaluation_code": profile["evaluation_code"],
        "evaluator_fingerprint": evaluator_fingerprint(profile),
        "profile_sha256": sha256_file(args.profile),
        "checkpoint_sha256": case["public_checkpoint"]["model_sha256"],
        "config_sha256": case["public_checkpoint"]["config_sha256"],
        "metrics_sha256": case["public_checkpoint"]["metrics_sha256"],
        "test_split_sha256": case["test_split_sha256"],
        "evaluation_ready_sha256": case["evaluation_ready_sha256"],
        "weight_variant": "raw",
        "sample_id": case["sample_id"],
        "test_split_index": source_index,
        "condition_mode_sequence": profile["protocol"][
            "condition_mode_sequence"
        ],
        "reported_condition_mode": "Pred_Cond",
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"{result_path}: provenance {key}={provenance.get(key)!r}, "
                f"expected {value!r}"
            )
    git_state = provenance.get("actual_git_state")
    if not isinstance(git_state, dict):
        raise ValueError(f"{result_path}: actual Git state was not recorded")
    if git_state.get("dirty") is not False:
        raise ValueError(f"{result_path}: evaluator worktree was dirty")

    record = records[0]
    numeric_keys = (
        "pcd",
        "jerk",
        "coverage",
        "inference_time_ms",
        "latency_ms",
    )
    for key in numeric_keys:
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{result_path}: non-finite {key}={value!r}")
    summary = result["summary"]
    for summary_key, record_key in (
        ("mean_pcd", "pcd"),
        ("mean_jerk", "jerk"),
        ("mean_coverage", "coverage"),
    ):
        if not math.isclose(
            float(summary.get(summary_key, math.nan)),
            float(record[record_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{result_path}: summary {summary_key} does not match the "
                f"single episode record"
            )
    public_reference = case.get("public_metric_reference")
    if not isinstance(public_reference, dict):
        raise ValueError(f"Inference profile has no public metric reference")
    tolerances = profile.get("protocol", {}).get("metric_tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("Inference profile has no metric tolerances")
    verify_metric_reference(result_path, record, public_reference, tolerances)

    verified_ply = None
    if not args.skip_ply_hash:
        prediction_path = (
            run_dir
            / f"episode_{source_index:03d}_Pred_Cond"
            / "pred_trajectory_points.ply"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(
                "Prediction PLY not found. Re-run with --save-artifacts or pass "
                "--skip-ply-hash for metrics-only validation."
            )
        actual_sha = sha256_file(prediction_path)
        expected_sha = case["public_render_reference"][
            "prediction_ply_sha256"
        ]
        if actual_sha != expected_sha:
            raise ValueError(
                f"Prediction PLY SHA mismatch: {actual_sha}; expected {expected_sha}"
            )
        verified_ply = str(prediction_path)

    print(
        json.dumps(
            {
                "verified": True,
                "category": args.category,
                "dataset": case["dataset"],
                "test_split_index": source_index,
                "sample_id": case["sample_id"],
                "result": str(result_path),
                "prediction_ply": verified_ply,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
