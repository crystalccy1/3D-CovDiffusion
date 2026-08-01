#!/usr/bin/env python3
"""Validate a 3D-CovDiffusion train-ready dataset release."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


SCHEMA_VERSION = "3dcov-train-v1"
DEFAULT_CATEGORIES = ("windows-v2", "cuboids-v2", "shelves-v2", "containers-v2")
CHUNK_SIZE = 4096


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    return parser.parse_args()


def logical_sha256(array, chunk_size=CHUNK_SIZE):
    digest = hashlib.sha256()
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array[()]).tobytes())
        return digest.hexdigest()
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size])
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite value found in array {array.path}")
        digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def validate_category(root, category, expected_manifest=None):
    category_root = root / "data" / category
    zarr_path = category_root / "train.zarr"
    manifest_path = category_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if expected_manifest is not None and manifest != expected_manifest:
        raise ValueError(
            f"{category}: category manifest differs from dataset_manifest.json"
        )
    store = zarr.open(str(zarr_path), mode="r")

    if store.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{category}: unexpected schema version")
    if store.attrs.get("category") != category:
        raise ValueError(f"{category}: Zarr category attribute does not match")
    if store.attrs.get("split") != "train":
        raise ValueError(f"{category}: Zarr split attribute is not train")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{category}: manifest schema does not match")
    if store.attrs.get("code_commit") != manifest.get("code_commit"):
        raise ValueError(f"{category}: Zarr/manifest code revisions differ")

    arrays = {
        "meta/episode_ends": store["meta/episode_ends"],
        "data/action": store["data/action"],
        "data/stroke_ids": store["data/stroke_ids"],
        "obs/point_cloud": store["obs/point_cloud"],
    }
    episode_ends = np.asarray(arrays["meta/episode_ends"][:], dtype=np.int64)
    if arrays["meta/episode_ends"].ndim != 1:
        raise ValueError(f"{category}: episode_ends must be one-dimensional")
    if arrays["data/action"].ndim != 2 or arrays["data/action"].shape[1] != 24:
        raise ValueError(f"{category}: action must have shape [steps, 24]")
    if arrays["data/stroke_ids"].ndim != 1:
        raise ValueError(f"{category}: stroke_ids must be one-dimensional")
    if (
        arrays["obs/point_cloud"].ndim != 3
        or tuple(arrays["obs/point_cloud"].shape[1:]) != (5120, 3)
    ):
        raise ValueError(
            f"{category}: point_cloud must have shape [episodes, 5120, 3]"
        )
    if len(episode_ends) == 0 or not np.all(np.diff(episode_ends) > 0):
        raise ValueError(f"{category}: episode boundaries are not strictly increasing")
    if int(episode_ends[-1]) != len(arrays["data/action"]):
        raise ValueError(f"{category}: action length does not match episode boundaries")
    if manifest.get("episodes") != len(episode_ends):
        raise ValueError(f"{category}: manifest episode count does not match data")
    if manifest.get("steps") != int(episode_ends[-1]):
        raise ValueError(f"{category}: manifest step count does not match data")
    if len(arrays["data/stroke_ids"]) != len(arrays["data/action"]):
        raise ValueError(f"{category}: temporal array lengths differ")
    if len(arrays["obs/point_cloud"]) != len(episode_ends):
        raise ValueError(f"{category}: point-cloud count does not match episodes")

    for name, array in arrays.items():
        expected = manifest["arrays"][name]
        if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
            raise ValueError(f"{category}: shape or dtype mismatch for {name}")
        actual_hash = logical_sha256(array)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"{category}: checksum mismatch for {name}")

    logical_bytes = sum(array.nbytes for array in arrays.values())
    if manifest.get("logical_bytes") != logical_bytes:
        raise ValueError(f"{category}: logical byte count does not match data")
    stored_bytes = sum(
        path.stat().st_size for path in zarr_path.rglob("*") if path.is_file()
    )
    if manifest.get("stored_bytes") != stored_bytes:
        raise ValueError(f"{category}: stored byte count does not match Zarr files")

    return {
        "category": category,
        "episodes": len(episode_ends),
        "steps": int(episode_ends[-1]),
        "stored_bytes": manifest["stored_bytes"],
    }


def main():
    args = parse_args()
    top_manifest_path = args.root / "dataset_manifest.json"
    if not top_manifest_path.is_file():
        raise FileNotFoundError(f"Missing top-level manifest: {top_manifest_path}")
    with top_manifest_path.open(encoding="utf-8") as handle:
        top_manifest = json.load(handle)
    if top_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Top-level dataset manifest has an unexpected schema version")
    top_code_commit = top_manifest.get("code_commit")
    if not top_code_commit:
        raise ValueError("Top-level dataset manifest is missing code_commit")
    category_manifests = {
        item.get("category"): item for item in top_manifest.get("categories", [])
    }
    missing = [category for category in args.categories if category not in category_manifests]
    if missing:
        raise ValueError(f"Top-level manifest is missing categories: {missing}")
    mismatched_commits = [
        category
        for category in args.categories
        if category_manifests[category].get("code_commit") != top_code_commit
    ]
    if mismatched_commits:
        raise ValueError(
            f"Category manifests have mismatched code revisions: {mismatched_commits}"
        )
    summaries = [
        validate_category(
            args.root,
            category,
            expected_manifest=category_manifests[category],
        )
        for category in args.categories
    ]
    for summary in summaries:
        print(
            f"{summary['category']}: OK — {summary['episodes']} episodes, "
            f"{summary['steps']} steps, {summary['stored_bytes'] / 1e6:.1f} MB"
        )
    print(f"Validated {len(summaries)} categories with schema {SCHEMA_VERSION}.")


if __name__ == "__main__":
    main()
