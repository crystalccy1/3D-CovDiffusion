"""Deterministic rollout evaluator for the released 3D-CovDiffusion policy.

The public runner intentionally implements one protocol only: autoregressive
horizon-16 action-chunk generation on the released fixed test split,
optionally in ground-truth- or prediction-conditioned mode. It reports the
five paper metrics, writes one machine-readable result file, and can export the optional
prediction PLY. Historical OOD, ablation, camera, GIF, and interactive
rendering paths are deliberately outside the release.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import tqdm

from covdiffusion.common.pytorch_util import dict_apply
from covdiffusion.evaluation.paper_metrics import (
    checkpoint_selection_pose_chamfer_distance,
)
from covdiffusion.evaluation.metric_suite import MetricSuite
from covdiffusion.env_runner.base_runner import BaseRunner
from covdiffusion.policy.base_policy import BasePolicy


CONDITION_MODES = ("GT_Cond", "Pred_Cond")
METRICS = ("pcd", "smoothness", "coverage", "inference_time", "latency")


def _source_episode_index(data, fallback: int) -> int:
    value = data.get("episode_idx", fallback)
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(np.asarray(value).reshape(-1)[0])
    return int(value)


def _metric_record(values, episode: int) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != len(METRICS) or not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"Expected {len(METRICS)} finite metrics, got {array.tolist()}"
        )
    return {
        "episode": int(episode),
        "pcd": float(array[0]),
        "jerk": float(array[1]),
        "coverage": float(array[2]),
        "inference_time_ms": float(array[3]),
        "latency_ms": float(array[4]),
    }


class CovDiffusionRunner(BaseRunner):
    """Run the fixed evaluation protocol one episode at a time."""

    def __init__(
        self,
        output_dir=None,
        eval_episodes=10,
        tqdm_interval_sec=1.0,
        batch_size=1,
        num_workers=0,
        seed=42,
        training_seed=None,
        save_artifacts=False,
        condition_modes=None,
        checkpoint_selection=False,
        config=None,
        chamfer_distance_threshold=0.05,
    ):
        super().__init__(output_dir)
        del num_workers, chamfer_distance_threshold
        if int(batch_size) != 1:
            raise ValueError("The released evaluator requires batch_size=1.")
        if config is None:
            raise ValueError("CovDiffusionRunner requires the checkpoint config.")

        self.eval_episodes = int(eval_episodes)
        self.tqdm_interval_sec = float(tqdm_interval_sec)
        self.seed = int(seed)
        self.training_seed = int(training_seed if training_seed is not None else seed)
        self.save_artifacts = bool(save_artifacts)
        self.checkpoint_selection = bool(checkpoint_selection)
        self.condition_modes = list(condition_modes or ["Pred_Cond"])
        invalid = set(self.condition_modes) - set(CONDITION_MODES)
        if invalid:
            raise ValueError(f"Unsupported condition modes: {sorted(invalid)}")
        self.config = config

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _validate_ground_truth(batch) -> torch.Tensor:
        trajectory = batch["full_trajectory"]
        if not torch.is_tensor(trajectory):
            raise TypeError("full_trajectory must be a tensor")
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
        if trajectory.ndim != 3 or trajectory.shape[0] != 1:
            raise ValueError(
                "Expected full_trajectory shape [1, time, 24], "
                f"got {tuple(trajectory.shape)}"
            )
        if trajectory.shape[-1] != 24 or trajectory.shape[1] == 0:
            raise ValueError(
                "full_trajectory must contain at least one 24-D pose chunk"
            )
        if (trajectory == -100).any():
            raise ValueError("The rollout trajectory must not contain padding rows.")
        return trajectory

    @staticmethod
    def _rollout(policy, batch, ground_truth, condition_mode):
        # Allocate exactly as the archived runner did before the first policy
        # call.  Besides avoiding uninitialised data, this preserves CUDA
        # allocator order for the locked selected-episode regression.
        prediction = torch.zeros(
            (1, ground_truth.shape[1], ground_truth.shape[2]),
            device=policy.device,
        )
        point_cloud = batch["obs"]["point_cloud"]
        policy.reset()
        observation = {
            "obs": {"point_cloud": point_cloud},
            "prev_true_trajectory": ground_truth[:, 0, :],
        }
        inference_times = []
        latencies = []
        completed = 0

        while completed < ground_truth.shape[1]:
            iteration_start = time.perf_counter()
            inference_start = time.perf_counter()
            result = policy.predict_action(observation)
            inference_times.append(
                (time.perf_counter() - inference_start) * 1000.0
            )

            segment = result.get("action")
            if not torch.is_tensor(segment) or segment.ndim != 3:
                raise ValueError("policy.predict_action() must return action [B, T, D]")
            if segment.shape[0] != 1 or segment.shape[-1] != 24:
                raise ValueError(f"Unexpected action shape: {tuple(segment.shape)}")
            count = min(segment.shape[1], ground_truth.shape[1] - completed)
            if count <= 0:
                raise RuntimeError("Policy returned an empty action segment")
            current = segment[:, :count]
            if (current == -100).any() or not torch.isfinite(current).all():
                raise RuntimeError("Policy returned padding or non-finite values")
            prediction[:, completed : completed + count] = current
            completed += count

            if condition_mode == "Pred_Cond":
                # Preserve the historical feedback rule: use the last element
                # of the complete predicted chunk, even on the final short fill.
                observation["prev_true_trajectory"] = segment[:, -1, :]
            else:
                observation["prev_true_trajectory"] = ground_truth[:, completed - 1, :]
            latencies.append((time.perf_counter() - iteration_start) * 1000.0)

        return prediction, inference_times, latencies

    @staticmethod
    def _trajectory_xyz_keypoint_major(trajectory) -> np.ndarray:
        if torch.is_tensor(trajectory):
            trajectory = trajectory.detach().cpu().numpy()
        array = np.asarray(trajectory)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2 or array.shape[1] != 24:
            raise ValueError(f"Unexpected trajectory shape for PLY: {array.shape}")
        points = np.concatenate(
            [array[:, offset : offset + 3] for offset in range(0, 24, 6)],
            axis=0,
        )
        return points[~np.all(points == -100.0, axis=1)]

    @classmethod
    def _save_centered_trajectory_ply(cls, trajectory, save_path: Path) -> None:
        try:
            import open3d as o3d
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PLY export requires `pip install -r requirements-visualization.txt`."
            ) from error

        points = cls._trajectory_xyz_keypoint_major(trajectory)
        if points.shape[0] == 0:
            raise ValueError("Cannot export an empty predicted trajectory")
        points = points - points.mean(axis=0)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if not o3d.io.write_point_cloud(str(save_path), cloud):
            raise RuntimeError(f"Open3D failed to write trajectory PLY: {save_path}")

    @staticmethod
    def _aggregate(results, metrics_handler, prefix: str) -> dict:
        if results:
            mean_metrics = np.nanmean(np.asarray(results, dtype=float), axis=0)
        else:
            mean_metrics = np.full(metrics_handler.tot_num_of_metrics(), np.nan)
        aggregated = {"mean_metrics": mean_metrics}
        value_index = 0
        for metric in metrics_handler.metrics:
            output_names = metrics_handler.output_metrics_names[
                metrics_handler.metric_index[metric]
            ]
            for output_name in output_names:
                safe_name = output_name.replace(" ", "_").replace("%", "perc")
                aggregated[f"{prefix}mean_{safe_name}"] = mean_metrics[value_index]
                value_index += 1
        return aggregated

    def _write_results(self, run_dir, run_name, dataset_name, records) -> None:
        table_path = run_dir / "test_scene_metrics.txt"
        with table_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "CovDiffusion TEST Scene Metrics (per object)\n"
                f"Run: {run_name}\nDataset: {dataset_name or 'unknown'}\n"
                f"Training seed: {self.training_seed}\n"
                f"Rollout seed: {self.seed}\n"
                + "=" * 70
                + "\n"
            )
            handle.write(
                f"{'Episode':<10} {'PCD':<12} {'Jerk':<12} {'Coverage':<10} "
                f"{'InfTime_ms':<10} {'Latency_ms':<10}\n"
            )
            handle.write("-" * 70 + "\n")
            for record in records:
                handle.write(
                    f"{record['episode']:<10} {record['pcd']:<12.4f} "
                    f"{record['jerk']:<12.6f} {record['coverage']:<10.2f} "
                    f"{record['inference_time_ms']:<10.2f} "
                    f"{record['latency_ms']:<10.2f}\n"
                )

        summary = {
            "mean_pcd": float(np.mean([record["pcd"] for record in records])),
            "mean_jerk": float(np.mean([record["jerk"] for record in records])),
            "mean_coverage": float(
                np.mean([record["coverage"] for record in records])
            ),
            "n_episodes": len(records),
        }
        protocol = {
            "metrics_version": "3dcov-paper-v1",
            "pcd": "1e4 * symmetric mean nearest squared XYZ distance",
            "jerk": "mean translational third finite difference",
            "coverage": (
                "area-weighted"
                if bool(self.config.get("coverage_area_weighted", False))
                else "face-count"
            ),
            "coverage_spray_radius": float(
                self.config.get("coverage_spray_radius", 0.1)
            ),
            "condition_mode": "Pred_Cond",
            "rollout_seed": self.seed,
        }
        result = {
            "per_episode": records,
            "summary": summary,
            "protocol": protocol,
            "run_name": run_name,
            "dataset": dataset_name,
            "training_seed": self.training_seed,
            "rollout_seed": self.seed,
            "seed": self.training_seed,
        }
        (run_dir / "test_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Results: {run_dir / 'test_results.json'}")

    def run(
        self,
        policy: BasePolicy,
        dataloader=None,
        dataset=None,
        split="test",
        run_name="default_run",
        dataset_name=None,
        paper_vis_mode=False,
    ):
        del dataset, paper_vis_mode
        if dataloader is None:
            raise ValueError("A fixed-split dataloader is required.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive after split resolution.")

        self._seed_everything()
        policy.eval()
        device = policy.device
        metrics_handler = MetricSuite(config=self.config, metrics=list(METRICS))
        episode_limit = min(len(dataloader), self.eval_episodes)
        run_dir = Path(self.output_dir or ".").expanduser().resolve() / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Evaluating {episode_limit} {split} episode(s) in "
            f"{self.condition_modes} -> {run_dir}"
        )

        results_by_mode = {mode: [] for mode in CONDITION_MODES}
        selection_by_mode = {mode: [] for mode in CONDITION_MODES}
        prediction_records = []

        iterator = tqdm.tqdm(
            enumerate(dataloader),
            total=episode_limit,
            desc=f"Eval on {split}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )
        with torch.no_grad():
            for local_index, data in iterator:
                if local_index >= episode_limit:
                    break
                episode_index = _source_episode_index(data, local_index)
                batch = dict_apply(
                    data,
                    lambda value: value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value,
                )
                ground_truth = self._validate_ground_truth(batch)
                if "traj_as_pc" not in batch:
                    raise KeyError("The released evaluator requires traj_as_pc.")

                for mode in self.condition_modes:
                    prediction, inference_times, latencies = self._rollout(
                        policy, batch, ground_truth, mode
                    )
                    values = metrics_handler.compute(
                        y_pred=prediction,
                        y=ground_truth,
                        traj_as_pc=batch["traj_as_pc"],
                        batch_data=batch,
                        inference_times=inference_times,
                        latencies=latencies,
                    )
                    record = _metric_record(values, episode_index)
                    results_by_mode[mode].append(values)
                    if self.checkpoint_selection:
                        predicted_episode = prediction.detach().cpu().numpy()
                        reference_episode = (
                            batch["traj_as_pc"].detach().cpu().numpy()
                        )
                        selection_value = checkpoint_selection_pose_chamfer_distance(
                            predicted_episode[0],
                            reference_episode[0],
                        )
                        selection_by_mode[mode].append(selection_value)
                        record["selection_pose_chamfer_6d"] = selection_value
                    if mode == "Pred_Cond":
                        prediction_records.append(record)

                    if self.save_artifacts:
                        episode_dir = run_dir / f"episode_{episode_index:03d}_{mode}"
                        episode_dir.mkdir(parents=True, exist_ok=True)
                        self._save_centered_trajectory_ply(
                            prediction, episode_dir / "pred_trajectory_points.ply"
                        )
                    iterator.set_postfix_str(
                        f"Ep {episode_index} PCD ({mode}): {record['pcd']:.4f}"
                    )

        if not prediction_records:
            raise RuntimeError("Pred_Cond must be enabled to export paper metrics.")
        self._write_results(run_dir, run_name, dataset_name, prediction_records)

        pred = self._aggregate(
            results_by_mode["Pred_Cond"], metrics_handler, "Pred_Cond_"
        )
        gt = self._aggregate(results_by_mode["GT_Cond"], metrics_handler, "GT_Cond_")
        if self.checkpoint_selection:
            pred["Pred_Cond_mean_historical_pose_chamfer_distance"] = float(
                np.mean(selection_by_mode["Pred_Cond"])
            )
            if selection_by_mode["GT_Cond"]:
                gt["GT_Cond_mean_historical_pose_chamfer_distance"] = float(
                    np.mean(selection_by_mode["GT_Cond"])
                )
        print("\nAggregated Results (Prediction Conditioned):")
        metrics_handler.pprint(pred["mean_metrics"], prefix="Mean Metrics (Pred_Cond):")
        if results_by_mode["GT_Cond"]:
            print("\nAggregated Results (Ground Truth Conditioned):")
            metrics_handler.pprint(gt["mean_metrics"], prefix="Mean Metrics (GT_Cond):")
        return {"pred_cond": pred, "gt_cond": gt}
