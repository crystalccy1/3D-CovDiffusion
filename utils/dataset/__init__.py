"""Dataset adapters used by 3D-CovDiffusion."""

__all__ = [
    "CanonicalEvaluationDataset",
    "CovDiffusionRolloutDataset",
    "ProcessedCovDiffusionDataset",
]


def __getattr__(name):
    """Load dataset adapters only when a caller explicitly requests them."""

    if name == "CanonicalEvaluationDataset":
        from .canonical_evaluation_dataset import CanonicalEvaluationDataset

        return CanonicalEvaluationDataset

    if name == "CovDiffusionRolloutDataset":
        from .covdiffusion_rollout_dataset import CovDiffusionRolloutDataset

        return CovDiffusionRolloutDataset
    if name == "ProcessedCovDiffusionDataset":
        from .processed_covdiffusion_dataset import ProcessedCovDiffusionDataset

        return ProcessedCovDiffusionDataset
    raise AttributeError(name)
