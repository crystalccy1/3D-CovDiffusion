"""Episode-safe execution-history indexing."""

from __future__ import annotations

import numpy as np


def previous_action_index(
    buffer_start_idx: int,
    episode_start: int,
    action_valid_mask: np.ndarray,
) -> int:
    """Return a previous valid index without crossing an episode boundary.

    At the first sample of an episode, or when no earlier valid action exists
    inside that episode, the current episode's first sampled action is used.
    """

    if buffer_start_idx <= episode_start:
        return int(buffer_start_idx)
    local_valid = np.flatnonzero(
        np.asarray(action_valid_mask[episode_start:buffer_start_idx], dtype=bool)
    )
    if local_valid.size == 0:
        return int(buffer_start_idx)
    return int(episode_start + local_valid[-1])


__all__ = ["previous_action_index"]
