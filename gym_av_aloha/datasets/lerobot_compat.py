"""Compatibility helpers for the LeRobot v3.0 dataset format.

LeRobot 0.3 dropped a handful of helpers that this repo relied on, and moved
per-episode statistics out of ``meta/episodes_stats.jsonl`` into ``stats/*``
columns of the episodes table. The functions here restore the small pieces we
need so the rest of the codebase does not have to track LeRobot internals.
"""

from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from lerobot.datasets.utils import EPISODES_DIR, load_nested_dataset, unflatten_dict


def get_episode_data_index(
    episodes: dict, episode_indices: list[int] | None = None
) -> dict[str, torch.Tensor]:
    """Return the frame range of each episode as ``{"from": ..., "to": ...}``.

    ``episodes`` maps an episode index to metadata carrying at least a
    ``length`` field. When ``episode_indices`` is omitted every episode is used,
    in ascending order.
    """
    selected = sorted(episodes) if episode_indices is None else list(episode_indices)
    from_indices, to_indices = [], []
    cursor = 0
    for ep_idx in selected:
        length = int(episodes[ep_idx]["length"])
        from_indices.append(cursor)
        cursor += length
        to_indices.append(cursor)
    return {
        "from": torch.tensor(from_indices, dtype=torch.long),
        "to": torch.tensor(to_indices, dtype=torch.long),
    }


def get_lerobot_episode_data_index(dataset) -> dict[str, torch.Tensor]:
    """Frame ranges of the episodes selected in a ``LeRobotDataset``.

    Replaces the ``dataset.episode_data_index`` attribute dropped in v3.0. The
    returned indices are contiguous over the *selection*, matching how
    ``LeRobotDataset`` re-indexes its frames when ``episodes=`` is given.

    Uses ``dataset_from_index``/``dataset_to_index`` rather than the ``length``
    column: the two can disagree on datasets converted from v2.1, and only the
    former matches the rows actually stored in the parquet files.
    """
    selected = list(dataset.episodes) if dataset.episodes else list(range(dataset.meta.total_episodes))
    lengths = {}
    for ep_idx in selected:
        ep = dataset.meta.episodes[ep_idx]
        lengths[ep_idx] = {"length": ep["dataset_to_index"] - ep["dataset_from_index"]}
    return get_episode_data_index(lengths, selected)


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """Verify consecutive frames are spaced by ``1/fps`` within ``tolerance_s``.

    Gaps at episode boundaries are ignored, since those are expected.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask out the diff spanning the boundary between two episodes.
    mask = np.ones(len(diffs), dtype=bool)
    mask[episode_data_index["to"][:-1] - 1] = False

    outside_tolerances = diffs[mask][~within_tolerance[mask]]
    if len(outside_tolerances) > 0:
        if raise_value_error:
            raise ValueError(
                "One or several timestamps unexpectedly violate the tolerance inside "
                "episode range. This might be due to synchronization issues during data "
                f"collection.\n{pformat(outside_tolerances)}"
            )
        return False

    return True


def load_episodes_stats(root: str | Path) -> list[dict[str, dict[str, np.ndarray]]]:
    """Load per-episode statistics from a v3.0 dataset, indexed by episode.

    ``LeRobotDatasetMetadata`` deliberately strips the ``stats/*`` columns when
    loading episodes, so we re-read the episodes table without that filter.
    """
    episodes = load_nested_dataset(Path(root) / EPISODES_DIR)
    stats_per_episode = []
    for row in episodes:
        flat = {k: v for k, v in row.items() if k.startswith("stats/")}
        stats = unflatten_dict(flat)["stats"]
        stats_per_episode.append(
            {
                key: {k: np.asarray(v) for k, v in feature_stats.items()}
                for key, feature_stats in stats.items()
            }
        )
    return stats_per_episode


def get_task_strings(meta) -> list[str]:
    """Return the task strings of a dataset, ordered by task index.

    In v3.0 ``meta.tasks`` is a DataFrame indexed by the task string with a
    ``task_index`` column, replacing the old ``{index: task}`` dict.
    """
    return list(meta.tasks.sort_values("task_index").index)
