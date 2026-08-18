"""Shared utilities for the LongEgoRefer grounding scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EGO4D_FPS = 30  # Ego4D full-scale videos
EGOTRACKS_FPS = 5  # EgoTracks annotation frames


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load ``longegorefer.json`` (a list of occurrence records)."""
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty-printed JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def gt_interval_seconds(record: dict[str, Any]) -> tuple[float, float]:
    """Ground-truth [start, end] of an occurrence in source-video seconds."""
    return (
        record["video_start_frame_number"] / EGO4D_FPS,
        record["video_end_frame_number"] / EGO4D_FPS,
    )


def time_str_to_seconds(time_str: str) -> int:
    """Convert an 'MM:SS' or 'HH:MM:SS' string to total seconds.

    Raises:
        ValueError: If ``time_str`` does not match either format.
    """
    parts = [int(p) for p in time_str.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid time string: {time_str!r} (expected 'MM:SS' or 'HH:MM:SS')")


def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to an 'MM:SS' string."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"
