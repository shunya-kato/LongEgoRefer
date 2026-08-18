"""Evaluation metrics for temporal and spatial grounding."""

from __future__ import annotations

Box = list[float]  # [x_min, y_min, x_max, y_max]
Tube = dict[int, Box]  # {video_frame_number: [x_min, y_min, x_max, y_max]}

RECALL_THRESHOLDS = (0.1, 0.3, 0.5)


def _recalls(iou: float) -> dict[str, int]:
    return {f"recall@{threshold}": int(iou >= threshold) for threshold in RECALL_THRESHOLDS}


# ---------------------------------------------------------------------------
# Temporal grounding
# ---------------------------------------------------------------------------


def temporal_iou(gt_start: float, gt_end: float, pred_start: float, pred_end: float) -> float:
    """IoU between two time intervals in seconds.

    Raises:
        ValueError: If either interval ends before it starts.
    """
    if gt_end < gt_start or pred_end < pred_start:
        raise ValueError("Interval end precedes its start.")
    intersection = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    return intersection / union if union > 0 else 0.0


def temporal_metrics(gt_start: float, gt_end: float, pred_start: float, pred_end: float) -> dict[str, float]:
    """Temporal IoU plus Recall@{0.1, 0.3, 0.5} for a single prediction."""
    iou = temporal_iou(gt_start, gt_end, pred_start, pred_end)
    return {"iou": iou, **_recalls(iou)}


# ---------------------------------------------------------------------------
# Spatial (spatio-temporal) grounding
# ---------------------------------------------------------------------------


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(box_a: Box, box_b: Box) -> float:
    width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return max(0.0, width) * max(0.0, height)


def box_iou(box_a: Box, box_b: Box) -> float:
    """IoU between two [x_min, y_min, x_max, y_max] boxes."""
    intersection = _intersection_area(box_a, box_b)
    union = _box_area(box_a) + _box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def tube_viou(pred_tube: Tube, gt_tube: Tube) -> float:
    """vIoU: mean per-frame IoU over the union of annotated frames.

    Frames present in only one of the tubes contribute an IoU of 0.
    """
    union_frames = set(pred_tube) | set(gt_tube)
    if not union_frames:
        return 0.0
    common_frames = set(pred_tube) & set(gt_tube)
    total = sum(box_iou(pred_tube[frame], gt_tube[frame]) for frame in common_frames)
    return total / len(union_frames)


def tube_stiou(pred_tube: Tube, gt_tube: Tube) -> float:
    """STIoU: sum of per-frame intersection areas over sum of per-frame union areas."""
    total_intersection = 0.0
    total_union = 0.0
    for frame in set(pred_tube) | set(gt_tube):
        pred_box = pred_tube.get(frame)
        gt_box = gt_tube.get(frame)
        intersection = _intersection_area(pred_box, gt_box) if pred_box and gt_box else 0.0
        pred_area = _box_area(pred_box) if pred_box else 0.0
        gt_area = _box_area(gt_box) if gt_box else 0.0
        total_intersection += intersection
        total_union += pred_area + gt_area - intersection
    return total_intersection / total_union if total_union > 0 else 0.0


def tube_iou_plus_n(pred_tube: Tube, gt_tube: Tube, n_frames: int) -> float:
    """IoU+n: mean per-frame IoU over all ``n_frames`` sampled frames of the video.

    Frames absent from both tubes (true negatives) count as IoU 1, so the metric
    also rewards correctly predicting that the object is not visible.
    """
    if n_frames <= 0:
        return 0.0
    annotated_frames = set(pred_tube) | set(gt_tube)
    total = float(max(0, n_frames - len(annotated_frames)))  # true negatives
    for frame in annotated_frames:
        pred_box = pred_tube.get(frame)
        gt_box = gt_tube.get(frame)
        if pred_box and gt_box:
            total += box_iou(pred_box, gt_box)
    return total / n_frames


def spatial_metrics(pred_tube: Tube, gt_tube: Tube, n_frames: int) -> dict[str, float]:
    """vIoU, Recall@{0.1, 0.3, 0.5}, STIoU, and IoU+n for a single prediction."""
    viou = tube_viou(pred_tube, gt_tube)
    return {
        "viou": viou,
        **_recalls(viou),
        "stiou": tube_stiou(pred_tube, gt_tube),
        "iou+n": tube_iou_plus_n(pred_tube, gt_tube, n_frames),
    }
