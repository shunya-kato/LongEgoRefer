"""Spatial grounding baseline for LongEgoRefer (Grounding DINO + SAM2).

Localizes the target object within a predicted (or ground-truth) time interval:

1. Cut the interval out of the source Ego4D video, capped at ``--max-interval``
   seconds around its center.
2. Detect the target object on the middle frame of the clip with Grounding DINO,
   using the referring expression as the text prompt.
3. Track the detected box across the clip with SAM2 (bidirectional mask
   propagation) and convert the masks to per-frame bounding boxes.

Evaluated with vIoU / Recall@{0.1, 0.3, 0.5} / STIoU / IoU+n against the
``bbox`` annotations.

Usage::

    # On top of predicted intervals from temporal_grounding.py:
    python src/spatial_grounding.py \
        --temporal-results outputs/temporal_grounding/gemini-2.5-flash/temporal_grounding_results.json \
        --video-dir /path/to/ego4d/full_scale \
        --output-dir outputs/spatial_grounding/gemini-2.5-flash_grounded-sam2

    # Oracle: use the ground-truth intervals instead.
    python src/spatial_grounding.py --interval gt \
        --video-dir /path/to/ego4d/full_scale \
        --output-dir outputs/spatial_grounding/oracle_grounded-sam2
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2VideoModel, Sam2VideoProcessor
from transformers.video_utils import load_video

from common import EGO4D_FPS, EGOTRACKS_FPS, load_dataset, save_json
from metrics import RECALL_THRESHOLDS, Box, Tube, spatial_metrics


@contextmanager
def disable_tqdm() -> Iterator[None]:
    """Temporarily silence the progress bars SAM2 spawns internally."""
    original_init = tqdm.__init__

    def silenced_init(self: tqdm, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    tqdm.__init__ = silenced_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        tqdm.__init__ = original_init  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Interval handling
# ---------------------------------------------------------------------------


def cap_interval(start_s: float, end_s: float, max_duration_s: float) -> tuple[float, float]:
    """Shrink [start_s, end_s] symmetrically around its center to ``max_duration_s``."""
    if end_s - start_s <= max_duration_s:
        return start_s, end_s
    center = (start_s + end_s) / 2
    return center - max_duration_s / 2, center + max_duration_s / 2


def align_to_annotation_grid(clip_start_s: float, gt_bboxes: list[dict[str, Any]]) -> tuple[float, int]:
    """Snap the clip start to the EgoTracks annotation grid.

    GT boxes lie on every (EGO4D_FPS // EGOTRACKS_FPS)-th video frame; without
    this alignment, the fps-sampled clip frames would never coincide with the
    GT frames and the metrics would be systematically zero.
    """
    annotation_step = EGO4D_FPS // EGOTRACKS_FPS
    gt_offset = gt_bboxes[0]["video_frame_number"] % annotation_step
    clip_start_frame = int(clip_start_s * EGO4D_FPS)
    clip_start_frame -= (clip_start_frame - gt_offset) % annotation_step
    if clip_start_frame < 0:
        clip_start_frame += annotation_step
    return clip_start_frame / EGO4D_FPS, clip_start_frame


# ---------------------------------------------------------------------------
# Detection & tracking
# ---------------------------------------------------------------------------


def detect_box(
    clip: VideoFileClip,
    caption: str,
    model: AutoModelForZeroShotObjectDetection,
    processor: AutoProcessor,
) -> tuple[int, Box]:
    """Detect the described object on the middle frame of the clip with Grounding DINO.

    Returns:
        The annotated frame time (seconds from the clip start) and the
        highest-scoring box.
    """
    frame_time_s = (int(clip.duration) + 1) // 2
    image = Image.fromarray(clip.get_frame(frame_time_s))
    inputs = processor(images=image, text=[caption], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0, text_threshold=0, target_sizes=[image.size[::-1]])[0]
    best = result["scores"].argmax().item()
    return frame_time_s, result["boxes"][best].tolist()


def mask_to_box(mask_logits: torch.Tensor) -> Box:
    """Convert a single-object mask (logits) to an [x_min, y_min, x_max, y_max] box."""
    mask = (mask_logits.squeeze() > 0.0).cpu().numpy()
    if not mask.any():
        return [0, 0, 0, 0]
    rows, cols = np.where(mask)
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


def track_with_sam2(
    clip_path: str,
    clip_start_frame: int,
    annotated_time_s: int,
    box: Box,
    fps: int,
    model: Sam2VideoModel,
    processor: Sam2VideoProcessor,
    device: str,
) -> list[dict[str, Any]]:
    """Track ``box`` across the clip with SAM2.

    Returns per-frame boxes as {video_frame_number, x, y, width, height} dicts
    in source-video frame numbers.
    """
    video_frames, _ = load_video(clip_path, fps=fps)
    session = processor.init_video_session(video=video_frames, inference_device=device, dtype=torch.bfloat16)

    annotated_frame_idx = annotated_time_s * fps
    processor.add_inputs_to_inference_session(inference_session=session, frame_idx=annotated_frame_idx, obj_ids=1, input_boxes=[[box]])
    # Segment the annotated frame first, then propagate in both directions.
    model(inference_session=session, frame_idx=annotated_frame_idx)

    frame_step = EGO4D_FPS // fps
    pred_boxes: dict[int, dict[str, Any]] = {}
    with disable_tqdm():
        for reverse in (False, True):
            for output in model.propagate_in_video_iterator(session, start_frame_idx=annotated_frame_idx, reverse=reverse):
                masks = processor.post_process_masks([output.pred_masks], original_sizes=[[session.video_height, session.video_width]], binarize=False)[0]
                x_min, y_min, x_max, y_max = mask_to_box(masks)
                video_frame_number = output.frame_idx * frame_step + clip_start_frame
                pred_boxes.setdefault(
                    video_frame_number,
                    {
                        "video_frame_number": video_frame_number,
                        "x": x_min,
                        "y": y_min,
                        "width": x_max - x_min,
                        "height": y_max - y_min,
                    },
                )
    return [pred_boxes[frame] for frame in sorted(pred_boxes)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def build_gt_tube(gt_bboxes: list[dict[str, Any]], clip_start_frame: int, frame_step: int) -> Tube:
    """GT boxes that fall on the fps-sampling grid anchored at ``clip_start_frame``."""
    return {
        b["video_frame_number"]: [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
        for b in gt_bboxes
        if (b["video_frame_number"] - clip_start_frame) % frame_step == 0
    }


def empty_result(record: dict[str, Any], n_frames: int, frame_step: int) -> dict[str, Any]:
    """Result for a failed or degenerate sample: score an empty prediction tube.

    The GT sampling grid is anchored at the first GT box.
    """
    gt_bboxes = record["bbox"]
    gt_tube = build_gt_tube(gt_bboxes, gt_bboxes[0]["video_frame_number"], frame_step)
    return {"id": record["id"], "gt_bbox": gt_bboxes, "pred_bbox": [], **spatial_metrics({}, gt_tube, n_frames)}


def build_pred_tube(pred_bboxes: list[dict[str, Any]]) -> Tube:
    return {b["video_frame_number"]: [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]] for b in pred_bboxes if b["width"] > 0 and b["height"] > 0}


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results)
    score = {"mvIoU": sum(r["viou"] for r in results) / n}
    for threshold in RECALL_THRESHOLDS:
        key = f"recall@{threshold}"
        score[key] = sum(r[key] for r in results) / n
    score["mSTIoU"] = sum(r["stiou"] for r in results) / n
    score["mIoU+n"] = sum(r["iou+n"] for r in results) / n
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--interval", choices=["pred", "gt"], default="pred", help="Use predicted intervals (--temporal-results) or ground-truth intervals (oracle)"
    )
    parser.add_argument(
        "--temporal-results", type=Path, default=None, help="temporal_grounding_results.json from temporal_grounding.py (required for --interval pred)"
    )
    parser.add_argument("--dataset", default="dataset/longegorefer.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing full-scale Ego4D videos ({video_uid}.mp4)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=None, help="Where interval clips are written (default: {output_dir}/clips)")
    parser.add_argument("--fps", type=int, default=1, help="Frame sampling rate for SAM2 tracking")
    parser.add_argument("--max-interval", type=float, default=300, help="Cap on the tracked interval length in seconds")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-base-plus")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    args = parser.parse_args()
    if args.interval == "pred" and args.temporal_results is None:
        parser.error("--temporal-results is required with --interval pred")
    if args.clip_dir is None:
        args.clip_dir = args.output_dir / "clips"
    return args


def load_intervals(args: argparse.Namespace, dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map sample id -> {start_time, end_time, iou} in source-video seconds."""
    if args.interval == "gt":
        return {
            record["id"]: {
                "start_time": record["video_start_frame_number"] / EGO4D_FPS,
                "end_time": record["video_end_frame_number"] / EGO4D_FPS,
                "iou": 1.0,
            }
            for record in dataset
        }
    with open(args.temporal_results) as f:
        return {result["id"]: result for result in json.load(f)}


def main() -> None:
    args = parse_args()

    sam2_model = Sam2VideoModel.from_pretrained(args.sam2_model).to(args.device, dtype=torch.bfloat16)
    sam2_processor = Sam2VideoProcessor.from_pretrained(args.sam2_model)
    dino_processor = AutoProcessor.from_pretrained(args.dino_model)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model).to(args.device)

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    intervals = load_intervals(args, dataset)
    args.clip_dir.mkdir(parents=True, exist_ok=True)

    frame_step = EGO4D_FPS // args.fps
    results: list[dict[str, Any]] = []
    for record in tqdm(dataset):
        interval = intervals.get(record["id"])
        if interval is None:
            continue
        gt_bboxes = record["bbox"]

        video = None
        n_frames = 0
        try:
            video = VideoFileClip(str(args.video_dir / f"{record['video_uid']}.mp4"))
            n_frames = video.n_frames // frame_step  # fps-sampled frames in the full video
            start_s = min(interval["start_time"], video.duration)
            end_s = min(interval["end_time"], video.duration)

            # A degenerate interval, or a predicted interval that missed the GT
            # entirely (temporal IoU 0), can only score a vIoU of 0.
            if end_s - start_s < 1 or interval.get("iou", 1.0) == 0:
                results.append(empty_result(record, n_frames, frame_step))
                continue

            start_s, end_s = cap_interval(start_s, end_s, args.max_interval)
            start_s, clip_start_frame = align_to_annotation_grid(start_s, gt_bboxes)

            # +1 s so that the frame at end_s itself is included in the subclip.
            clip = video.subclipped(start_s, min(end_s + 1, video.duration))
            clip_path = args.clip_dir / f"{record['id']}.mp4"
            clip.write_videofile(str(clip_path), audio=False, logger=None)

            annotated_time_s, box = detect_box(clip, record["caption"], dino_model, dino_processor)
            pred_bboxes = track_with_sam2(
                str(clip_path),
                clip_start_frame,
                annotated_time_s,
                box,
                args.fps,
                sam2_model,
                sam2_processor,
                args.device,
            )
            clip.close()

            gt_tube = build_gt_tube(gt_bboxes, clip_start_frame, frame_step)
            metrics = spatial_metrics(build_pred_tube(pred_bboxes), gt_tube, n_frames)
            results.append({"id": record["id"], "gt_bbox": gt_bboxes, "pred_bbox": pred_bboxes, **metrics})
        except Exception as e:  # noqa: BLE001 -- one failed sample must not kill the run
            print(f"[{record['id']}] {type(e).__name__}: {e}")
            results.append(empty_result(record, n_frames, frame_step))
        finally:
            if video is not None:
                video.close()

    if not results:
        print("No results to evaluate.")
        return

    score = aggregate(results)
    for name, value in score.items():
        print(f"{name}: {value:.4f}")

    save_json(results, args.output_dir / "spatial_grounding_results.json")
    save_json(score, args.output_dir / "spatial_grounding_score.json")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
