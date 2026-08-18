"""Streamlit visualizer for the LongEgoRefer dataset.

Pick a sample ID and inspect it: the referring expression and metadata, the
occurrence interval rendered as a clip with the ground-truth bounding boxes
drawn, and the full source video.

Usage::

    uv run streamlit run src/visualizer.py -- \
        --video-dir /path/to/ego4d_data/v1/full_scale

Videos are expected at ``{video_dir}/{video_uid}.mp4`` (full-scale, 30 fps).
Rendered clips are cached under ``--clip-cache-dir`` and reused on the next
visit. Requires the ``ffmpeg`` command (the HTML5 player needs H.264).
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from common import EGO4D_FPS, EGOTRACKS_FPS, gt_interval_seconds, load_dataset, seconds_to_time_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/longegorefer.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing full-scale Ego4D videos ({video_uid}.mp4)")
    parser.add_argument("--clip-cache-dir", type=Path, default=Path("outputs/visualizer"), help="Where rendered bbox clips are cached")
    return parser.parse_args()


@st.cache_data
def get_dataset(path: str) -> list[dict[str, Any]]:
    return load_dataset(path)


def render_clip_with_bboxes(video_path: Path, record: dict[str, Any], output_path: Path) -> None:
    """Write the occurrence interval as a short clip with GT boxes drawn.

    Only the annotated (5 fps) frames are written, so every output frame shows
    a box; the clip plays at the annotation frame rate.
    """
    bbox_map = {b["video_frame_number"]: (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]) for b in record["bbox"]}
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = record["video_start_frame_number"]
        end_frame = record["video_end_frame_number"]
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = str(Path(tmp_dir) / "raw.mp4")
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), EGOTRACKS_FPS, (width, height))
            frames_written = 0
            for frame_number in range(start_frame, end_frame + 1):
                ok, frame = video.read()
                if not ok:
                    break
                box = bbox_map.get(frame_number)
                if box is None:
                    continue
                x_min, y_min, x_max, y_max = (int(v) for v in box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
                writer.write(frame)
                frames_written += 1
            writer.release()
            if frames_written == 0:
                raise RuntimeError("No annotated frames found in the occurrence interval.")

            # Streamlit uses the HTML5 video player, which does not support
            # OpenCV's default mp4 codec; re-encode with H.264.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
                check=True,
            )
    finally:
        video.release()


def render_metadata(record: dict[str, Any], gt_start: float, gt_end: float) -> None:
    columns = st.columns(4)
    columns[0].metric("Object", record["object_title"])
    columns[1].metric("Interval", f"{seconds_to_time_str(gt_start)} - {seconds_to_time_str(gt_end)}")
    columns[2].metric("Interval length", f"{gt_end - gt_start:.1f} s")
    columns[3].metric("Video duration", f"{record['video_duration'] / 60:.1f} min")

    flags = [name for name in ("is_active", "is_moving", "is_location_changed", "is_transformed", "is_recognizable") if record[name]]
    st.markdown(f"**Flags:** {', '.join(f'`{name}`' for name in flags)}")
    if record["interaction"]:
        st.markdown(f"**Interactions:** {', '.join(f'`{phrase}`' for phrase in record['interaction'])}")
    st.caption(
        f"video_uid: `{record['video_uid']}` / clip_uid: `{record['clip_uid']}` / object_id: `{record['object_id']}` / GT boxes: {len(record['bbox'])} frames"
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="LongEgoRefer Visualizer", layout="wide")
    st.title("LongEgoRefer Visualizer")

    dataset = get_dataset(args.dataset)
    records_by_id = {record["id"]: record for record in dataset}

    with st.sidebar:
        query = st.text_input("Filter by id / object / caption").strip().lower()
        ids = [
            record["id"]
            for record in dataset
            if not query or query in record["id"].lower() or query in record["object_title"].lower() or query in record["caption"].lower()
        ]
        st.caption(f"{len(ids)} / {len(dataset)} samples")
        if not ids:
            st.warning("No samples match the filter.")
            st.stop()
        selected_id = st.selectbox("Sample ID", ids, format_func=lambda i: f"{i} ({records_by_id[i]['object_title']})")

    record = records_by_id[selected_id]
    gt_start, gt_end = gt_interval_seconds(record)

    st.subheader("Referring expression")
    st.info(record["caption"])
    render_metadata(record, gt_start, gt_end)

    video_path = args.video_dir / f"{record['video_uid']}.mp4"
    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        st.stop()

    clip_column, full_column = st.columns(2)

    with clip_column:
        st.subheader("Occurrence clip with GT boxes")
        st.caption(f"Annotated frames only, played at {EGOTRACKS_FPS} fps")
        clip_path = args.clip_cache_dir / f"{record['id']}.mp4"
        try:
            if not clip_path.exists():
                with st.spinner("Rendering clip..."):
                    render_clip_with_bboxes(video_path, record, clip_path)
            st.video(str(clip_path))
        except (RuntimeError, subprocess.CalledProcessError) as e:
            st.error(f"Failed to render the clip: {e}")

    with full_column:
        st.subheader("Full video")
        size_gb = video_path.stat().st_size / 1024**3
        if st.checkbox(f"Load full video ({size_gb:.2f} GB)", value=size_gb < 0.5):
            st.video(str(video_path), start_time=int(gt_start))
            st.caption(f"Playback starts at the occurrence ({seconds_to_time_str(gt_start)}); frame {record['video_start_frame_number']} at {EGO4D_FPS} fps.")


if __name__ == "__main__":
    main()
