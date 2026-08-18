"""Re-encode Ego4D videos at a lower frame rate (preprocessing).

The temporal grounding baselines work on downsampled videos: the Gemini backend
uploads whole videos, so a low frame rate (e.g. 1 fps) is needed to keep them
under the Files API size limit, and the OpenAI backend samples frames faster
from small files.

Only videos referenced by the dataset are converted; audio is dropped. Existing
outputs are skipped, so the script can be re-run to resume.

Usage::

    python src/change_video_fps.py 1 \
        --video-dir /path/to/ego4d_data/v1/full_scale \
        --output-dir /path/to/ego4d/fps1

Requires the ``ffmpeg`` command. Pass ``--codec h264_nvenc`` to encode on an
NVIDIA GPU instead of libx264.
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from common import load_dataset


def change_fps(input_path: Path, output_path: Path, fps: int, codec: str, crf: int) -> bool:
    """Re-encode one video at ``fps`` without audio. Returns True on success."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-r",
        str(fps),
        "-an",  # drop audio
        "-vcodec",
        codec,
        "-crf",
        str(crf),
        "-preset",
        "fast",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # failure handled below
    if result.returncode != 0:
        print(f"Error processing {input_path.name}: {result.stderr.strip()}")
        output_path.unlink(missing_ok=True)  # remove partial output so a re-run retries it
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fps", type=int, help="Target frame rate")
    parser.add_argument("--dataset", default="dataset/longegorefer.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing full-scale Ego4D videos ({video_uid}.mp4)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where re-encoded videos are written")
    parser.add_argument("--codec", default="libx264", help="ffmpeg video codec (e.g. libx264, h264_nvenc)")
    parser.add_argument("--crf", type=int, default=30, help="Quality/size trade-off (higher = smaller)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent ffmpeg processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_uids = sorted({record["video_uid"] for record in load_dataset(args.dataset)})

    tasks: list[tuple[Path, Path]] = []
    missing = 0
    for uid in video_uids:
        input_path = args.video_dir / f"{uid}.mp4"
        output_path = args.output_dir / f"{uid}.mp4"
        if output_path.exists():
            continue
        if not input_path.exists():
            missing += 1
            continue
        tasks.append((input_path, output_path))

    if missing:
        print(f"Warning: {missing}/{len(video_uids)} source videos not found in {args.video_dir}")
    if not tasks:
        print("Nothing to convert.")
        return

    print(f"Converting {len(tasks)} videos to {args.fps} fps with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(change_fps, inp, out, args.fps, args.codec, args.crf) for inp, out in tasks]
        succeeded = sum(future.result() for future in tqdm(futures))
    print(f"Done: {succeeded}/{len(tasks)} videos converted.")


if __name__ == "__main__":
    main()
