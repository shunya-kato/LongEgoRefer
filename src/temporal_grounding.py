"""Temporal grounding baseline for LongEgoRefer.

Given a long egocentric video and a referring expression (``caption``), predict
the time interval in which the target object behaves as described.

Two API backends are supported:

* ``gemini`` -- uploads the whole video via the Gemini Files API and queries the
  model directly with the video.
* ``openai`` -- samples frames (1 fps by default), embeds them as base64 JPEG
  images with their timestamps, and queries a GPT model via Chat Completions.

Usage::

    export GOOGLE_API_KEY=...  # for --backend gemini
    export OPENAI_API_KEY=...  # for --backend openai

    python src/temporal_grounding.py --backend gemini \
        --video-dir /path/to/ego4d/fps1

    python src/temporal_grounding.py --backend openai --model gpt-4o-2024-11-20 \
        --video-dir /path/to/ego4d/fps1

Videos are expected at ``{video_dir}/{video_uid}.mp4``. Results and scores are
written to ``{output_dir}/temporal_grounding_{results,score}.json``. The script
resumes from an existing results file, skipping already-successful samples
(disable with ``--no-resume``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import json
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from common import gt_interval_seconds, load_dataset, save_json, seconds_to_time_str, time_str_to_seconds
from metrics import RECALL_THRESHOLDS, temporal_metrics

Predictor = Callable[[str, str], Awaitable["TemporalGroundingResult | None"]]

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-2024-11-20"}

SYSTEM_INSTRUCTION = (
    "You are a precise temporal grounding model for videos.\n"
    "Your goal is to identify the exact start and end times of the event described by the text query.\n"
    "You must provide the following two fields:\n"
    "1. `start_time`: The timestamp (in MM:SS format) when the described event begins.\n"
    "2. `end_time`: The timestamp (in MM:SS format) when the described event concludes.\n"
    "IMPORTANT: Do not use or process any audio information from the video. "
    "Only analyze the visual content (video frames/images) to identify temporal segments. "
    "Ignore all audio tracks completely."
)


class TemporalGroundingResult(BaseModel):
    """Structured model output: the time interval grounded in the video."""

    start_time: str = Field(..., description="The start time of the event described in the text, formatted as 'MM:SS'.")
    end_time: str = Field(..., description="The end time of the event described in the text, formatted as 'MM:SS'.")


def video_duration_seconds(video_path: str) -> float:
    video = cv2.VideoCapture(video_path)
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        return video.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0.0
    finally:
        video.release()


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------


def build_gemini_predictor(model: str) -> Predictor:
    from google import genai
    from google.genai import types

    client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions()))

    async def predict(video_path: str, expression: str) -> TemporalGroundingResult | None:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TemporalGroundingResult,
            system_instruction=SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_budget=32768 if "pro" in model else 24576),
        )
        # At the default media resolution, videos of an hour or more exceed the context window.
        if video_duration_seconds(video_path) >= 3600:
            config.media_resolution = types.MediaResolution.MEDIA_RESOLUTION_LOW

        video_file = await client.aio.files.upload(file=video_path)
        try:
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(10)
                video_file = await client.aio.files.get(name=video_file.name)
            response = await client.aio.models.generate_content(model=model, contents=[video_file, expression], config=config)
            return response.parsed
        finally:
            await client.aio.files.delete(name=video_file.name)

    return predict


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

# Stay safely under the API's 50 MB request-size limit.
MAX_IMAGE_PAYLOAD_BYTES = 43 * 1024 * 1024


def extract_frames(video_path: str, target_fps: float, max_dim: int, max_frames: int) -> list[tuple[str, str]]:
    """Sample video frames as ('MM:SS' timestamp, base64 JPEG) pairs."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return []
    try:
        original_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if original_fps <= 0:
            return []
        interval = max(1, int(original_fps / target_fps))
        if total_frames / interval > max_frames:
            interval = max(1, total_frames // max_frames)

        frames: list[tuple[str, str]] = []
        frame_index = 0
        while len(frames) < max_frames:
            ok, frame = video.read()
            if not ok:
                break
            if frame_index % interval == 0:
                height, width = frame.shape[:2]
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                timestamp = seconds_to_time_str(frame_index / original_fps)
                frames.append((timestamp, base64.b64encode(buffer).decode()))
            frame_index += 1
        return frames
    finally:
        video.release()


def subsample_to_payload_limit(frames: list[tuple[str, str]], max_bytes: int = MAX_IMAGE_PAYLOAD_BYTES) -> list[tuple[str, str]]:
    """Evenly subsample frames so the total base64 payload stays under ``max_bytes``."""
    total_bytes = sum(len(b64) for _, b64 in frames)
    if total_bytes <= max_bytes:
        return frames
    target_count = max(1, int(len(frames) * max_bytes / total_bytes))
    step = len(frames) / target_count
    return [frames[int(i * step)] for i in range(target_count)]


def build_openai_predictor(
    model: str,
    frame_fps: float,
    max_dim: int,
    max_frames: int,
    loop: asyncio.AbstractEventLoop,
    pool: ProcessPoolExecutor,
) -> Predictor:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(max_retries=5)

    async def predict(video_path: str, expression: str) -> TemporalGroundingResult | None:
        # Frame extraction is CPU-bound, so run it in a separate process.
        frames = await loop.run_in_executor(pool, functools.partial(extract_frames, video_path, frame_fps, max_dim, max_frames))
        if not frames:
            return None
        frames = subsample_to_payload_limit(frames)

        content: list[dict[str, Any]] = [{"type": "text", "text": f"Query expression: {expression}\nHere are the sampled frames from the video:"}]
        for timestamp, b64 in frames:
            content.append({"type": "text", "text": f"Time: {timestamp}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        response = await client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": content},
            ],
            response_format=TemporalGroundingResult,
        )
        return response.choices[0].message.parsed

    return predict


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def predict_all(predict: Predictor, records: list[dict[str, Any]], video_dir: Path, concurrency: int) -> list[TemporalGroundingResult | None]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(record: dict[str, Any]) -> TemporalGroundingResult | None:
        video_path = str(video_dir / f"{record['video_uid']}.mp4")
        async with semaphore:
            try:
                return await predict(video_path, record["caption"])
            except Exception as e:  # noqa: BLE001 -- one failed sample must not kill the batch
                print(f"[{record['id']}] {type(e).__name__}: {e}")
                return None

    return await tqdm_asyncio.gather(*(run_one(record) for record in records))


def evaluate(record: dict[str, Any], prediction: TemporalGroundingResult | None) -> dict[str, Any]:
    gt_start, gt_end = gt_interval_seconds(record)
    result = {"id": record["id"], "gt_start_time": gt_start, "gt_end_time": gt_end}
    try:
        pred_start = time_str_to_seconds(prediction.start_time)
        pred_end = time_str_to_seconds(prediction.end_time)
        metrics = temporal_metrics(gt_start, gt_end, pred_start, pred_end)
    except (AttributeError, ValueError):  # no prediction, or malformed timestamps
        failed = {f"recall@{threshold}": 0 for threshold in RECALL_THRESHOLDS}
        return {**result, "start_time": -1, "end_time": -1, "iou": 0.0, **failed}
    return {**result, "start_time": pred_start, "end_time": pred_end, **metrics}


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    score = {"mIoU": sum(r["iou"] for r in results) / len(results)}
    for threshold in RECALL_THRESHOLDS:
        key = f"recall@{threshold}"
        score[key] = sum(r[key] for r in results) / len(results)
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["gemini", "openai"], required=True)
    parser.add_argument("--model", default=None, help="API model name (default depends on --backend)")
    parser.add_argument("--dataset", default="dataset/longegorefer.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing {video_uid}.mp4")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/temporal_grounding/{model}")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing results file")
    # Frame sampling options (openai backend only).
    parser.add_argument("--frame-fps", type=float, default=1.0, help="Frame sampling rate sent to the model")
    parser.add_argument("--max-frames", type=int, default=500, help="Max number of frames sent to the model")
    parser.add_argument("--max-dim", type=int, default=512, help="Max frame edge length in pixels")
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODELS[args.backend]
    if args.output_dir is None:
        args.output_dir = Path("outputs/temporal_grounding") / args.model
    return args


async def main_async(args: argparse.Namespace) -> None:
    results_path = args.output_dir / "temporal_grounding_results.json"
    score_path = args.output_dir / "temporal_grounding_score.json"

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    completed: list[dict[str, Any]] = []
    if results_path.exists() and not args.no_resume:
        with open(results_path) as f:
            completed = [r for r in json.load(f) if r.get("start_time", -1) != -1]
        print(f"Resuming: skipping {len(completed)} completed samples from {results_path}")
    completed_ids = {r["id"] for r in completed}
    records = [r for r in dataset if r["id"] not in completed_ids]

    print(f"Running {len(records)} samples (backend={args.backend}, model={args.model})")
    if args.backend == "gemini":
        predict = build_gemini_predictor(args.model)
        predictions = await predict_all(predict, records, args.video_dir, args.concurrency)
    else:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            predict = build_openai_predictor(args.model, args.frame_fps, args.max_dim, args.max_frames, loop, pool)
            predictions = await predict_all(predict, records, args.video_dir, args.concurrency)

    results = completed + [evaluate(record, prediction) for record, prediction in zip(records, predictions, strict=True)]
    if not results:
        print("No results to evaluate.")
        return

    score = aggregate(results)
    for name, value in score.items():
        print(f"{name}: {value:.4f}")

    save_json(results, results_path)
    save_json(score, score_path)
    print(f"Saved results to {args.output_dir}")


def main() -> None:
    load_dotenv()
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
