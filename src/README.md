# `src/` — Grounding Baselines

Baselines that take a referring expression (`caption`) from
[`dataset/longegorefer.json`](../dataset/longegorefer.json) and localize the
target object in long egocentric videos.

| File | Description |
|---|---|
| `temporal_grounding.py` | Predict the time interval of the described occurrence (Gemini or OpenAI backend). |
| `spatial_grounding.py` | Predict the object's bounding-box track within the interval (Grounding DINO + SAM2). |
| `change_video_fps.py` | Preprocessing: re-encode Ego4D videos at a lower fps. |
| `metrics.py` | Temporal IoU / vIoU / STIoU / IoU+n and Recall@{0.1, 0.3, 0.5}. |
| `common.py` | Dataset loading, time parsing, shared constants. |

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python src/temporal_grounding.py --help
```

API keys are read from the environment (or a `.env` file at the repo root):

- `GOOGLE_API_KEY` — `temporal_grounding.py --backend gemini`
- `OPENAI_API_KEY` — `temporal_grounding.py --backend openai`

Videos are **not** included in this repository; download Ego4D full-length
videos yourself and point `--video-dir` at a directory laid out as
`{video_uid}.mp4`. The spatial script needs the full-scale (30 fps) videos;
the temporal script works on low-fps re-encodes (see below).

## Preprocessing: low-fps videos

The Gemini backend uploads whole videos, so re-encode them at a low frame rate
(e.g. 1 fps) to stay under the Files API size limit; the OpenAI backend also
benefits from the smaller files. Requires the `ffmpeg` command
(`--codec h264_nvenc` encodes on an NVIDIA GPU):

```bash
python src/change_video_fps.py 1 \
    --video-dir /path/to/ego4d_data/v1/full_scale \
    --output-dir /path/to/ego4d/fps1
```

Only videos referenced by the dataset are converted; audio is dropped and
existing outputs are skipped, so the script can be re-run to resume.

## Temporal grounding

```bash
python src/temporal_grounding.py --backend gemini --model gemini-2.5-flash \
    --video-dir /path/to/ego4d/fps1

python src/temporal_grounding.py --backend openai --model gpt-4o-2024-11-20 \
    --video-dir /path/to/ego4d/fps1
```

Writes `temporal_grounding_results.json` (per-sample predictions and metrics)
and `temporal_grounding_score.json` (mIoU, Recall@{0.1, 0.3, 0.5}) under
`--output-dir` (default `outputs/temporal_grounding/{model}`). Reruns resume
from the results file, retrying only failed samples (`--no-resume` to restart).

The ground-truth interval is `video_start_frame_number / 30` to
`video_end_frame_number / 30` seconds in the source video.

- Gemini: uploads the whole video via the Files API; long videos (≥ 1 h) are
  processed at low media resolution to fit the context window.
- OpenAI: samples frames (`--frame-fps`, at most `--max-frames`, resized to
  `--max-dim`) and sends them with their timestamps as base64 JPEG images.

## Spatial grounding

```bash
# On top of predicted intervals:
python src/spatial_grounding.py \
    --temporal-results outputs/temporal_grounding/gemini-2.5-flash/temporal_grounding_results.json \
    --video-dir /path/to/ego4d/full_scale \
    --output-dir outputs/spatial_grounding/gemini-2.5-flash_grounded-sam2

# Oracle (ground-truth intervals):
python src/spatial_grounding.py --interval gt \
    --video-dir /path/to/ego4d/full_scale \
    --output-dir outputs/spatial_grounding/oracle_grounded-sam2
```

Pipeline per sample: cut the interval (capped at `--max-interval` seconds
around its center) → detect the object on the middle frame with Grounding DINO
prompted by the caption → track it bidirectionally with SAM2 → convert masks
to boxes. Writes `spatial_grounding_results.json` and
`spatial_grounding_score.json` (mvIoU, Recall@{0.1, 0.3, 0.5}, mSTIoU, mIoU+n).

Predicted intervals with temporal IoU 0 are scored as vIoU 0 without running
detection/tracking. The clip start is snapped to the EgoTracks annotation grid
(every 6th video frame) so sampled frames coincide with GT boxes.
