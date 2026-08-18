# LongEgoRefer: A Benchmark for Long-Form Egocentric Video Referring Expression Comprehension

[![arXiv](https://img.shields.io/badge/arXiv-2607.02096-b31b1b.svg)](https://arxiv.org/abs/2607.02096)

Official repository for **LongEgoRefer** (ECCV 2026).

LongEgoRefer is a benchmark for Video Referring Expression Comprehension (Video REC) in **long-form egocentric videos**: given a natural-language referring expression describing the state changes and interactions of a target object, localize **when** the described occurrence happens (temporal grounding) and **where** the object appears (spatial grounding). Built from [Ego4D](https://ego4d-data.org/) / [EgoTracks](https://github.com/EGO4D/episodic-memory), it contains **1,498 referring expressions** over videos with an average duration of **45 minutes**, characterized by extreme target sparsity, detailed linguistic descriptions, and complex human-object interactions.

## Repository structure

- [`dataset/`](dataset/) — the benchmark annotations (`longegorefer.json`) and their format documentation ([dataset/README.md](dataset/README.md))
- [`src/`](src/) — temporal / spatial grounding baselines, evaluation metrics, preprocessing, and a dataset visualizer ([src/README.md](src/README.md))

Videos are **not** included; download them from [Ego4D](https://ego4d-data.org/) following its terms of use.

## Quick start

```bash
uv sync

# Temporal grounding (Gemini or OpenAI backend)
uv run python src/temporal_grounding.py --backend gemini --video-dir /path/to/ego4d/fps1

# Spatial grounding (Grounding DINO + SAM2)
uv run python src/spatial_grounding.py \
    --temporal-results outputs/temporal_grounding/gemini-2.5-flash/temporal_grounding_results.json \
    --video-dir /path/to/ego4d/full_scale \
    --output-dir outputs/spatial_grounding/gemini-2.5-flash_grounded-sam2

# Browse the dataset
uv run streamlit run src/visualizer.py -- --video-dir /path/to/ego4d/full_scale
```

See [src/README.md](src/README.md) for details (API keys, preprocessing, metrics).

## Citation

```bibtex
@inproceedings{kato2026longegorefer,
  title     = {LongEgoRefer: A Benchmark for Long-Form Egocentric Video Referring Expression Comprehension},
  author    = {Kato, Shunya and Miyanishi, Taiki and Kurita, Shuhei and Ukai, Mahiro and Inoue, Nakamasa and Chu, Chenhui},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

The annotations follow the Ego4D / EgoTracks terms of use. See [dataset/README.md](dataset/README.md) for details.
