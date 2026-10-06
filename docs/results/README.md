# Measured results

| File | What it backs | How it was produced |
|---|---|---|
| `val_metrics.csv` | Detector mAP in the top-level README | Validation rows (one per epoch) extracted from RF-DETR's `output/metrics.csv` from the 100-epoch run (`scripts/train.py`). |
| `detect_latency.json` | Detector FPS in the top-level README | `uv run python scripts/bench_detect.py` — 1000 timed `detect()` calls after 50 warm-up, 384 px ROI, TensorRT engine. |

**Detector accuracy.** `scripts/export_engine.py` exports `output/checkpoint_best_ema.pth`, which
holds **epoch 98** (EMA weights). Its row in `val_metrics.csv`: mAP@50 = 0.843,
mAP@50:95 = 0.564. These are on the 10 % validation split of the merged dataset
(`scripts/prepare_dataset.py`, seed 42), not on frames from the target game, so they say
nothing about real-world accuracy there.

**Detector speed.** 7.6 ms mean / 9.2 ms p99 per `detect()` (≈131 FPS mean, ≈132 FPS from
p50) on an RTX 3090, TensorRT 11.1, torch 2.12.1+cu130. This covers BGR→RGB, resize and
normalize, engine execution and decoding, on one fixed frame; it excludes capture, tracking
and aiming.
