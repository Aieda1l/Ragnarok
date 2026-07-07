"""Export the trained RF-DETR checkpoint -> ONNX -> TensorRT engine, and point
the app config at it.

Run (after scripts/train.py):  uv run python scripts/export_engine.py
INT8 (smaller/faster, tiny accuracy loss):  uv run python scripts/export_engine.py --int8
   (calibrates on dataset/train; TensorRT 11 reads the ONNX Q/DQ nodes.)

The TRT runtime (detection.rfdetr_trt) reads the engine's output shapes, so the
2-class trained model works without any code change.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

from rfdetr import RFDETRSmall

from ragnarok.detection.rfdetr_trt import build_trt_engine
from ragnarok.config.store import load_config, save_config
from ragnarok.app import _config_path


def _best_checkpoint() -> str:
    for pat in ("output/checkpoint_best_ema.pth", "output/checkpoint_best_total.pth",
                "output/checkpoint_best_regular.pth", "output/checkpoint_best*.pth",
                "output/checkpoint.pth", "output/*.pth"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError("no checkpoint in output/ — run scripts/train.py first")


def main() -> None:
    int8 = "--int8" in sys.argv
    ckpt = _best_checkpoint()
    print("checkpoint:", ckpt, "| INT8" if int8 else "| FP16/TF32")
    Path("engines").mkdir(exist_ok=True)
    export_kwargs = dict(output_dir="engines", format="onnx")
    if int8:                                   # PTQ: rfdetr writes Q/DQ nodes into the ONNX
        export_kwargs.update(quantization="int8", calibration_data="dataset/train",
                             max_images=100)
        print("INT8: calibrating on dataset/train (100 imgs)…")
    onnx = RFDETRSmall(pretrain_weights=ckpt).export(**export_kwargs)
    print("onnx:", onnx)
    engine = build_trt_engine(str(onnx), "engines/rfdetr-trained.engine")
    print("engine:", engine)
    cfg = load_config(_config_path())
    new_det = cfg.detection.model_copy(
        update={"backend": "rfdetr_trt", "engine_path": str(Path(engine).resolve())})
    save_config(cfg.model_copy(update={"detection": new_det}), _config_path())
    print("config updated -> app now uses the trained TensorRT engine")


if __name__ == "__main__":
    main()
