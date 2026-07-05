"""Train RF-DETR-Small on the local COCO dataset (dataset/{train,valid,test}).

Prereq: run ``scripts/prepare_dataset.py`` first (merges the two YOLO datasets
into one COCO dataset with classes {enemy, enemy_head}).

Run:  uv run python scripts/train.py

Checkpoints land in ``output/``. Then build the TensorRT engine + point the app
at it with:  uv run python scripts/export_engine.py
"""
from __future__ import annotations

from rfdetr import RFDETRSmall


def main() -> None:
    RFDETRSmall().train(
        dataset_dir="dataset",
        epochs=100,
        batch_size=4,          # RF-DETR-Small @512 fits a 24 GB RTX 3090
        grad_accum_steps=4,    # effective batch = 16
        lr=1e-4,
        num_workers=6,
        early_stopping=True,   # stop when validation mAP plateaus
        output_dir="output",
    )
    print("\nTraining done -> output/. Next: uv run python scripts/export_engine.py")


if __name__ == "__main__":
    main()
