"""Merge the two YOLO datasets under dataset/ into one COCO dataset for RF-DETR.

- Reconciles classes to {0: enemy, 1: enemy_head}; drops the unused
  ``Valorant-enemy`` class from dataset 2.
- Converts YOLO (class cx cy w h, normalized) -> COCO (bbox x,y,w,h absolute px).
- Splits into train/valid/test (80/10/10) and writes ``_annotations.coco.json``
  per split (the layout RF-DETR's train() expects).
- Deletes the two source subfolders afterwards (only once every image is copied).

Run:  uv run python scripts/prepare_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

DATASET = Path("dataset")
# source subfolder -> {yolo_class: canonical_class}. Classes not listed are dropped.
SOURCES = {
    "dataset 1": {0: 0, 1: 1},          # ['enemy', 'enemy_head']
    "dataset 2": {1: 0, 2: 1},          # ['Valorant-enemy'(drop), 'enemy', 'enemy_head']
}
CATEGORIES = [                          # COCO is 1-indexed: canonical c -> category_id c+1
    {"id": 1, "name": "enemy", "supercategory": "none"},
    {"id": 2, "name": "enemy_head", "supercategory": "none"},
]
SPLIT = {"train": 0.80, "valid": 0.10, "test": 0.10}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42


def gather():
    samples = []                        # (image_path, label_path, remap)
    for sub, remap in SOURCES.items():
        img_dir, lbl_dir = DATASET / sub / "images", DATASET / sub / "labels"
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                samples.append((img, lbl_dir / (img.stem + ".txt"), remap))
    return samples


def main():
    samples = gather()
    total = len(samples)
    random.Random(SEED).shuffle(samples)
    n_train = int(total * SPLIT["train"])
    n_valid = int(total * SPLIT["valid"])

    def split_for(i):
        return "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")

    for s in SPLIT:
        (DATASET / s).mkdir(parents=True, exist_ok=True)
    coco = {s: {"images": [], "annotations": [], "categories": CATEGORIES} for s in SPLIT}
    iid = {s: 0 for s in SPLIT}
    aid = {s: 0 for s in SPLIT}
    copied = 0

    for i, (img, lbl, remap) in enumerate(samples):
        s = split_for(i)
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue
        dst = f"{img.parent.parent.name.replace(' ', '')}_{img.name}"   # unique across sources
        shutil.copy2(img, DATASET / s / dst)
        copied += 1
        image_id = iid[s]; iid[s] += 1
        coco[s]["images"].append({"id": image_id, "file_name": dst, "width": w, "height": h})
        if lbl.exists():
            for line in lbl.read_text().split("\n"):
                p = line.split()
                if len(p) != 5:
                    continue
                c = int(p[0])
                if c not in remap:            # drop unused classes (Valorant-enemy)
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
                coco[s]["annotations"].append({
                    "id": aid[s], "image_id": image_id, "category_id": remap[c] + 1,
                    "bbox": [x, y, ww, hh], "area": ww * hh, "iscrowd": 0})
                aid[s] += 1

    for s in SPLIT:
        (DATASET / s / "_annotations.coco.json").write_text(json.dumps(coco[s]))
        print(f"  {s:6s}: {len(coco[s]['images']):5d} images  {len(coco[s]['annotations']):6d} boxes")

    assert copied == total, f"copied {copied} != {total}; NOT deleting sources"
    for sub in SOURCES:
        shutil.rmtree(DATASET / sub)
    print(f"merged {total} images; deleted the {len(SOURCES)} source subfolders.")


if __name__ == "__main__":
    main()
