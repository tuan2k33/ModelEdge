"""COCO val2017 mAP (pycocotools) of an engine made by train.py; boxes are mapped back from the letterboxed frame to the original image.
Data: data/coco/{val2017.zip,annotations_trainval2017.zip} (images.cocodataset.org). Only for COCO-trained engines.
    python eval.py weight/coco/model.engine [--start N] [--limit N]"""
import argparse, json, zipfile
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.io import ImageReadMode, decode_jpeg
from ultralytics.data.converter import coco80_to_coco91_class

from model import Engine, letterbox

COCO_DIR = Path("data/coco")
ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("--start", type=int, default=0, help="skip the first N images (val2017 sorted by name)")
ap.add_argument("--limit", type=int, help="N images only (after --start)")
a = ap.parse_args()
ann_file = COCO_DIR / "instances_val2017.json"
if not ann_file.exists():
    zipfile.ZipFile(COCO_DIR / "annotations_trainval2017.zip").extract("annotations/instances_val2017.json", COCO_DIR)
    (COCO_DIR / "annotations/instances_val2017.json").replace(ann_file)
ann, z, cat, eng = COCO(str(ann_file)), zipfile.ZipFile(COCO_DIR / "val2017.zip"), coco80_to_coco91_class(), Engine(a.engine)
names = sorted(n for n in z.namelist() if n.endswith(".jpg"))[a.start:][:a.limit]
res = []
for n in names:
    g = decode_jpeg(torch.frombuffer(bytearray(z.read(n)), dtype=torch.uint8), device="cuda", mode=ImageReadMode.RGB).permute(1, 2, 0)  # HWC uint8 on the GPU
    r, left, top = letterbox(*g.shape[:2])
    res += [dict(image_id=int(Path(n).stem), category_id=cat[int(c)], bbox=[(x1 - left) / r, (y1 - top) / r, (x2 - x1) / r, (y2 - y1) / r], score=float(s))
            for x1, y1, x2, y2, s, c in eng(g[None])[0][0] if s >= 0.001]
e = COCOeval(ann, ann.loadRes(res), "bbox")
e.params.imgIds = [int(Path(n).stem) for n in names]
e.evaluate(); e.accumulate(); e.summarize()
print(json.dumps(dict(images=len(names), map50_95=round(100 * e.stats[0], 2), map50=round(100 * e.stats[1], 2), map75=round(100 * e.stats[2], 2))))
