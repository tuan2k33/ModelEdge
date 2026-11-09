"""python predict.py weight/<name>/model.engine image.jpg [--conf 0.25] [--out result.jpg]  (any image size up to 2048x2048, the letterbox runs in the engine)"""
import argparse, json
from pathlib import Path

import numpy as np
import onnx
from PIL import Image, ImageDraw

from model import Engine, letterbox

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("image")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--out", help="save the image with the boxes drawn")
a = ap.parse_args()
names = json.loads({p.key: p.value for p in onnx.load(Path(a.engine).with_suffix(".onnx"), load_external_data=False).metadata_props}["classes"])
img = Image.open(a.image).convert("RGB")
r, left, top = letterbox(img.height, img.width)  # boxes come out in the 640x640 letterboxed frame
draw = ImageDraw.Draw(img)
for x1, y1, x2, y2, conf, cls in Engine(a.engine)(np.asarray(img)[None])[0][0]:
    if conf < a.conf:
        continue
    box = [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r]
    print(f"{names[int(cls)]:15s} {100 * conf:5.1f}%  x1 {box[0]:.0f} y1 {box[1]:.0f} x2 {box[2]:.0f} y2 {box[3]:.0f}")
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0] + 3, box[1] + 3), f"{names[int(cls)]} {100 * conf:.0f}%", fill="red")
if a.out:
    img.save(a.out)
