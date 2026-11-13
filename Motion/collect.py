"""Data for fuse.py: every 15th frame of a video (after the background has formed) with the Motion mask, a reference detector's boxes and the boxes of the detectors under test.
    python collect.py data/traffic_3min.mp4 cache/traffic.npz [scale]
scale (0.33...): the detectors under test get the frame shrunk by it, to stand for a low-resolution camera; the reference and Motion keep the full frame."""
import sys

import cv2, numpy as np, torch
from ultralytics import YOLO

from model import CACHE, VER, Engine, compile_engine, export

src, dst = sys.argv[1:3]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS)
form, top = round(5 * fps / 5), round(600 * fps / 5)
eng = CACHE / f"bg{VER}-{form}-{top}-25.engine"
if not eng.exists():
    export(eng.with_suffix(".onnx"), form, top, 25)
    compile_engine(eng.with_suffix(".onnx"), eng)
net, frames, masks, i = Engine(eng), [], [], 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    m = net(torch.from_numpy(fr).cuda().flip(-1)[None], update=i % 5 == 0)
    if i >= 20 * fps and i % 15 == 0:
        frames.append(fr)
        masks.append(m.cpu().numpy().copy())
    i += 1
del net
out = {"masks": np.stack(masks), "shape": np.array(frames[0].shape[:2])}
for name, size, conf in [("yolo26x", 1280, 0.25), ("yolo26n", 640, 0.001), ("yolo26m", 640, 0.001)]:
    model, rows = YOLO(str(CACHE / f"{name}.pt")), []
    for k, fr in enumerate(frames):
        small = name != "yolo26x" and scale != 1
        b = model.predict(cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if small else fr, imgsz=size, conf=conf, max_det=300, quantize=16, verbose=False)[0].boxes
        rows.append(np.concatenate([np.full((len(b), 1), k), b.xyxy.cpu().numpy() / (scale if small else 1), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1))
    out[name] = np.concatenate(rows).astype(np.float32)
    print(name, len(out[name]), "boxes on", len(frames), "frames")
    del model
    torch.cuda.empty_cache()
np.savez_compressed(dst, **out)
