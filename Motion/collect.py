"""Data for fuse.py: every 15th frame of a video (after the background has formed) with the Motion mask, a reference detector's boxes and the boxes of the detectors under test.
    python collect.py data/traffic_3min.mp4 cache/traffic.npz [scale [blur [noise]]]
scale, blur (Gaussian sigma after shrinking) and noise (sigma, 0..255) stand for a poor camera: Motion and the detectors under test get that frame, the reference the clean one."""
import sys

import cv2, numpy as np, torch
from ultralytics import YOLO

from model import CACHE, VER, Engine, compile_engine, export

src, dst = sys.argv[1:3]
scale, blur, noise = (float(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((3, 1), (4, 0), (5, 0)))
rng = np.random.default_rng(0)


def poor(fr):
    if scale != 1:
        fr = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if blur:
        fr = cv2.GaussianBlur(fr, (0, 0), blur)
    if noise:
        fr = (fr + rng.normal(0, noise, fr.shape)).clip(0, 255).astype(np.uint8)
    return fr


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
    low = poor(fr)
    m = net(torch.from_numpy(low).cuda().flip(-1)[None], update=i % 5 == 0)
    if i >= 20 * fps and i % 15 == 0:
        frames.append((fr, low))
        masks.append(m[0].cpu().numpy())
    i += 1
del net
out = {"masks": np.stack(masks), "shape": np.array(frames[0][0].shape[:2])}
for name, size, conf in [("yolo26x", 1280, 0.25), ("yolo26n", 640, 0.001), ("yolo26m", 640, 0.001)]:
    model, rows = YOLO(str(CACHE / f"{name}.pt")), []
    for k, (fr, low) in enumerate(frames):
        small = name != "yolo26x"
        b = model.predict(low if small else fr, imgsz=size, conf=conf, max_det=300, quantize=16, verbose=False)[0].boxes
        rows.append(np.concatenate([np.full((len(b), 1), k), b.xyxy.cpu().numpy() / (scale if small else 1), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1))
    out[name] = np.concatenate(rows).astype(np.float32)
    print(name, len(out[name]), "boxes on", len(frames), "frames")
    del model
    torch.cuda.empty_cache()
np.savez_compressed(dst, **out)
