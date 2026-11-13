"""Background model on a video: left the frame with the moving regions boxed, right the background it has learned. H264 out.
    python run.py in.mp4 out.mp4 [--every 5] [--form 5] [--cap 600] [--thr 25]
    python run.py in.mp4 out.mp4 --detector ../Detector/weight/coco/model.onnx [--boost 0.5] [--conf 0.25]   one engine with the detector: its boxes instead, yellow where the mask raised the score"""
import argparse, json, subprocess, time
from pathlib import Path

import cv2, numpy as np, onnx, torch

from model import CACHE, PH, PW, VER, Engine, compile_engine, export, fuse

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--every", type=int, default=5, help="update the background every Nth frame (the mask is computed on all of them)")
ap.add_argument("--form", type=float, default=5, help="seconds the background takes to re-form after most of the frame changes at once (lights on or off)")
ap.add_argument("--cap", type=float, default=600, help="seconds: the most weight a background can gather, so the longest a still object stays an object (and a moved one takes to be accepted)")
ap.add_argument("--thr", type=float, default=25, help="a pixel matches when no channel differs by more than this (0..255)")
ap.add_argument("--min-area", type=int, default=12, help="smallest box, in mask pixels")
ap.add_argument("--detector", help="model.onnx of a Detector run: build and run the fused engine (frames up to 2048x2048)")
ap.add_argument("--boost", type=float, default=0.5, help="added to the logit of the score of a box whose centre is on the mask")
ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
fps, w0, h0 = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
form, top = (max(1, round(s * fps / a.every)) for s in (a.form, a.cap))
if a.detector:
    names = json.loads({p.key: p.value for p in onnx.load(a.detector, load_external_data=False).metadata_props}["classes"])
    eng = CACHE / f"fused{VER}-{Path(a.detector).parent.name}-{form}-{top}-{a.thr:g}-{a.boost:g}.engine"
    if not eng.exists():
        fuse(a.detector, eng.with_suffix(".onnx"), form, top, a.thr, a.boost)
        compile_engine(eng.with_suffix(".onnx"), eng, (2048, 2048))
    r = min(640 / h0, 640 / w0)  # the detector's letterbox (Detector/model.py)
    left, up = round((640 - round(w0 * r)) / 2 - 0.1), round((640 - round(h0 * r)) / 2 - 0.1)
else:
    eng = CACHE / f"bg{VER}-{form}-{top}-{a.thr:g}.engine"
    if not eng.exists():
        export(eng.with_suffix(".onnx"), form, top, a.thr)
        compile_engine(eng.with_suffix(".onnx"), eng)
net = Engine(eng)

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * w0}x{h0}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
t, area, i = [], [], 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    x = torch.from_numpy(fr).cuda().flip(-1)[None]
    m = net(x, update=i % a.every == 0)[0].cpu().numpy()
    area.append(m.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    fr[big] = fr[big] // 2 + np.array([0, 0, 127], np.uint8)
    for x1, y1, x2, y2, conf, c in net.det[0].cpu().numpy() if a.detector else []:
        if conf >= a.conf:
            x1, y1, x2, y2 = int((x1 - left) / r), int((y1 - up) / r), int((x2 - left) / r), int((y2 - up) / r)
            on = m[min(max((y1 + y2) * PH // (2 * h0), 0), PH - 1), min(max((x1 + x2) * PW // (2 * w0), 0), PW - 1)]
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 255) if on else (0, 255, 0), 2)
            cv2.putText(fr, f"{names[int(c)]} {conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if on else (0, 255, 0), 1)
    for x1, y1, bw, bh, s in [] if a.detector else stats[1:]:
        if s >= a.min_area:
            cv2.rectangle(fr, (x1 * w0 // PW, y1 * h0 // PH), ((x1 + bw) * w0 // PW, (y1 + bh) * h0 // PH), (0, 255, 0), 2)
    bg = cv2.resize(net.background().flip(-1).cpu().numpy(), (w0, h0))
    ff.stdin.write(np.hstack([fr, bg]).tobytes())
    i += 1
ff.stdin.close(); ff.wait()
for _ in range(220):  # timed after the encoder has gone: x264 takes every core and the wait for the GPU then measures the CPU queue, not the engine
    t.append(time.perf_counter())
    net(x, update=False)
print(f"{i} frames, engine {1000 * np.median(np.diff(t[20:])):.2f} ms/frame, foreground {100 * np.mean(area):.1f}% of the frame on average")
