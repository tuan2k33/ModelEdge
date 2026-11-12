"""Background model on a video: left the frame with the moving regions boxed, right the background it has learned. H264 out.
    python run.py in.mp4 out.mp4 [--every 5] [--form 5] [--cap 600] [--thr 25]"""
import argparse, subprocess, time

import cv2, numpy as np, torch

from model import CACHE, PH, PW, VER, Engine, compile_engine, export

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--every", type=int, default=5, help="update the background every Nth frame (the mask is computed on all of them)")
ap.add_argument("--form", type=float, default=5, help="seconds the background takes to re-form after most of the frame changes at once (lights on or off)")
ap.add_argument("--cap", type=float, default=600, help="seconds: the most weight a background can gather, so the longest a still object stays an object (and a moved one takes to be accepted)")
ap.add_argument("--thr", type=float, default=25, help="a pixel matches when no channel differs by more than this (0..255)")
ap.add_argument("--min-area", type=int, default=12, help="smallest box, in mask pixels")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
fps, w0, h0 = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
form, top = (max(1, round(s * fps / a.every)) for s in (a.form, a.cap))
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
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mask = net(x, update=i % a.every == 0)
    t.append(time.perf_counter() - t0)
    m = mask.cpu().numpy()
    area.append(m.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
    fr[big] = fr[big] // 2 + np.array([0, 0, 127], np.uint8)
    for x1, y1, bw, bh, s in stats[1:]:
        if s >= a.min_area:
            cv2.rectangle(fr, (x1 * w0 // PW, y1 * h0 // PH), ((x1 + bw) * w0 // PW, (y1 + bh) * h0 // PH), (0, 255, 0), 2)
    bg = cv2.resize(net.background.flip(-1).cpu().numpy(), (w0, h0))
    ff.stdin.write(np.hstack([fr, bg]).tobytes())
    i += 1
ff.stdin.close(); ff.wait()
print(f"{i} frames, engine {1000 * np.median(t[5:]):.2f} ms/frame, foreground {100 * np.mean(area):.1f}% of the frame on average")
