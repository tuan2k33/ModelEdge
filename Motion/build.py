"""The fused engine for several cameras per call: YOLO26 weights + the background model -> one TensorRT engine, `image` [B,H,W,3] and `state` [B,8,180,320] in,
`detections` [B,300,6], `state_out` and `mask` out. Run it with model.Engine(engine, cameras=N).
    python build.py cache/yolo26n.pt cache/fused.engine [--batch 4] [--top 720 1280] [--boost 0.5] [--fps 25] [--every 5] [--form 5] [--cap 600] [--thr 25]
--batch times --top is what the GPU memory limits: 4 x 1280x720 builds on 4 GB, 4 x 1920x1080 does not."""
import argparse, importlib.util
from pathlib import Path

import numpy as np, onnx
from onnx import numpy_helper as nh
from ultralytics import YOLO

from model import compile_engine, fuse

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("engine")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--top", type=int, nargs=2, default=(720, 1280), help="largest frame: height width")
ap.add_argument("--boost", type=float, default=0.5)
ap.add_argument("--fps", type=float, default=25)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--form", type=float, default=5)
ap.add_argument("--cap", type=float, default=600)
ap.add_argument("--thr", type=float, default=25)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("detector", Path(__file__).resolve().parent.parent / "Detector" / "model.py")
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)
model = YOLO(a.weights)
m = detector.with_front_end(model.export(format="onnx", imgsz=640, nms=False, half=True, dynamic=True, device=0), [model.names[i] for i in range(len(model.names))])
for i in m.graph.initializer:
    if i.name == "pre_fixed":  # the Detector pins its letterbox output to batch 1: 0 keeps the batch of the input
        i.CopyFrom(nh.from_array(np.array([0, 3, detector.SIDE, detector.SIDE], np.int64), "pre_fixed"))
m.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "B"
path = Path(a.engine).with_suffix(".onnx")
onnx.save(m, path)
fuse(path, path, *(max(1, round(s * a.fps / a.every)) for s in (a.form, a.cap)), a.thr, a.boost)
compile_engine(path, a.engine, tuple(a.top), a.batch)
print(f"engine: {a.engine}")
