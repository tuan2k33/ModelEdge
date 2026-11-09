"""python train.py [data.yaml] [--epochs N] [--model cache/yolo26n.pt] [--name NAME] [--imgsz 640] [--batch 16]
Fine-tunes YOLO26 (Ultralytics) on a dataset in Ultralytics format, then exports the best weights as ONE FP16 TensorRT engine: uint8 image of any size -> [1,300,6]
(NMS-free: x1 y1 x2 y2 conf class), the letterbox runs inside the engine. Output: weight/<name>/{weights/best.pt,model.onnx,model.engine,...}.
--epochs 0 skips training and exports --model as it is (the COCO yolo26n.pt: python train.py --epochs 0 --name coco)."""
import argparse
from pathlib import Path

import onnx
from ultralytics import YOLO

from model import CACHE, compile_engine, with_front_end

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", default="coco8.yaml", help="Ultralytics dataset yaml (coco8.yaml downloads a 8-image demo)")
ap.add_argument("--model", default=str(CACHE / "yolo26n.pt"), help="weights to start from (downloaded if missing and named like an Ultralytics release)")
ap.add_argument("--epochs", type=int, default=100, help="0: export only")
ap.add_argument("--imgsz", type=int, default=640, help="the engine letterboxes to 640: other sizes need SIDE in model.py changed")
ap.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory (4 GB: 8)")
ap.add_argument("--name")
a = ap.parse_args()
assert a.imgsz == 640, "model.py letterboxes to 640x640"
name = a.name or Path(a.data).stem
out = Path("weight").resolve() / name
out.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

model = YOLO(a.model)
if a.epochs:
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, project=str(out.parent), name=name, exist_ok=True)
    model = YOLO(out / "weights" / "best.pt")
tmp = model.export(format="onnx", imgsz=a.imgsz, nms=False, half=True, device=0)  # FP16 weights, end-to-end head (no NMS)
onnx.save(with_front_end(tmp, [model.names[i] for i in range(len(model.names))]), out / "model.onnx")
compile_engine(out / "model.onnx", out / "model.engine")
print(f"engine: {out / 'model.engine'}")
