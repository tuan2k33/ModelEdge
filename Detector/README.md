# Detector

YOLO26 object detection as **one FP16 TensorRT engine**: raw `uint8` image of any size in, boxes out, all on the GPU. Fine-tune on your own data with Ultralytics, export, run.

```
uint8 image [1,H,W,3]
  -> letterbox to 640x640, /255      (inside the engine, Ultralytics' letterbox)
  -> YOLO26, FP16                    (Ultralytics export, NMS-free end-to-end head)
  -> [1,300,6]  x1 y1 x2 y2 conf class   (in the letterboxed 640x640 frame)
```

No NMS to run: the head already outputs at most 300 boxes. `predict.py` maps them back to the original image.

## Results

COCO val2017, all 5000 images, pycocotools, `yolo26n`. Speed is GPU time per 640x640 image (`trtexec`) on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, batch 1.

| Engine | mAP50-95 | mAP50 | mAP75 | Time | Size |
|---|---|---|---|---|---|
| **FP16 (this folder)** | **40.21** | 55.77 | 43.56 | **1.5 ms** | 49 MB |
| FP32 | 40.23 | 55.80 | 43.58 | 3.2 ms | 65 MB |
| INT8 (tried, removed) | 36.82 | 53.17 | 40.11 | 3.6 ms | 51 MB |

FP16 loses nothing against FP32 and is 2x faster. INT8 loses about 3.4 points and is not faster: YOLO's SiLU activation cannot be fused into the conv, so every layer becomes four kernels (INT8 conv, dequantize, float SiLU, quantize) and only about 20% of the time is the INT8 conv itself. Calibrating on 1000 COCO images instead of 4 did not help (MinMax 31.7, Entropy 34.9, Percentile 36.4 against 36.4 for the 4 images, on a held-out half of val2017).

## Usage

Needs an NVIDIA GPU, PyTorch with CUDA, TensorRT 11 (Python package plus `trtexec` on `PATH`), `ultralytics`, `onnx`, `cuda-python`, `torchvision`, `pillow`. Run from `Detector/`:

```bash
cd Detector
python train.py --epochs 0 --name coco                  # COCO yolo26n as it is -> weight/coco/model.engine
python train.py path/to/data.yaml --epochs 100 --name mine   # fine-tune on your data (Ultralytics dataset yaml), then export
python predict.py weight/coco/model.engine image.jpg --out result.jpg
python eval.py weight/coco/model.engine                 # COCO val2017 mAP, needs data/coco (images.cocodataset.org)
```

`train.py` options: `--model` (start weights, default `cache/yolo26n.pt`, downloaded on first use), `--epochs` (0 = export only), `--batch` (default 16, use 8 on a 4 GB GPU), `--name`. Each run writes `weight/<name>/`: `weights/best.pt` and the Ultralytics training files, `model.onnx` and `model.engine`. `predict.py` reads the class names from the `.onnx` next to the engine.

Code: `train.py` (train and export), `model.py` (letterbox graph, FP16 ONNX wrapper, engine runner), `predict.py`, `eval.py`.

## Limits

- The engine is specific to the GPU and TensorRT version that built it. Batch 1, image sides from 8 to 2048.
- Images are letterboxed to 640x640 (aspect ratio kept, padded with gray), so small objects in large images get small. Training at another size would need `SIDE` in `model.py` changed.
- Boxes can fall slightly outside the image (a few pixels); clip them if you need to.
- `eval.py` only makes sense for COCO-trained engines. The mAP above is for the pretrained COCO `yolo26n`, not for a fine-tuned one.
- YOLO26 is AGPL-3.0 (Ultralytics).
