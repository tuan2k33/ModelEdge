# ModelEdge

A small INT8 image classifier built from a frozen, pre-quantized ResNet50 and a small MLP head you train in under a minute.
The result is **one TensorRT engine**: raw `uint8` image in, label and confidence out, all on the GPU.

```
uint8 image [N,H,W,3]
  -> resize 224 / normalize                      (inside the engine)
  -> ResNet50, INT8 QDQ                         (frozen, from the ONNX Model Zoo)
  -> int8 global-average-pool tensor [N,2048]
  -> int8 MLP head 2048-1024-classes (trained here)  (int8 x int8 GEMMs)
  -> label, confidence %                        (LUT softmax)
```

Only the head is trained, on cached embeddings, so training a new dataset takes minutes, not hours.

## Results

Frozen backbone + MLP head (2048-1024-classes, ReLU, int8 hidden activation), default settings, no hyperparameter tuning, one run per dataset with a random seed (CIFAR-100 and MNIST: the better of two runs). Test accuracy is measured through the final engine.

| Dataset | Classes | Train / val / test | Train acc | Val acc | Test acc |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 45k / 5k / 10k | 95.3% | 89.2% | **89.4%** |
| CIFAR-100 | 100 | 45k / 5k / 10k | 84.7% | 70.5% | **69.5%** |
| MNIST | 10 | 54k / 6k / 10k | 99.8% | 98.1% | **98.3%** |
| Tiny ImageNet | 200 | 90k / 10k / 10k | 80.7% | 72.4% | **72.2%** |
| Intel Image Classification (Kaggle) | 6 | 12.6k / 1.4k / 3k | 95.7% | 93.4% | **92.0%** |
| EuroSAT RGB | 10 | 19.4k / 2.2k / 5.4k | 97.7% | 94.8% | **94.8%** |

Speed on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, CUDA 13:

| | |
|---|---|
| Engine latency, batch 1 / batch 64 | about 2-3 ms / 18 ms (about 3,500 img/s at batch 64) |
| Embeddings for 60k CIFAR images | about 20 s (once, then cached) |
| Head training | 7-19 s (14 s CIFAR-10, 19 s Tiny ImageNet) |
| Engine build | about 2 min per run |
| GPU memory for the engine | about 350 MB |

## Usage

Needs an NVIDIA GPU, PyTorch with CUDA, TensorRT 11 (Python package plus the `trtexec` binary on `PATH`), `cuda-python`, `onnx`, `onnxruntime`, `numpy` and `pillow`.
The INT8 ResNet50 (`resnet50-v1-12-int8.onnx`) and the datasets are downloaded on first use with `curl`, so it must be installed. Run from `IntClassifier/`:

```bash
cd IntClassifier
python train.py cifar10
python train.py tiny-imagenet          # also: cifar100, mnist (downloaded on first use)
python train.py path/to/dataset          # path/to/dataset/{train,test}/<class>/*.jpg
python predict.py weight/run-<time>/model.engine image.jpg
```

Options: `--epochs N` (default: train until early stop), `--stages` (default `1e-3:10,5e-4:8,2e-4:6,1e-4:4`: lr:patience pairs, lower lr and smaller patience each time the val loss stalls), `--eps` (Adam eps, default 1e-8), `--bs` (head batch size, default 1024), `--solver lbfgs` (linear head only; about 15x faster to train but 0.6-0.7 point less accurate in our runs), `--hidden H` (MLP head 2048-H-classes with ReLU, default 1024, exported with an int8 hidden activation; 0 = one linear layer), `--seed` (default: random, saved in `metrics.json`), `--per-class N`, `--test-per-class N`.

Each run writes `weight/run-<time>/`: `model.onnx`, `model.engine`, `metrics.json` (dataset, train/val/test accuracy and loss, timings, GPU and RAM use) and `train.log`.
`predict.py` reads the class names from the `.onnx` next to the `.engine`.

Code in `IntClassifier/`: `train.py` (head training and the run), `model.py` (QLinear to QDQ conversion, engine runner, head graph), `dataset.py` (loaders), `predict.py`.
Your own data: put it in `<root>/train/<class>/` and `<root>/test/<class>/` (same class folders in both) and pass `<root>`. To add a downloadable dataset, write a loader in `dataset.py` and register it in `DATASETS`. The Intel and EuroSAT numbers above come from the Kaggle downloads arranged this way (Intel: `seg_train` and `seg_test` renamed to `train` and `test`; EuroSAT: every 5th image of each class moved to `test`).

## INT8 vs FP32

`Classifier/` is the same pipeline in float32 (ResNet50 FP32 from the Model Zoo, same in-engine preprocessing, linear head trained in FP32, one FP32 TensorRT engine).
`Classifier/` is self-contained (its own copies of `dataset.py`, `model.py`, `train.py`, plus `compare.py`, and its own `data/`). From `Classifier/`:

```bash
cd Classifier
python train.py cifar10
python compare.py cifar10 ../IntClassifier/weight/run-<int8 run> weight/run-<fp32 run>
```

`compare.py` measures both engines back to back on the same 10k test images. Same seed and val split, linear head, one run each, RTX 3050 Ti Laptop:

| | CIFAR-10 INT8 | CIFAR-10 FP32 | Tiny ImageNet INT8 | Tiny ImageNet FP32 |
|---|---|---|---|---|
| Test accuracy | 87.77% | 88.03% | 70.82% | 71.07% |
| Engine, batch 1 | 2.2 ms | 4.9 ms | 1.8 ms | 5.0 ms |
| Engine, batch 64 | 18.1 ms | 72.4 ms | 18.3 ms | 72.9 ms |
| End-to-end | 2137 img/s | 603 img/s | 2214 img/s | 595 img/s |
| Engine file | 24 MB | 90 MB | 25 MB | 92 MB |

This comparison was run with the earlier preprocessing (resize 256, center-crop 224) and a linear head, and `Classifier/` has not been updated to the current one. INT8 is about 2-3x faster at batch 1, 4x at batch 64 and 3.6x end to end, and the engine is 3.7x smaller, for about 0.25 points of accuracy. That gap has the same sign on both datasets but is of the size of the seed-to-seed spread, so treat it as roughly 0.2-0.3 points. GPU memory for the engine was about the same (about 350 MB, dominated by fixed buffers) and build time is similar. FP32 here is TensorRT's default float32 path, which may use TF32 tensor cores on this GPU.

## How it works

- **QLinear to QDQ.** The Model Zoo model uses uint8 activations with non-zero zero-points, which TensorRT rejects. `model.py` rewrites it as Quantize/Dequantize pairs with int8 symmetric activations. Weights and biases keep their exact values. Post-ReLU tensors lose about one bit, and the embeddings end up at cosine about 0.9 against the original model, which did not hurt accuracy here.
- **Head.** MLP 2048-1024-classes: Adam on the int8 embeddings with staged learning rates (1e-3 down to 1e-4, patience shrinking each stage), 10% of train held out for validation, best-val-loss weights kept, early stopping. Weights are then quantized to int8 and appended to the backbone graph. The hidden ReLU output is requantized to int8 with scale 127/`hidden_max`, where `hidden_max` is the 99.9th percentile of that output on 4096 train rows at the best checkpoint (the top 0.1% is clipped).
- **Confidence.** A 256-entry lookup table replaces `exp()`.

## Limits

- The engine is specific to the GPU and TensorRT version that built it.
- Input is resized straight to 224x224 with plain bilinear (no antialiasing, aspect ratio not kept), no crop, in training and inference. Fine for small or square images, it aliases when downscaling large photos. The engine takes sides up to 512, bigger images are shrunk to 512 on the CPU first (antialiased). Batch at most 64.
- The normalization at the start of the engine, the global-average-pool, the bias add and the LUT softmax are float operations on integer values. TensorRT has no `MatMulInteger` or `BitShift`.
- Accuracy numbers come from single runs and are not tuned.
