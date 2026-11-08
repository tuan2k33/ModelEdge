# ModelEdge

A small INT8 image classifier built from a frozen, pre-quantized ResNet50 and a linear head you train in about a minute.
The result is **one TensorRT engine**: raw `uint8` image in, label and confidence out, all on the GPU.

```
uint8 image [N,H,W,3]
  -> resize 256 / center-crop 224 / normalize   (inside the engine)
  -> ResNet50, INT8 QDQ                         (frozen, from the ONNX Model Zoo)
  -> int8 global-average-pool tensor [N,2048]
  -> int8 linear head (trained here)            (int8 x int8 GEMM)
  -> label, confidence %                        (LUT softmax)
```

Only the head is trained, on cached embeddings, so training a new dataset takes minutes, not hours.

## Results

Frozen backbone + linear head, no hyperparameter tuning, a few runs per dataset. Test accuracy is measured through the final engine.

| Dataset | Classes | Train / val / test | Test accuracy |
|---|---|---|---|
| CIFAR-10 | 10 | 45k / 5k / 10k | 87.7-88.1% (3 seeds: 87.72, 87.96, 88.07) |
| Tiny ImageNet | 200 | 90k / 10k / 10k | 70.8% (3 runs: 70.75, 70.82, 70.83) |

The int8 head and the engine match the float head to within 0.1 point, so quantizing the head costs no measurable accuracy.
An MLP head (`--hidden 2048`, int8 hidden activation, same engine speed) gave 88.57% on CIFAR-10 and 70.56% on Tiny ImageNet in one run each, so it helps a little on CIFAR-10 and not on Tiny ImageNet.

Speed on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, CUDA 13:

| | |
|---|---|
| Engine latency, batch 1 / batch 64 | about 2 ms / 18 ms (about 3,500 img/s at batch 64) |
| Embeddings for 60k CIFAR images | about 20 s |
| Head training | about 30 s (CIFAR), about 20 s (Tiny ImageNet) |
| Engine build | about 2 min per run |
| GPU memory for the engine | about 350 MB |

Read the accuracy with care: the backbone was pre-trained on ImageNet-1k and Tiny ImageNet is a subset of it, so 70.8% is probably optimistic. CIFAR and Tiny ImageNet images are 32 and 64 px and get upscaled to 224.

## Usage

Needs an NVIDIA GPU, PyTorch with CUDA, TensorRT 11 (Python package plus the `trtexec` binary on `PATH`), `cuda-python`, `onnx`, `onnxruntime`, `numpy` and `pillow`.
The INT8 ResNet50 (`resnet50-v1-12-int8.onnx`) and the datasets are downloaded on first use with `curl`, so it must be installed. Run from `IntClassifier/`:

```bash
cd IntClassifier
python train.py cifar10
python train.py tiny-imagenet
python train.py path/to/dataset          # one subfolder per class, every 5th image held out as test
python predict.py weight/run-<time>/model.engine image.jpg
```

Options: `--epochs N` (default: train until early stop), `--stages` (default `1e-3:10,5e-4:5,2e-4:2,1e-4:1`: lr:patience pairs, lower lr and smaller patience each time the val loss stalls), `--eps` (Adam eps, default 1e-8), `--bs` (head batch size, default 1024), `--solver lbfgs` (linear head only; about 15x faster to train but 0.6-0.7 point less accurate in our runs), `--hidden H` (MLP head 2048-H-classes with ReLU instead of one linear layer; exported with an int8 hidden activation), `--seed` (default: random, saved in `metrics.json`), `--per-class N`, `--test-per-class N`.

Each run writes `weight/run-<time>/`: `model.onnx`, `model.engine`, `metrics.json` (accuracy, timings, GPU and RAM use) and `train.log`.
`predict.py` reads the class names from the `.onnx` next to the `.engine`.

Code in `IntClassifier/`: `train.py` (head training and the run), `model.py` (QLinear to QDQ conversion, engine runner, head graph), `dataset.py` (loaders), `predict.py`.
To add a dataset, write a loader in `dataset.py` and register it in `DATASETS`.

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

INT8 is about 2-3x faster at batch 1, 4x at batch 64 and 3.6x end to end, and the engine is 3.7x smaller, for about 0.25 points of accuracy. That gap has the same sign on both datasets but is of the size of the seed-to-seed spread, so treat it as roughly 0.2-0.3 points. GPU memory for the engine was about the same (about 350 MB, dominated by fixed buffers) and build time is similar. FP32 here is TensorRT's default float32 path, which may use TF32 tensor cores on this GPU.

## How it works

- **QLinear to QDQ.** The Model Zoo model uses uint8 activations with non-zero zero-points, which TensorRT rejects. `model.py` rewrites it as Quantize/Dequantize pairs with int8 symmetric activations. Weights and biases keep their exact values. Post-ReLU tensors lose about one bit, and the embeddings end up at cosine about 0.9 against the original model, which did not hurt accuracy here.
- **Head.** Adam on the int8 embeddings with staged learning rates (1e-3 down to 1e-4, patience shrinking each stage), 10% of train held out for validation, best-val-loss weights kept, early stopping. Weights are then quantized to int8 and appended to the backbone graph.
- **Confidence.** A 256-entry lookup table replaces `exp()`.

## Limits

- The engine is specific to the GPU and TensorRT version that built it.
- Input is resized to 256x256 with plain bilinear (no antialiasing, aspect ratio not kept) and center-cropped to 224. Fine for small or square images, it aliases when downscaling large photos. Max side 512, batch at most 64.
- The normalization at the start of the engine, the global-average-pool, the bias add and the LUT softmax are float operations on integer values. TensorRT has no `MatMulInteger` or `BitShift`.
- Accuracy numbers come from single runs and are not tuned.
