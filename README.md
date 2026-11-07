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

Frozen backbone + linear head, no hyperparameter tuning, one run (one seed) each. Test accuracy is measured through the final engine.

| Dataset | Classes | Train / val / test | Test accuracy |
|---|---|---|---|
| CIFAR-10 | 10 | 45k / 5k / 10k | 87.96% |
| Tiny ImageNet | 200 | 90k / 10k / 10k | 70.75% |

The int8 head and the engine match the float head to within 0.1 point, so quantizing the head costs no measurable accuracy.

Speed on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, CUDA 13:

| | |
|---|---|
| Engine latency, batch 1 / batch 64 | about 3 ms / 20 ms (about 3,000 img/s at batch 64) |
| Embeddings for 60k CIFAR images | about 20 s |
| Head training | about 1 min (CIFAR), about 35 s (Tiny ImageNet) |
| Engine build | about 2 min per run |
| GPU memory for the engine | about 350 MB |

Read the accuracy with care: the backbone was pre-trained on ImageNet-1k and Tiny ImageNet is a subset of it, so 70.75% is probably optimistic. CIFAR and Tiny ImageNet images are 32 and 64 px and get upscaled to 224.

## Usage

Needs an NVIDIA GPU, PyTorch with CUDA, TensorRT 11 (Python package plus the `trtexec` binary on `PATH`), `cuda-python`, `onnx`, `onnxruntime`, `numpy` and `pillow`.
The INT8 ResNet50 (`resnet50-v1-12-int8.onnx`) and the datasets are downloaded on first use. Run from `IntClassifier/`:

```bash
cd IntClassifier
python train.py cifar10
python train.py tiny-imagenet
python train.py path/to/dataset          # one subfolder per class, every 5th image held out as test
python predict.py weight/run-<time>/model.engine image.jpg
```

Options: `--epochs N` (default: train until early stop), `--patience P` (default 10), `--lr` (default 1e-3), `--eps` (Adam eps, default 1e-8), `--per-class N`, `--test-per-class N`.

Each run writes `weight/run-<time>/`: `model.onnx`, `model.engine`, `metrics.json` (accuracy, timings, GPU and RAM use) and `train.log`.
`predict.py` reads the class names from the `.onnx` next to the `.engine`.

To add a dataset, write a loader in `IntClassifier/dataset.py` and register it in `DATASETS`.

## How it works

- **QLinear to QDQ.** The Model Zoo model uses uint8 activations with non-zero zero-points, which TensorRT rejects. `train.py` rewrites it as Quantize/Dequantize pairs with int8 symmetric activations. Weights and biases keep their exact values. Post-ReLU tensors lose about one bit, and the embeddings end up at cosine about 0.9 against the original model, which did not hurt accuracy here.
- **Head.** Adam with a constant learning rate on the uint8 embeddings, 10% of train held out for validation, best-val-loss weights kept, early stopping. Weights are then quantized to int8 and appended to the backbone graph.
- **Confidence.** A 256-entry lookup table replaces `exp()`.

## Limits

- The engine is specific to the GPU and TensorRT version that built it.
- Input is resized to 256x256 with plain bilinear (no antialiasing, aspect ratio not kept) and center-cropped to 224. Fine for small or square images, it aliases when downscaling large photos. Max side 512, batch at most 64.
- The normalization at the start of the engine, the global-average-pool, the bias add and the LUT softmax are float operations on integer values. TensorRT has no `MatMulInteger` or `BitShift`.
- Accuracy numbers come from single runs and are not tuned.
