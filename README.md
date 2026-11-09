# ModelEdge

A small INT8 image classifier built from a frozen, pre-quantized ResNet50 and a small MLP head you train in under a minute.
The result is **one TensorRT engine**: raw `uint8` image in, label and per-class logits out, all on the GPU.

```
uint8 image [N,H,W,3]
  -> resize 224 / normalize                      (inside the engine)
  -> ResNet50, INT8 QDQ                         (frozen, from the ONNX Model Zoo)
  -> int8 global-average-pool tensor [N,2048]
  -> int8 MLP head 2048-1024-classes (trained here)  (int8 x int8 GEMMs)
  -> label, logits [N,classes]                  (confidence of a class = sigmoid(logit), computed by the caller)
```

Only the head is trained, on cached embeddings, so training a new dataset takes minutes, not hours.

## Results

Frozen backbone + MLP head (2048-1024-classes, ReLU, int8 hidden activation), default settings, no hyperparameter tuning, one sigmoid (BCE) output per class, one run per dataset (seed 1). Test accuracy is measured through the final engine.

| Dataset | Classes | Train / val / test | Train acc | Val acc | Test acc |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 45k / 5k / 10k | 95.4% | 88.9% | **89.8%** |
| CIFAR-100 | 100 | 45k / 5k / 10k | 85.3% | 70.6% | **70.0%** |
| MNIST | 10 | 54k / 6k / 10k | 99.8% | 98.3% | **98.5%** |
| Tiny ImageNet | 200 | 90k / 10k / 10k | 83.5% | 72.9% | **72.5%** |
| Intel Image Classification (Kaggle) | 6 | 12.6k / 1.4k / 3k | 95.4% | 93.4% | **92.2%** |
| EuroSAT RGB | 10 | 19.4k / 2.2k / 5.4k | 98.6% | 95.6% | **95.2%** |

Speed on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, CUDA 13:

| | |
|---|---|
| Engine latency, batch 1 / batch 64 | about 2-3 ms / 17.5 ms (about 3,600 img/s at batch 64) |
| Embeddings for 60k CIFAR images | about 20 s (once, then cached) |
| Head training | 6-21 s (14 s CIFAR-10, 21 s Tiny ImageNet) |
| Engine build | about 100 s per run |
| GPU memory for the engine | about 350 MB |

## Usage

Needs an NVIDIA GPU, PyTorch with CUDA, TensorRT 11 (Python package plus the `trtexec` binary on `PATH`), `cuda-python`, `onnx`, `onnxruntime`, `numpy` and `pillow`.
The INT8 ResNet50 (`resnet50-v1-12-int8.onnx`) and the datasets are downloaded on first use with `curl`, so it must be installed. Datasets live in `IntClassifier/data/`. Run from `IntClassifier/`:

```bash
cd IntClassifier
python train.py cifar10
python train.py tiny-imagenet          # also: cifar100, mnist (downloaded on first use)
python train.py intel                   # your own folder: data/intel/{train,test}/<class>/*.jpg (or any path)
python predict.py weight/cifar10/model.engine image.jpg
```

Options: `--epochs N` (default: train until early stop), `--stages` (default `1e-3:10,5e-4:8,2e-4:6,1e-4:4`: lr:patience pairs, lower lr and smaller patience each time the val loss stalls), `--eps` (Adam eps, default 1e-8), `--bs` (head batch size, default 1024), `--hidden H` (MLP head 2048-H-classes with ReLU, default 1024, exported with an int8 hidden activation; 0 = one linear layer), `--seed` (default: random, saved in `metrics.json`), `--per-class N`, `--test-per-class N`.

Each run writes `weight/<dataset>/` (replacing the previous run of that dataset): `model.onnx`, `model.engine`, `metrics.json` (dataset, train/val/test accuracy and loss, timings, GPU and RAM use) and `train.log`.
`predict.py` reads the class names from the `.onnx` next to the `.engine` and turns the top logit into a confidence with a sigmoid.

Code in `IntClassifier/`: `train.py` (head training and the run), `model.py` (QLinear to QDQ conversion, engine runner, head graph), `dataset.py` (loaders), `predict.py`.
Your own data: put it in `<root>/train/<class>/` and `<root>/test/<class>/` (same class folders in both) and pass `<root>` (a bare name is looked up in `data/`). To add a downloadable dataset, write a loader in `dataset.py` and register it in `DATASETS`. The Intel and EuroSAT numbers above come from the Kaggle downloads arranged this way (Intel: `seg_train` and `seg_test` renamed to `train` and `test`; EuroSAT: every 5th image of each class moved to `test`).

## How it works

- **QLinear to QDQ.** The Model Zoo model uses uint8 activations with non-zero zero-points, which TensorRT rejects. `model.py` rewrites it as Quantize/Dequantize pairs with int8 symmetric activations. Weights and biases keep their exact values. Post-ReLU tensors lose about one bit, and the embeddings end up at cosine about 0.9 against the original model, which did not hurt accuracy here.
- **Head.** MLP 2048-1024-classes: one sigmoid per class trained with binary cross-entropy against the one-hot label, Adam on the int8 embeddings with staged learning rates (1e-3 down to 1e-4, patience shrinking each stage), 10% of train held out for validation, best-val-loss weights kept, early stopping. Weights are then quantized to int8 and appended to the backbone graph. The hidden ReLU output is requantized to int8 with scale 127/`hidden_max`, where `hidden_max` is the 99.9th percentile of that output on 4096 train rows at the best checkpoint (the top 0.1% is clipped).
- **Output.** The label is the argmax of the logits and the engine also returns the logits. Sigmoid is monotonic, so the top logit gives the top confidence; the classes are independent, so confidences do not sum to 1. The engine contains no `exp()`.

## Limits

- The engine is specific to the GPU and TensorRT version that built it.
- Input is resized straight to 224x224 with plain bilinear (no antialiasing, aspect ratio not kept), no crop, in training and inference. Fine for small or square images, it aliases when downscaling large photos. The engine takes sides up to 512, bigger images are shrunk to 512 on the CPU first (antialiased). Batch at most 64.
- The normalization at the start of the engine, the global-average-pool, the bias add and the final scaling of the logits are float operations on integer values. TensorRT has no `MatMulInteger` or `BitShift`.
- Accuracy numbers come from single runs and are not tuned.
