# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ModelEdge turns pre-quantized / pretrained models into **single TensorRT engines** where pre-processing runs inside the engine (raw `uint8` image in). Independent sub-projects. Read the README (results, options, limits) before changing one that has it:

- `IntClassifier/` — frozen INT8 ResNet50 (ONNX Model Zoo, QLinear → QDQ) + small MLP head trained on cached embeddings.
- `Detector/` — YOLO26 FP16 engine with the letterbox built into the graph (INT8 was tried and removed; see its README).
- `FaceID/` — ArcFace ResNet100 INT8 → QDQ engine, evaluated on LFW / CFP-FP / AgeDB-30 (`eval.py` only, no train).
- `Motion/` — background model for fixed cameras as a stateful TensorRT engine (no trained weights); the module docstrings in `model.py`, `run.py`, `build.py` document inputs, state layout and options. No README.
- `peoplenet/` — scratch experiments (NGC PeopleNet INT8 CNN, `run.py`, `patch_qdq.py`), git-ignored, not part of the shipped code.

## Running

No build, lint or test suite. Each sub-project is a set of scripts run **from inside its own directory** (they import siblings like `model`, `dataset` and use relative `data/`, `cache/`, `weight/`). Use the project venv (`.venv` → `~/.dev/venvs/ModelEdge`, py3.14). Needs an NVIDIA GPU, TensorRT 11 with `trtexec` on `PATH`, and `curl` (datasets and Model Zoo weights download on first use).

```bash
cd IntClassifier && python train.py cifar10        # also cifar100, mnist, emnist, tiny-imagenet, or a folder with train/<class>/ and test/<class>/
cd IntClassifier && python predict.py weight/cifar10/model.engine image.jpg
cd Detector      && python train.py --epochs 0 --name coco     # export only; or pass a data.yaml to fine-tune
cd Detector      && python predict.py weight/coco/model.engine image.jpg --out result.jpg
cd Detector      && python eval.py weight/coco/model.engine --limit 200   # COCO val2017 mAP; --start/--limit for subsets
cd FaceID        && python eval.py --sets lfw --pairs 600
cd Motion        && python run.py in.mp4 out.mp4 [--detector ../Detector/weight/coco/model.onnx]   # background model; build.py makes the fused multi-camera engine
```

"Single test" equivalents: `--limit N` (Detector eval), `--pairs N` (FaceID eval), `--per-class N` / `--test-per-class N` (IntClassifier train).

## Architecture notes

- Each sub-project has `model.py` holding the ONNX graph construction (QDQ rewrite or letterbox/normalize wrapper), `compile_engine` (shells out to `trtexec`) and an `Engine` runner using TensorRT + torch CUDA tensors. Scripts are thin around it.
- QLinear → QDQ conversion is needed because TensorRT rejects uint8 activations with non-zero zero-points; activations become int8 symmetric. Weights/biases keep exact values.
- IntClassifier caches backbone embeddings in `cache/<dataset>-pool-*.npz`, trains the head on the GPU on those, then appends the int8-quantized head to the backbone graph and builds the engine. Output is label + logits; confidence = sigmoid(logit) computed by the caller (no `exp()` in the engine).
- Outputs go to `weight/<name>/` (IntClassifier: `model.onnx`, `model.engine`, `metrics.json`, `train.log`; Detector: also `weights/best.pt`) and a rerun replaces the previous one. `predict.py` reads class names from the `.onnx` next to the `.engine`.
- Motion keeps its memory outside the engine: `state` [B,8,180,320] goes in and `state_out` comes out, one row per camera; state stays FP32 on purpose (FP16 loses the small updates). Raise `VER` in `Motion/model.py` when its graph changes (part of the engine cache name).
- Engines are tied to the GPU + TensorRT version that built them; `data/`, `cache/`, `weight/`, `__pycache__/` are git-ignored per sub-project.

## Conventions

- Test/run precision order: FP16 >> BF16 >> FP32; no autocast on FP32 weights.
- Videos written by scripts must be H264.
- Keep inputs/outputs inside the project, not `/tmp`. No `Co-Authored-By` or AI trailers in commits; AI disclosure goes in the README ("AI Disclosure" section).
- Project decisions that cannot be read from code are kept in `/mnt/c/Dev/.claude/memory/project_ModelEdge.md`.
