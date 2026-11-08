"""python compare.py <cifar10|tiny-imagenet> <int8 run dir> <fp32 run dir>
Both model.engine files on the same test images, measured back to back in one process: accuracy, speed, engine size, GPU memory, build time."""
import json, sys, time
from pathlib import Path

import numpy as np
from cuda.bindings import runtime as cudart

from dataset import load
from model import BATCH, Engine, arr, infer
from train import bench, gpu_mb

data, runs = sys.argv[1], {"int8": Path(sys.argv[2]), "fp32": Path(sys.argv[3])}
cudart.cudaFree(0)
_, _, te, yt, _ = load(data, 1, None)  # train capped at 1 image per class: only the test set matters here
x64 = np.stack([arr(i) for i in te[:BATCH]])
rows = {}
for name, r in runs.items():
    g0 = gpu_mb(); eng = Engine(r / "model.engine"); infer(eng, te[:BATCH]); mem = gpu_mb() - g0
    t0 = time.perf_counter()
    pred = np.concatenate([infer(eng, te[s:s + BATCH])[0] for s in range(0, len(te), BATCH)])
    rows[name] = dict(acc=100 * float((pred == yt).mean()), engine_mb=(r / "model.engine").stat().st_size / 2**20, gpu_mb=mem,
                      b1=bench(eng, x64[:1], 300), b64=bench(eng, x64, 50), e2e=len(te) / (time.perf_counter() - t0),
                      build=json.load(open(r / "metrics.json"))["timing"]["compile_model_engine_s"])
i, f = rows["int8"], rows["fp32"]
print(f"{data}: {len(te)} test images\n{'':28s}{'INT8':>10s}{'FP32':>10s}{'FP32/INT8':>12s}")
for label, k, fmt in [("test accuracy (%)", "acc", "%.2f"), ("engine batch 1 (ms)", "b1", "%.2f"), ("engine batch 64 (ms)", "b64", "%.1f"),
                      ("end-to-end (img/s)", "e2e", "%.0f"), ("engine file (MB)", "engine_mb", "%.0f"), ("GPU memory (MB)", "gpu_mb", "%.0f"), ("engine build (s)", "build", "%.0f")]:
    print(f"{label:28s}{fmt % i[k]:>10s}{fmt % f[k]:>10s}{f[k] / i[k]:>11.2f}x")
