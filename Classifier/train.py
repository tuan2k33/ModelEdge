"""FP32 baseline of IntClassifier: python train.py [cifar10|tiny-imagenet|<dir with class subfolders>] [--stages lr:patience,...] [--bs N] [--seed N]
ResNet50 FP32 (ONNX Model Zoo) behind the same in-engine preprocessing -> TensorRT FP32 engine -> linear head trained in FP32 on its float embeddings ->
ONE FP32 engine (uint8 image -> label, confidence %). Output: weight/run-<time>/{model.onnx,model.engine,metrics.json,train.log}. Compare: compare.py."""
import argparse, json, resource, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from cuda.bindings import runtime as cudart

from dataset import load
from model import BATCH, CACHE, Engine, arr, backbone, build_model, compile_engine, infer


class Tee:
    """stdout -> terminal and run/train.log"""
    def __init__(self, *fs): self.fs = fs
    def write(self, s): [f.write(s) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


def train_head(X, y, Xv, yv, k, epochs, stages, bs=1024, eps=1e-8, seed=0, scale=255):
    """Linear head on X/scale (scale=255 for the int8 embeddings, any float scale for float ones) on the GPU: Adam, no weight decay. `stages` = [(lr, patience), ...]: train at lr until val loss is
    stale for `patience` epochs, then go back to the best weights and continue with the next (lower lr, smaller patience); the last stage ends the
    run (early stop). Returns metrics + the module holding the best-val-loss weights."""
    X, Xv = torch.as_tensor(X, device="cuda"), torch.as_tensor(Xv, device="cuda").float() / scale  # X stays int8, converted per batch
    y, yv = torch.as_tensor(y, device="cuda").long(), torch.as_tensor(yv, device="cuda").long()
    torch.manual_seed(seed)
    head = torch.nn.Linear(2048, k).cuda()
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=stages[0][0], eps=eps)
    best, bad, ep, stage = np.inf, 0, 0, 0
    while epochs is None or ep < epochs:
        ep += 1
        idx = torch.randperm(len(X), device="cuda")
        for i in range(0, len(X), bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(head(X[j].float() / scale), y[j]).backward()
            opt.step()
        with torch.no_grad():
            z = head(Xv)
            loss, acc = F.cross_entropy(z, yv).item(), (z.argmax(1) == yv).float().mean().item()
        print(f"epoch {ep} val_loss {loss:.4f} val_acc {acc:.4f} lr {stages[stage][0]:.0e}", flush=True)
        if loss < best - 1e-4:
            best, bad, info = loss, 0, dict(best_epoch=ep, val_loss=loss, val_acc=acc)
            state = {n: p.clone() for n, p in head.state_dict().items()}
        elif (bad := bad + 1) >= stages[stage][1]:
            stage, bad = stage + 1, 0
            if stage == len(stages):
                break
            head.load_state_dict(state)
            for g in opt.param_groups:
                g["lr"] = stages[stage][0]
    head.load_state_dict(state)
    torch.cuda.empty_cache()  # give the GPU back before trtexec builds the final engine
    return info | dict(epochs_run=ep), head


def bench(f, x, n):
    f(x)
    t0 = time.perf_counter()
    for _ in range(n):
        f(x)
    return (time.perf_counter() - t0) / n * 1e3


def gpu_mb():
    _, free, total = cudart.cudaMemGetInfo()
    return (total - free) / 2**20


def embed(eng, imgs):
    return np.concatenate([infer(eng, imgs[s:s + BATCH])[0] for s in range(0, len(imgs), BATCH)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="cifar10")
    ap.add_argument("--epochs", type=int, help="default: until the last stage stalls")
    ap.add_argument("--stages", default="1e-3:10,5e-4:5,2e-4:2,1e-4:1", help="lr:patience,...")
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
    ap.add_argument("--bs", type=int, default=1024, help="head batch size")
    ap.add_argument("--seed", type=int, help="val split + batch order; default: random")
    ap.add_argument("--per-class", type=int, help="default: all")
    ap.add_argument("--test-per-class", type=int)
    a = ap.parse_args()
    run = Path("weight") / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    sys.stdout = Tee(sys.stdout, open(run / "train.log", "w"))
    T0 = time.perf_counter(); cudart.cudaFree(0); gpu = dict(base=gpu_mb()); tm = {}
    tr, y, te, yt, classes = load(a.data, a.per_class, a.test_per_class)
    tm["load"] = time.perf_counter() - T0
    CACHE.mkdir(exist_ok=True)
    bb_onnx, bb_eng = CACHE / "backbone-fp32.onnx", CACHE / "backbone-fp32.engine"
    if not bb_eng.exists():
        t0 = time.perf_counter()
        onnx.save(backbone(), bb_onnx)
        compile_engine(bb_onnx, bb_eng)
        tm["build_backbone_engine_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    eng = Engine(bb_eng)
    E, Et = embed(eng, tr), embed(eng, te)
    tm["embed_s"] = time.perf_counter() - t0
    tm["embed_ms_per_img"] = tm["embed_s"] / (len(tr) + len(te)) * 1e3
    del eng
    scale = 2 * float(np.percentile(E[:20000], 99.9))  # puts the float embeddings in ~[0, 0.5], like the int8 ones divided by 255
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    print(f"seed {seed}")
    perm = np.random.default_rng(seed).permutation(len(E))
    nv = len(E) // 10
    v, t = perm[:nv], perm[nv:]
    t0 = time.perf_counter()
    info, head_t = train_head(E[t], y[t], E[v], y[v], len(classes), a.epochs, [(float(l), int(p)) for l, p in (x.split(":") for x in a.stages.split(","))],
                              bs=a.bs, eps=a.eps, seed=seed, scale=scale)
    tm["head_train_s"], tm["head_ms_per_epoch"] = time.perf_counter() - t0, (time.perf_counter() - t0) / info["epochs_run"] * 1e3
    with torch.no_grad():
        acc_f = float((head_t(torch.as_tensor(Et, device="cuda").float() / scale).argmax(1).cpu().numpy() == yt).mean())
    onnx.save(build_model(head_t.weight.detach().T.cpu().numpy(), head_t.bias.detach().cpu().numpy(), scale, classes), run / "model.onnx")
    t0 = time.perf_counter()
    compile_engine(run / "model.onnx", run / "model.engine")
    tm["compile_model_engine_s"] = time.perf_counter() - t0
    g0 = gpu_mb(); fe = Engine(run / "model.engine")
    x64 = np.stack([arr(i) for i in te[:BATCH]]); fe(x64); gpu["model_engine"] = gpu_mb() - g0
    tm["engine_batch1_ms"], tm["engine_batch64_ms"] = bench(fe, x64[:1], 200), bench(fe, x64, 30)
    t0 = time.perf_counter()
    acc_e = float(np.mean(np.concatenate([infer(fe, te[s:s + BATCH])[0] for s in range(0, len(te), BATCH)]) == yt))
    tm["eval_img_per_s"] = len(te) / (time.perf_counter() - t0)
    assert abs(acc_e - acc_f) < 0.01, f"engine accuracy {acc_e} != float head {acc_f}"
    m = info | dict(test_acc_engine=acc_e, test_acc_float_head=acc_f, train=len(t), val=nv, test=len(Et), classes=len(classes), stages=a.stages, eps=a.eps,
                    bs=a.bs, seed=seed, emb_scale=scale, engine_mb=round((run / "model.engine").stat().st_size / 2**20, 1), gpu_mb=gpu,
                    timing={k: round(v, 3) for k, v in tm.items()}, host_peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(run, json.dumps(m, indent=1))
