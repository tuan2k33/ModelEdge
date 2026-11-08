"""python train.py [cifar10|tiny-imagenet|<dir with class subfolders>] [--epochs N] [--stages lr:patience,...] [--per-class N]  (loaders: dataset.py, model: model.py)
Backbone: ONNX Model Zoo ResNet50 INT8 QDQ -> TensorRT INT8 engine on the GPU (cached). Head: linear softmax on the int8 embeddings, torch on the GPU
(Adam, staged lr with shrinking patience, 10% val split, best-val-loss weights kept; no --epochs = until the last stage stalls), quantized to int8 and appended to the
backbone with a LUT softmax. Output: weight/run-<time>/{model.onnx,model.engine,metrics.json,train.log}: uint8 image -> label, confidence %."""
import argparse, json, resource, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from cuda.bindings import runtime as cudart

from dataset import load
from model import BATCH, CACHE, Engine, arr, build_engine, build_model, compile_engine, infer, quantize


class Tee:
    """stdout -> terminal and run/train.log"""
    def __init__(self, *fs): self.fs = fs
    def write(self, s): [f.write(s) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


def train_head(X, y, Xv, yv, k, epochs, stages, bs=1024, eps=1e-8, seed=0, hidden=0, scale=255):
    """Linear head on X/scale (scale=255 for the int8 embeddings, any float scale for float ones) on the GPU: Adam, no weight decay. `stages` = [(lr, patience), ...]: train at lr until val loss is
    stale for `patience` epochs, then go back to the best weights and continue with the next (lower lr, smaller patience); the last stage ends the
    run (early stop). hidden=H > 0 puts Linear(2048, H) + ReLU in front; then the metrics also carry `hidden_max`, the 99.9th percentile of the ReLU
    output measured at the best checkpoint (its int8 scale at export). Returns metrics + the module holding the best-val-loss weights."""
    X, Xv = torch.as_tensor(X, device="cuda"), torch.as_tensor(Xv, device="cuda").float() / scale  # X stays int8, converted per batch
    y, yv = torch.as_tensor(y, device="cuda").long(), torch.as_tensor(yv, device="cuda").long()
    torch.manual_seed(seed)
    if hidden:
        head = torch.nn.Sequential(torch.nn.Linear(2048, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k)).cuda()
    else:
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
            if hidden:
                with torch.no_grad():
                    info["hidden_max"] = torch.quantile(head[1](head[0](X[:4096].float() / scale)).flatten(), 0.999).item()
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


def fit_lbfgs(X, y, Xv, yv, k, patience=3, chunk=10, max_rounds=100, seed=0):
    """Linear head only (convex), full-batch L-BFGS on the GPU: no lr to tune. One "epoch" here = `chunk` L-BFGS iterations; keeps the
    best-val-loss weights and stops after `patience` epochs without gain. Same return value as train_head."""
    X, Xv = torch.as_tensor(X, device="cuda").float() / 255, torch.as_tensor(Xv, device="cuda").float() / 255
    y, yv = torch.as_tensor(y, device="cuda").long(), torch.as_tensor(yv, device="cuda").long()
    head = torch.nn.Linear(2048, k).cuda()
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), lr=1, max_iter=chunk, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(X), y)
        loss.backward()
        return loss

    best, bad = np.inf, 0
    for ep in range(1, max_rounds + 1):
        opt.step(closure)
        with torch.no_grad():
            z = head(Xv)
            loss, acc = F.cross_entropy(z, yv).item(), (z.argmax(1) == yv).float().mean().item()
        print(f"epoch {ep} val_loss {loss:.4f} val_acc {acc:.4f} lbfgs_iters {ep * chunk}", flush=True)
        if loss < best - 1e-4:
            best, bad, info = loss, 0, dict(best_epoch=ep, val_loss=loss, val_acc=acc)
            state = {n: p.clone() for n, p in head.state_dict().items()}
        elif (bad := bad + 1) >= patience:
            break
    head.load_state_dict(state)
    del X, Xv
    torch.cuda.empty_cache()
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
    ap.add_argument("--epochs", type=int, help="default: until early stop")
    ap.add_argument("--stages", default="1e-3:10,5e-4:5,2e-4:2,1e-4:1", help="lr:patience,... ; each stage ends after `patience` epochs without val-loss gain")
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
    ap.add_argument("--bs", type=int, default=1024, help="head batch size")
    ap.add_argument("--solver", choices=["adam", "lbfgs"], default="adam", help="lbfgs: linear head only, no lr/stages")
    ap.add_argument("--hidden", type=int, default=0, help="hidden layer width (experiment, float only); 0 = linear head")
    ap.add_argument("--seed", type=int, help="val split + batch order; default: random (printed and saved in metrics.json)")
    ap.add_argument("--per-class", type=int, help="default: all")
    ap.add_argument("--test-per-class", type=int)
    a = ap.parse_args()
    run = Path("weight") / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    sys.stdout = Tee(sys.stdout, open(run / "train.log", "w"))
    feats = CACHE / f"{Path(a.data).name}-pool-{a.per_class}-{a.test_per_class}.npz"
    T0 = time.perf_counter(); cudart.cudaFree(0); gpu = dict(base=gpu_mb())
    tr, y, te, yt, classes = load(a.data, a.per_class, a.test_per_class)
    tm = dict(load=time.perf_counter() - T0)
    t0 = time.perf_counter()
    eng_path = build_engine(te[:8])
    tm["build_backbone_engine_s"] = time.perf_counter() - t0
    if feats.exists():
        f = np.load(feats)
        X, Xt = f["X"], f["Xt"]
    else:
        t0 = time.perf_counter()
        eng = Engine(eng_path)
        infer(eng, te[:BATCH]); gpu["backbone_engine"] = gpu_mb() - gpu["base"]
        E, Et = embed(eng, tr), embed(eng, te)
        assert E.min() >= 0 and Et.min() >= 0  # post-ReLU, the head treats them as unsigned
        X, Xt = E, Et
        np.savez(feats, X=X, Xt=Xt)
        n = len(tr) + len(te)
        tm["embed_s"] = time.perf_counter() - t0
        tm["embed_ms_per_img"] = tm["embed_s"] / n * 1e3
        del eng
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    print(f"seed {seed}")
    perm = np.random.default_rng(seed).permutation(len(X))
    nv = len(X) // 10
    v, t = perm[:nv], perm[nv:]
    t0 = time.perf_counter()
    if a.solver == "lbfgs":
        assert not a.hidden, "lbfgs is for the linear head"
        info, head_t = fit_lbfgs(X[t], y[t], X[v], y[v], len(classes))
    else:
        info, head_t = train_head(X[t], y[t], X[v], y[v], len(classes), a.epochs, [(float(l), int(p)) for l, p in (x.split(":") for x in a.stages.split(","))],
                                  bs=a.bs, eps=a.eps, seed=seed, hidden=a.hidden)
    tm["head_train_s"], tm["head_ms_per_epoch"] = time.perf_counter() - t0, (time.perf_counter() - t0) / info["epochs_run"] * 1e3
    with torch.no_grad():
        acc_f = float((head_t(torch.as_tensor(Xt, device="cuda").float() / 255).argmax(1).cpu().numpy() == yt).mean())
    if a.hidden:
        l1, l2 = head_t[0], head_t[2]
        head = quantize(l1.weight.detach().T.cpu().numpy(), l1.bias.detach().cpu().numpy(), l2.weight.detach().T.cpu().numpy(),
                        l2.bias.detach().cpu().numpy(), info["hidden_max"])
    else:
        head = quantize(head_t.weight.detach().T.cpu().numpy(), head_t.bias.detach().cpu().numpy())
    with torch.no_grad():  # integer logits in float64 on the GPU: exact (|sum| < 2^53)
        g = lambda k: torch.as_tensor(head[k], device="cuda").double()
        z = torch.as_tensor(Xt, device="cuda").double()
        if a.hidden:
            z = torch.round(torch.relu(z @ g("W1") + g("b1")) / head["m1"]).clamp(0, 127)
        acc = float(((z @ g("W") + g("b")).argmax(1).cpu().numpy() == yt).mean())

    onnx.save(build_model(head, classes), run / "model.onnx")
    t0 = time.perf_counter()
    compile_engine(run / "model.onnx", run / "model.engine")
    tm["compile_model_engine_s"] = time.perf_counter() - t0
    g0 = gpu_mb(); fe = Engine(run / "model.engine")
    x64 = np.stack([arr(i) for i in te[:BATCH]]); fe(x64); gpu["model_engine"] = gpu_mb() - g0
    tm["engine_batch1_ms"], tm["engine_batch64_ms"] = bench(fe, x64[:1], 200), bench(fe, x64, 30)
    t0 = time.perf_counter()
    acc_e = float(np.mean(np.concatenate([infer(fe, te[s:s + BATCH])[0] for s in range(0, len(te), BATCH)]) == yt))
    tm["eval_img_per_s"] = len(te) / (time.perf_counter() - t0)
    assert abs(acc_e - acc) < 0.01, f"engine accuracy {acc_e} != numpy int8 head {acc}"
    m = info | dict(test_acc_engine=acc_e, test_acc_int8_head=acc, test_acc_float_head=acc_f, train=len(t), val=nv, test=len(Xt),
                    classes=len(classes), hidden=a.hidden, stages=a.stages, eps=a.eps, bs=a.bs, solver=a.solver, seed=seed, gpu_mb=gpu, timing={k: round(v, 3) for k, v in tm.items()},
                    host_peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(run, json.dumps(m, indent=1))
