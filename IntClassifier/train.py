"""python train.py [cifar10|cifar100|mnist|emnist|tiny-imagenet|<dataset dir>] [--backbone resnet50|convnext] [--epochs N] [--stages lr:patience,...] [--per-class N]  (loaders: dataset.py, model: model.py)
A dataset dir has train/ and test/ class subfolders; a bare name is looked up in data/. Backbone: an INT8 Q/DQ ONNX from the Hub (model.BACKBONES, built by export.py) -> TensorRT engine on the GPU (cached).
Head: MLP on the int8 embeddings, one sigmoid per class (BCE), torch on the GPU (Adam, staged lr with shrinking patience, 10% val split, best-val-loss weights kept;
no --epochs = until the last stage stalls), quantized to int8 and appended to the backbone. Output: weight/<dataset>[-<backbone> unless resnet50]/{model.onnx,model.engine,metrics.json,train.log}: uint8 image -> label, logits (the confidence of a class is sigmoid(logit), see predict.py)."""
import argparse, json, resource, shutil, sys, time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from cuda.bindings import runtime as cudart

from dataset import load
from model import BACKBONES, BATCH, CACHE, Engine, arr, build_engine, build_model, compile_engine, infer, quantize


class Tee:
    """stdout -> terminal and run/train.log"""
    def __init__(self, *fs): self.fs = fs
    def write(self, s): [f.write(s) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


def head_loss(z, y, reduction="mean"):
    """One sigmoid per class against the one-hot target, summed over the classes."""
    l = F.binary_cross_entropy_with_logits(z, F.one_hot(y, z.shape[1]).to(z.dtype), reduction="none").sum(1)
    return l.mean() if reduction == "mean" else l.sum()


@torch.no_grad()
def train_stats(head, X, y, scale):
    """Loss and accuracy of the (best) head on the whole train set, in chunks."""
    loss = acc = 0
    for i in range(0, len(X), 8192):
        z = head(X[i:i + 8192].float() / scale)
        loss += head_loss(z, y[i:i + 8192], "sum").item(); acc += (z.argmax(1) == y[i:i + 8192]).sum().item()
    return dict(train_loss=loss / len(X), train_acc=acc / len(X))


def train_head(X, y, Xv, yv, k, epochs, stages, bs=1024, eps=1e-8, seed=0, hidden=0, scale=255):
    """Linear head on X/scale (scale=255 for the int8 embeddings, any float scale for float ones) on the GPU: Adam, no weight decay. `stages` = [(lr, patience), ...]: train at lr until val loss is
    stale for `patience` epochs, then go back to the best weights and continue with the next (lower lr, smaller patience); the last stage ends the
    run (early stop). hidden=H > 0 puts Linear(D, H) + ReLU in front (D = embedding size); then the metrics also carry `hidden_max`, the 99.9th percentile of the ReLU
    output measured at the best checkpoint (its int8 scale at export). Returns metrics + the module holding the best-val-loss weights."""
    X, Xv = torch.as_tensor(X, device="cuda"), torch.as_tensor(Xv, device="cuda").float() / scale  # X stays int8, converted per batch
    y, yv = torch.as_tensor(y, device="cuda").long(), torch.as_tensor(yv, device="cuda").long()
    torch.manual_seed(seed)
    if hidden:
        head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k)).cuda()
    else:
        head = torch.nn.Linear(X.shape[1], k).cuda()
        torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    torch.nn.init.constant_((head[2] if hidden else head).bias, -float(np.log(k - 1)))  # start at the prior of "none of the classes", or the k-1 negatives dominate the first steps
    opt = torch.optim.Adam(head.parameters(), lr=stages[0][0], eps=eps)
    best, bad, ep, stage = np.inf, 0, 0, 0
    while epochs is None or ep < epochs:
        ep += 1
        idx = torch.randperm(len(X), device="cuda")
        for i in range(0, len(X), bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            head_loss(head(X[j].float() / scale), y[j]).backward()
            opt.step()
        with torch.no_grad():
            z = head(Xv)
            loss, acc = head_loss(z, yv).item(), (z.argmax(1) == yv).float().mean().item()
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
    info |= train_stats(head, X, y, scale)
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
    return torch.cat([infer(eng, imgs[s:s + BATCH], gpu=True)[0] for s in range(0, len(imgs), BATCH)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="cifar10")
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, help="default: until early stop")
    ap.add_argument("--stages", default="1e-3:10,5e-4:8,2e-4:6,1e-4:4", help="lr:patience,... ; each stage ends after `patience` epochs without val-loss gain")
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
    ap.add_argument("--bs", type=int, default=1024, help="head batch size")
    ap.add_argument("--hidden", type=int, default=1024, help="hidden layer width of the MLP head; 0 = linear head")
    ap.add_argument("--seed", type=int, help="val split + batch order; default: random (printed and saved in metrics.json)")
    ap.add_argument("--per-class", type=int, help="default: all")
    ap.add_argument("--test-per-class", type=int)
    a = ap.parse_args()
    run = Path("weight") / (Path(a.data).name + ("" if a.backbone == "resnet50" else f"-{a.backbone}"))  # a new run replaces the previous one of the same dataset + backbone
    shutil.rmtree(run, ignore_errors=True); run.mkdir(parents=True)
    sys.stdout = Tee(sys.stdout, open(run / "train.log", "w"))
    feats = CACHE / f"{Path(a.data).name}-{a.backbone}-{a.per_class}-{a.test_per_class}.npz"
    T0 = time.perf_counter(); cudart.cudaFree(0); gpu = dict(base=gpu_mb())
    tr, y, te, yt, classes = load(a.data, a.per_class, a.test_per_class)
    tm = dict(load=time.perf_counter() - T0)
    t0 = time.perf_counter()
    src, eng_path = build_engine(a.backbone, te[:8])
    tm["build_backbone_engine_s"] = time.perf_counter() - t0
    if feats.exists():
        f = np.load(feats)
        X, Xt = torch.as_tensor(f["X"], device="cuda"), torch.as_tensor(f["Xt"], device="cuda")
    else:
        t0 = time.perf_counter()
        eng = Engine(eng_path)
        infer(eng, te[:BATCH]); gpu["backbone_engine"] = gpu_mb() - gpu["base"]
        E, Et = embed(eng, tr), embed(eng, te)
        X, Xt = E, Et
        np.savez(feats, X=X.cpu().numpy(), Xt=Xt.cpu().numpy())
        n = len(tr) + len(te)
        tm["embed_s"] = time.perf_counter() - t0
        tm["embed_ms_per_img"] = tm["embed_s"] / n * 1e3
        del eng
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    print(f"seed {seed}")
    perm = torch.randperm(len(X), device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    yg, ytg = torch.as_tensor(y, device="cuda"), torch.as_tensor(yt, device="cuda")
    nv = len(X) // 10
    v, t = perm[:nv], perm[nv:]
    t0 = time.perf_counter()
    info, head_t = train_head(X[t], yg[t], X[v], yg[v], len(classes), a.epochs, [(float(l), int(p)) for l, p in (x.split(":") for x in a.stages.split(","))],
                              bs=a.bs, eps=a.eps, seed=seed, hidden=a.hidden)
    tm["head_train_s"], tm["head_ms_per_epoch"] = time.perf_counter() - t0, (time.perf_counter() - t0) / info["epochs_run"] * 1e3
    if a.hidden:
        l1, l2 = head_t[0], head_t[2]
        head = quantize(l1.weight.detach().T, l1.bias.detach(), l2.weight.detach().T, l2.bias.detach(), info["hidden_max"])
    else:
        head = quantize(head_t.weight.detach().T, head_t.bias.detach())
    with torch.no_grad():  # integer logits in float64 on the GPU: exact (|sum| < 2^53)
        g = lambda k: torch.as_tensor(head[k], device="cuda").double()
        z = Xt.double()
        if a.hidden:
            z = torch.round(torch.relu(z @ g("W1") + g("b1")) / head["m1"]).clamp(0, 127)
        z = z @ g("W") + g("b")
        acc = (z.argmax(1) == ytg).double().mean().item()

    onnx.save(build_model(head, classes, src), run / "model.onnx")
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
    lg = infer(fe, te[:BATCH])[1]
    assert np.allclose(lg, (z[:BATCH] * head["S"]).cpu().numpy(), atol=1e-3 * np.abs(lg).max()), "engine logits != int8 head logits"
    m = dict(data=a.data, backbone=a.backbone) | info | dict(test_acc=acc_e, train=len(t), val=nv, test=len(Xt),
                    classes=len(classes), hidden=a.hidden, stages=a.stages, eps=a.eps, bs=a.bs, seed=seed, gpu_mb=gpu, timing={k: round(v, 3) for k, v in tm.items()},
                    host_peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(run, json.dumps(m, indent=1))
