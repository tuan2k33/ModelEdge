"""python train.py [cifar10|tiny-imagenet|<dir with class subfolders>] [--epochs N] [--patience P] [--per-class N]  (loaders: dataset.py)
Backbone: ONNX Model Zoo ResNet50 INT8 (QLinear) -> QDQ (int8, symmetric) -> TensorRT INT8 engine on the GPU (cached).
Head: linear softmax on the uint8 embeddings, torch on the GPU (Adam, constant lr, 10% val split, best-val-loss weights kept,
early stop; no --epochs = until early stop), quantized to int8 and appended to the backbone with a LUT softmax. Output: weight/run-<time>/{model.onnx,model.engine,metrics.json}:
uint8 image -> label, confidence %."""
import argparse, json, resource, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
import torch.nn.functional as F
from cuda.bindings import runtime as cudart
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from dataset import load

ZOO = "resnet50-v1-12-int8.onnx"
ZOO_URL = "https://huggingface.co/onnxmodelzoo/resnet50-v1-12-int8/resolve/main/" + ZOO
POOL = "resnetv17_pool1_fwd_quantized"  # uint8 [N,2048,1,1] after global average pool
LUT_STEP = 16  # LUT index = logit gap * 16; 256 entries cover gaps up to 16
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
CACHE = Path("cache")
MAXHW = 512  # largest input side the engine accepts (batch x side^2 x float32 temp must fit the GPU)
BATCH = 64


def arr(img):
    return np.asarray(img.convert("RGB"))


def zoo_cut():
    """The zoo model up to (and including) the global-avg-pool; output is the uint8 pool tensor."""
    if not Path(ZOO).exists():
        urllib.request.urlretrieve(ZOO_URL, ZOO)
    m = onnx.load(ZOO)
    g = m.graph
    del g.node[next(i for i, n in enumerate(g.node) if POOL in n.output) + 1:]
    del g.output[:]
    g.output.append(h.make_tensor_value_info(POOL, T.UINT8, [None, 2048, 1, 1]))
    return m


def qdq_backbone():
    """QLinear (uint8 acts) -> QDQ (int8, zero-point 0, what TensorRT accepts). Weights/bias keep their exact values.
    Activation scale s -> 2s for zp 0 (post-ReLU, loses 1 bit) or max(zp, 255-zp)*s/127 otherwise (~lossless). TensorRT 11
    builds uint8 Q/DQ but runs it wrong (engine != ORT), hence int8. Embeddings end up cos ~0.9 vs the original model.
    A zp-0 output was an implicit ReLU (uint8 saturation), so it becomes an explicit Relu. Input: uint8 NHWC image,
    resized to 256x256 (squashed, bilinear, no antialias: shortcut, fine for small or square images; downscaling big photos aliases),
    center-cropped to 224 and normalized by float nodes, all inside the engine. Output: the int8 global-avg-pool tensor [N,2048] as is (>= 0, post-ReLU)."""
    z = zoo_cut()
    init = {i.name: i for i in z.graph.initializer}
    arr = {k: nh.to_array(v) for k, v in init.items()}
    nodes, inits, par, dqd = [], [], {}, set()

    def reg(t, s, zp):
        s, zp = float(s), int(zp)
        par[t] = (s, zp)
        inits.extend([nh.from_array(np.array(2 * s if zp == 0 else max(zp, 255 - zp) * s / 127, np.float32), t + "_s"),
                      nh.from_array(np.array(0, np.int8), t + "_z")])

    def q(f, t, relu=False):
        if relu:
            nodes.append(h.make_node("Relu", [f], [f + "_r"])); f += "_r"
        nodes.append(h.make_node("QuantizeLinear", [f, t + "_s", t + "_z"], [t + "_q"]))

    def dq(t, s=None, zp=None):
        assert s is None or par[t] == (float(arr[s]), int(arr[zp])), t  # consumer sees the same scale the producer wrote
        if t not in dqd:
            dqd.add(t)
            nodes.append(h.make_node("DequantizeLinear", [t + "_q", t + "_s", t + "_z"], [t + "_dq"]))
        return t + "_dq"

    for n in z.graph.node:
        o, i, op = n.output[0], list(n.input), n.op_type
        if op == "QuantizeLinear":
            reg(o, arr[i[1]], arr[i[2]]); q("data", o)
        elif op == "QLinearConv":
            x = dq(i[0], i[1], i[2])
            inits.extend([init[i[3]], init[i[4]], init[i[5]]])
            nodes.append(h.make_node("DequantizeLinear", [i[3], i[4], i[5]], [o + "_w"], axis=0))
            ins = [x, o + "_w"]
            if len(i) > 8:
                inits.append(nh.from_array((arr[i[8]] * par[i[0]][0] * arr[i[4]]).astype(np.float32), o + "_b"))
                ins.append(o + "_b")
            nodes.append(h.make_node("Conv", ins, [o + "_f"], **{a.name: h.get_attribute_value(a) for a in n.attribute}))
            reg(o, arr[i[6]], arr[i[7]]); q(o + "_f", o, int(arr[i[7]]) == 0)
        elif op == "QLinearAdd":
            nodes.append(h.make_node("Add", [dq(i[0], i[1], i[2]), dq(i[3], i[4], i[5])], [o + "_f"]))
            reg(o, arr[i[6]], arr[i[7]]); q(o + "_f", o, int(arr[i[7]]) == 0)
        elif op == "MaxPool":
            at = {a.name: h.get_attribute_value(a) for a in n.attribute}
            pads, at["pads"] = at["pads"], [0] * 4
            assert par[i[0]][1] == 0  # post-ReLU input, so zero padding == -inf padding; TensorRT 11 fails to build a padded MaxPool
            inits.append(nh.from_array(np.array([0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]], np.int64), o + "_pads"))
            nodes.append(h.make_node("Pad", [dq(i[0]), o + "_pads"], [o + "_p"]))
            nodes.append(h.make_node("MaxPool", [o + "_p"], [o + "_f"], **at))
            reg(o, *par[i[0]]); q(o + "_f", o)
        elif op == "QLinearGlobalAveragePool":
            nodes.append(h.make_node("GlobalAveragePool", [dq(i[0], i[1], i[2])], [o + "_f"]))
            reg(o, arr[i[3]], arr[i[4]]); q(o + "_f", o)
        else:
            raise ValueError(f"unhandled op {op}")
    f32 = lambda n, v: nh.from_array(np.array(v, np.float32).reshape(1, 3, 1, 1), n)
    inits.extend([f32("pre_scale", 1 / (255 * STD)), f32("pre_shift", -MEAN / STD), nh.from_array(np.array([-1, 2048], np.int64), "shape")])
    i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
    inits.extend([i64("pre_s0", [0]), i64("pre_s2", [2]), i64("pre_hw", [256, 256]), i64("pre_lo", [16, 16]), i64("pre_hi", [240, 240]),
                  i64("pre_ax", [2, 3])])
    pre_nodes = [h.make_node("Cast", ["image"], ["pre_f0"], to=T.FLOAT), h.make_node("Transpose", ["pre_f0"], ["pre_f"], perm=[0, 3, 1, 2]),
                 h.make_node("Shape", ["pre_f"], ["pre_shp"]), h.make_node("Slice", ["pre_shp", "pre_s0", "pre_s2"], ["pre_nc"]),
                 h.make_node("Concat", ["pre_nc", "pre_hw"], ["pre_sizes"], axis=0),
                 h.make_node("Resize", ["pre_f", "", "", "pre_sizes"], ["pre_r"], mode="linear", coordinate_transformation_mode="half_pixel"),
                 h.make_node("Slice", ["pre_r", "pre_lo", "pre_hi", "pre_ax"], ["pre_c"]),
                 h.make_node("Mul", ["pre_c", "pre_scale"], ["pre_m"]), h.make_node("Add", ["pre_m", "pre_shift"], ["data"])]
    g = h.make_graph(pre_nodes + nodes + [h.make_node("Reshape", [POOL + "_q", "shape"], ["feat_q"])], "qdq",
                     [h.make_tensor_value_info("image", T.UINT8, ["N", "H", "W", 3])],
                     [h.make_tensor_value_info("feat_q", T.INT8, [None, 2048])], list({i.name: i for i in inits}.values()))
    return h.make_model(g, opset_imports=[h.make_opsetid("", 13)], ir_version=8)


class Tee:
    """stdout -> terminal and run/train.log"""
    def __init__(self, *fs): self.fs = fs
    def write(self, s): [f.write(s) for f in self.fs]
    def flush(self): [f.flush() for f in self.fs]


class Engine:
    """TensorRT engine runner: uint8 [N,H,W,3] (H, W <= MAXHW) -> list of output arrays, N <= BATCH."""
    DT = {trt.float32: np.float32, trt.int32: np.int32, trt.int8: np.int8}

    def __init__(self, path):
        self.e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.e.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]
        self.din = cudart.cudaMalloc(BATCH * MAXHW * MAXHW * 3)[1]
        self.outs = [(n, self.DT[self.e.get_tensor_dtype(n)], tuple(self.e.get_tensor_shape(n))[1:])
                     for n in (self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors))
                     if self.e.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dout = [cudart.cudaMalloc(BATCH * int(np.prod(sh)) * np.dtype(dt).itemsize)[1] for _, dt, sh in self.outs]

    def __call__(self, imgs):
        x, n = np.ascontiguousarray(imgs), len(imgs)
        ys = [np.empty((n, *sh), dt) for _, dt, sh in self.outs]
        self.ctx.set_input_shape("image", x.shape)
        self.ctx.set_tensor_address("image", int(self.din))
        for (name, _, _), d in zip(self.outs, self.dout):
            self.ctx.set_tensor_address(name, int(d))
        cudart.cudaMemcpy(self.din, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        assert self.ctx.execute_async_v3(int(self.stream))
        cudart.cudaStreamSynchronize(self.stream)
        for y, d in zip(ys, self.dout):
            cudart.cudaMemcpy(y.ctypes.data, d, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return ys


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x8x8x3",
                        f"--optShapes=image:{BATCH}x64x64x3", f"--maxShapes=image:{BATCH}x{MAXHW}x{MAXHW}x3"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if "[E]" in l)


def build_engine(sample):
    """Cached: zoo INT8 -> QDQ -> TensorRT INT8 engine. `sample`: a few PIL images to check the engine against the QDQ model."""
    eng = CACHE / "backbone.engine"
    if eng.exists():
        return eng
    CACHE.mkdir(exist_ok=True)
    onnx.save(qdq_backbone(), CACHE / "backbone-qdq.onnx")
    compile_engine(CACHE / "backbone-qdq.onnx", eng)
    u8 = np.stack([arr(i) for i in sample])
    ref = ort.InferenceSession(str(CACHE / "backbone-qdq.onnx")).run(None, {"image": u8})[0].astype(np.float32)
    got = Engine(eng)(u8)[0].astype(np.float32)
    cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
    assert cos.min() > 0.98, f"engine differs from its QDQ source: cos {cos.min():.3f}"
    return eng


def train_head(X, y, Xv, yv, k, epochs, patience, bs=256, lr=1e-3, eps=1e-8, seed=0):
    """Linear head on X/255 (uint8 embeddings) on the GPU: Adam with a constant lr, no scheduler or weight decay; early stop on val loss
    after `patience` stale epochs. Returns metrics + the best-val-loss weights (W [2048,k], b) as numpy."""
    X, Xv = torch.as_tensor(X, device="cuda"), torch.as_tensor(Xv, device="cuda").float() / 255  # X stays int8, converted per batch
    y, yv = torch.as_tensor(y, device="cuda").long(), torch.as_tensor(yv, device="cuda").long()
    torch.manual_seed(seed)
    head = torch.nn.Linear(2048, k).cuda()
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=lr, eps=eps)
    best, bad, ep = np.inf, 0, 0
    while epochs is None or ep < epochs:
        ep += 1
        idx = torch.randperm(len(X), device="cuda")
        for i in range(0, len(X), bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(head(X[j].float() / 255), y[j]).backward()
            opt.step()
        with torch.no_grad():
            z = head(Xv)
            loss, acc = F.cross_entropy(z, yv).item(), (z.argmax(1) == yv).float().mean().item()
        print(f"epoch {ep} val_loss {loss:.4f} val_acc {acc:.4f}", flush=True)
        if loss < best - 1e-4:
            best, bad, info = loss, 0, dict(best_epoch=ep, val_loss=loss, val_acc=acc)
            W, b = head.weight.detach().T.cpu().numpy(), head.bias.detach().cpu().numpy()
        elif (bad := bad + 1) >= patience:
            break
    torch.cuda.empty_cache()  # give the GPU back before trtexec builds the final engine
    return info | dict(epochs_run=ep), W, b


def quantize(W, b):
    W = W / 255  # fold /255 so the head takes raw uint8
    sw = 127 / np.abs(W).max()
    return dict(W=np.round(W * sw).astype(np.int8), b=np.round(b * sw).astype(np.int32),
                M=round(LUT_STEP / sw * 65536),
                lut=np.round(65535 * np.exp(-np.arange(256) / LUT_STEP)).astype(np.uint16))


def build_model(head, classes):
    """Backbone QDQ + head in one graph, TensorRT-parseable: uint8 image -> label, conf%. The backbone's int8 pool tensor feeds the
    2048xK GEMM directly, int8 x int8 (DQ with scale 1, so the output is the int32 accumulator). Bias, LUT softmax on K numbers are float ops on integer values:
    TensorRT has no BitShift/MatMulInteger and int32 d*M could overflow."""
    m = onnx.load(CACHE / "backbone-qdq.onnx")
    g = m.graph
    f = lambda n, v: nh.from_array(np.array(v, np.float32), n)
    g.initializer.extend([nh.from_array(head["W"], "W8"), f("W1", 1.0), nh.from_array(np.array(0, np.int8), "W0"),
                          f("b", head["b"]), f("lo", 0), f("lut", head["lut"]),
                          f("mul", head["M"] / 65536), f("top", 255), f("pct", 100 * 65535),
                          nh.from_array(np.array([1], np.int64), "ax")])
    n = h.make_node
    g.node.extend([
        n("DequantizeLinear", ["feat_q", "W1", "W0"], ["feat"]),
        n("DequantizeLinear", ["W8", "W1", "W0"], ["W"]), n("MatMul", ["feat", "W"], ["acc"]), n("Add", ["acc", "b"], ["z"]),
        n("ArgMax", ["z"], ["l64"], axis=1, keepdims=0), n("Cast", ["l64"], ["label"], to=T.INT32),
        n("ReduceMax", ["z"], ["zmax"], axes=[1], keepdims=1), n("Sub", ["zmax", "z"], ["d"]), n("Mul", ["d", "mul"], ["d2"]),
        n("Clip", ["d2", "lo", "top"], ["d3"]), n("Cast", ["d3"], ["idx"], to=T.INT32), n("Gather", ["lut", "idx"], ["ex"]),
        n("ReduceSum", ["ex", "ax"], ["esum"], keepdims=0), n("Div", ["pct", "esum"], ["c"]), n("Cast", ["c"], ["conf"], to=T.INT32)])
    del g.output[:]
    g.output.extend([h.make_tensor_value_info("label", T.INT32, [None]), h.make_tensor_value_info("conf", T.INT32, [None])])
    m.metadata_props.add(key="classes", value=json.dumps(list(classes)))
    return m


def bench(f, x, n):
    f(x)
    t0 = time.perf_counter()
    for _ in range(n):
        f(x)
    return (time.perf_counter() - t0) / n * 1e3


def gpu_mb():
    _, free, total = cudart.cudaMemGetInfo()
    return (total - free) / 2**20


def infer(eng, imgs):
    """PIL images -> engine outputs; same-size images go in one batch, others one by one."""
    a = [arr(i) for i in imgs]
    parts = [eng(np.stack(a))] if len({x.shape for x in a}) == 1 else [eng(x[None]) for x in a]
    return [np.concatenate(o) for o in zip(*parts)]


def embed(eng, imgs):
    return np.concatenate([infer(eng, imgs[s:s + BATCH])[0] for s in range(0, len(imgs), BATCH)])



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="cifar10")
    ap.add_argument("--epochs", type=int, help="default: until early stop")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
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
    info, W, b = train_head(X[t], y[t], X[v], y[v], len(classes), a.epochs, a.patience, lr=a.lr, eps=a.eps, seed=seed)
    tm["head_train_s"], tm["head_ms_per_epoch"] = time.perf_counter() - t0, (time.perf_counter() - t0) / info["epochs_run"] * 1e3
    head = quantize(W, b)
    acc = float(((Xt.astype(np.int64) @ head["W"].astype(np.int64) + head["b"]).argmax(1) == yt).mean())
    acc_f = float(((Xt / 255 @ W + b).argmax(1) == yt).mean())

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
                    classes=len(classes), lr=a.lr, eps=a.eps, seed=seed, gpu_mb=gpu, timing={k: round(v, 3) for k, v in tm.items()},
                    host_peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(run, json.dumps(m, indent=1))
