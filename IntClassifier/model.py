"""The model: backbone Q/DQ ONNX from the Hub (built by export.py; the ResNet50 QLinear -> QDQ conversion lives here), an engine runner, and the int8 head graph."""
import json, os, subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from onnx import TensorProto as T, helper as h, numpy_helper as nh

CACHE = Path("cache")
ZOO = CACHE / "zoo-int8" / "resnet50-v1-12-int8.onnx"
ZOO_URL = "https://huggingface.co/onnxmodelzoo/resnet50-v1-12-int8/resolve/main/resnet50-v1-12-int8.onnx"
POOL = "resnetv17_pool1_fwd_quantized"  # uint8 [N,2048,1,1] after the global average pool in the zoo model, int8 after conversion
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MAXHW = 512  # largest input side the engine accepts (batch x side^2 x float32 temp must fit the GPU)
BATCH = 64


def arr(img):
    img = img.convert("RGB")
    if max(img.size) > MAXHW:  # the engine takes sides up to MAXHW: shrink bigger photos here (antialiased, aspect kept)
        img.thumbnail((MAXHW, MAXHW))
    return np.asarray(img)


def infer(eng, imgs, gpu=False):
    """PIL images -> engine outputs (numpy, or torch tensors on the GPU if gpu); same-size images go in one batch, others one by one."""
    a = [arr(i) for i in imgs]
    parts = [eng(np.stack(a), gpu)] if len({x.shape for x in a}) == 1 else [eng(x[None], gpu) for x in a]
    return [(torch.cat if gpu else np.concatenate)(o) for o in zip(*parts)]


def front_end():
    """Nodes + initializers turning a uint8 NHWC `image` of any size into the normalized float NCHW `data` (inside the engine)."""
    f32 = lambda n, v: nh.from_array(np.array(v, np.float32).reshape(1, 3, 1, 1), n)
    i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
    inits = [f32("pre_scale", 1 / (255 * STD)), f32("pre_shift", -MEAN / STD), i64("pre_s0", [0]), i64("pre_s2", [2]), i64("pre_hw", [224, 224])]
    nodes = [h.make_node("Cast", ["image"], ["pre_f0"], to=T.FLOAT), h.make_node("Transpose", ["pre_f0"], ["pre_f"], perm=[0, 3, 1, 2]),
             h.make_node("Shape", ["pre_f"], ["pre_shp"]), h.make_node("Slice", ["pre_shp", "pre_s0", "pre_s2"], ["pre_nc"]),
             h.make_node("Concat", ["pre_nc", "pre_hw"], ["pre_sizes"], axis=0),
             h.make_node("Resize", ["pre_f", "", "", "pre_sizes"], ["pre_r"], mode="linear", coordinate_transformation_mode="half_pixel"),
             h.make_node("Mul", ["pre_r", "pre_scale"], ["pre_m"]), h.make_node("Add", ["pre_m", "pre_shift"], ["data"])]
    return nodes, inits


def zoo_cut():
    """The zoo model up to (and including) the global-avg-pool; output is the uint8 pool tensor."""
    if not ZOO.exists():
        ZOO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-C", "-", "-o", str(ZOO), ZOO_URL], check=True)
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
    resized straight to 224x224 (squashed, bilinear, no antialias: shortcut, fine for small or square images; downscaling big photos aliases),
    normalized by float nodes, all inside the engine. Output: the int8 global-avg-pool tensor [N,2048] as is (>= 0, post-ReLU)."""
    z = zoo_cut()
    init = {i.name: i for i in z.graph.initializer}
    val = {k: nh.to_array(v) for k, v in init.items()}
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
        assert s is None or par[t] == (float(val[s]), int(val[zp])), t  # consumer sees the same scale the producer wrote
        if t not in dqd:
            dqd.add(t)
            nodes.append(h.make_node("DequantizeLinear", [t + "_q", t + "_s", t + "_z"], [t + "_dq"]))
        return t + "_dq"

    for n in z.graph.node:
        o, i, op = n.output[0], list(n.input), n.op_type
        if op == "QuantizeLinear":
            reg(o, val[i[1]], val[i[2]]); q("data", o)
        elif op == "QLinearConv":
            x = dq(i[0], i[1], i[2])
            inits.extend([init[i[3]], init[i[4]], init[i[5]]])
            nodes.append(h.make_node("DequantizeLinear", [i[3], i[4], i[5]], [o + "_w"], axis=0))
            ins = [x, o + "_w"]
            if len(i) > 8:
                inits.append(nh.from_array((val[i[8]] * par[i[0]][0] * val[i[4]]).astype(np.float32), o + "_b"))
                ins.append(o + "_b")
            nodes.append(h.make_node("Conv", ins, [o + "_f"], **{a.name: h.get_attribute_value(a) for a in n.attribute}))
            reg(o, val[i[6]], val[i[7]]); q(o + "_f", o, int(val[i[7]]) == 0)
        elif op == "QLinearAdd":
            nodes.append(h.make_node("Add", [dq(i[0], i[1], i[2]), dq(i[3], i[4], i[5])], [o + "_f"]))
            reg(o, val[i[6]], val[i[7]]); q(o + "_f", o, int(val[i[7]]) == 0)
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
            reg(o, val[i[3]], val[i[4]]); q(o + "_f", o)
        else:
            raise ValueError(f"unhandled op {op}")
    pre_nodes, pre_inits = front_end()
    inits.extend(pre_inits + [nh.from_array(np.array([-1, 2048], np.int64), "shape")])
    g = h.make_graph(pre_nodes + nodes + [h.make_node("Reshape", [POOL + "_q", "shape"], ["feat_q"])], "qdq",
                     [h.make_tensor_value_info("image", T.UINT8, ["N", "H", "W", 3])],
                     [h.make_tensor_value_info("feat_q", T.INT8, [None, 2048])], list({i.name: i for i in inits}.values()))
    return h.make_model(g, opset_imports=[h.make_opsetid("", 13)], ir_version=8)


class Engine:
    """TensorRT engine runner: uint8 [N,H,W,3] (H, W <= MAXHW) -> list of output arrays (torch tensors on the GPU if gpu), N <= BATCH."""
    DT = {trt.float32: np.float32, trt.int32: np.int32, trt.int8: np.int8}
    TDT = {np.float32: torch.float32, np.int32: torch.int32, np.int8: torch.int8}

    def __init__(self, path):
        self.e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.e.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]
        self.din = cudart.cudaMalloc(BATCH * MAXHW * MAXHW * 3)[1]
        self.outs = [(n, self.DT[self.e.get_tensor_dtype(n)], tuple(self.e.get_tensor_shape(n))[1:])
                     for n in (self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors))
                     if self.e.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dout = [cudart.cudaMalloc(BATCH * int(np.prod(sh)) * np.dtype(dt).itemsize)[1] for _, dt, sh in self.outs]

    def __call__(self, imgs, gpu=False):
        x, n = np.ascontiguousarray(imgs), len(imgs)
        ys = [torch.empty((n, *sh), dtype=self.TDT[dt], device="cuda") if gpu else np.empty((n, *sh), dt) for _, dt, sh in self.outs]
        self.ctx.set_input_shape("image", x.shape)
        self.ctx.set_tensor_address("image", int(self.din))
        for (name, _, _), d in zip(self.outs, self.dout):
            self.ctx.set_tensor_address(name, int(d))
        cudart.cudaMemcpy(self.din, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        assert self.ctx.execute_async_v3(int(self.stream))
        cudart.cudaStreamSynchronize(self.stream)
        for y, d in zip(ys, self.dout):
            if gpu:
                cudart.cudaMemcpy(y.data_ptr(), d, y.nelement() * y.element_size(), cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            else:
                cudart.cudaMemcpy(y.ctypes.data, d, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return ys


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x8x8x3",
                        f"--optShapes=image:{BATCH}x64x64x3", f"--maxShapes=image:{BATCH}x{MAXHW}x{MAXHW}x3"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if "[E]" in l)


HUB = "tuannw/ModelEdge"  # private HF repo, one folder per model
BACKBONES = {"resnet50": "resnet50-qdq-int8", "convnext": "dinov3-convnext-tiny-qdq-int8"}  # folder = file stem, built by export.py


def backbone_onnx(name):
    """The backbone Q/DQ ONNX from the Hub (private repos: HF_TOKEN from the environment or the project .env), cached by huggingface_hub."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    if not token and Path("../.env").exists():
        token = next((l.split("=", 1)[1].strip().strip("'\"") for l in Path("../.env").read_text().splitlines() if l.startswith("HF_TOKEN=")), None)
    return Path(hf_hub_download(HUB, f"{BACKBONES[name]}/{BACKBONES[name]}.onnx", token=token))


def build_engine(name, sample):
    """Cached: backbone ONNX -> TensorRT engine (cache/<name>.engine). `sample`: a few PIL images to check the engine against the ONNX run by ORT
    (cos > 0.9, not ~1: rounding at bin edges differs between ORT and TensorRT and compounds over the quantized layers)."""
    src, eng = backbone_onnx(name), CACHE / f"{name}.engine"
    if eng.exists():
        return src, eng
    CACHE.mkdir(exist_ok=True)
    compile_engine(src, eng)
    u8 = np.stack([arr(i) for i in sample])
    ref = ort.InferenceSession(str(src)).run(None, {"image": u8})[0].astype(np.float32)
    got = Engine(eng)(u8)[0].astype(np.float32)
    cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
    assert cos.min() > 0.9, f"engine differs from its ONNX source: cos {cos.min():.3f}"
    return src, eng


def quantize(W, b, W2, b2, hmax):
    """MLP head Linear(D,H) + ReLU + Linear(H,k) -> int8 weights, int32 biases, S (logits are integers in units of 1/S);
    the ReLU output is requantized to int8 with scale 127/hmax (hmax = 99.9th percentile of that output, measured at the saved weights)."""
    t = lambda A: torch.as_tensor(A, device="cuda", dtype=torch.float64)  # rounding on the GPU, results back as numpy for the ONNX graph
    W, b, W2, b2 = t(W) / 255, t(b), t(W2), t(b2)  # fold /255 so the head takes the raw int8 embedding
    s = float(127 / W.abs().max())
    q = lambda A, sc: torch.round(A * sc).to(torch.int8).cpu().numpy()
    qi = lambda A, sc: torch.round(A * sc).to(torch.int32).cpu().numpy()
    sh, s2 = 127 / hmax, float(127 / W2.abs().max())  # logits end up in units of 1/(sh*s2)
    return dict(W1=q(W, s), b1=qi(b, s), m1=s / sh, W=q(W2, s2), b=qi(b2, sh * s2), S=1 / (sh * s2))


def build_model(head, classes, src):
    """Backbone QDQ + head in one graph, TensorRT-parseable: uint8 image -> label, logits (float [N,classes], one per class; sigmoid(logit) is that class's confidence, left to the caller). The backbone's int8
    pool tensor feeds the int8 x int8 GEMM(s) directly (DQ with scale 1, so a GEMM output is the int32 accumulator). Bias and ReLU are float ops on integer
    values: TensorRT has no BitShift/MatMulInteger."""
    m = onnx.load(src)
    g = m.graph
    f = lambda n, v: nh.from_array(np.array(v, np.float32), n)
    g.initializer.extend([f("one", 1.0), nh.from_array(np.array(0, np.int8), "zero"), nh.from_array(head["W"], "Wq"), f("b", head["b"]), f("lo", 0)])
    n = h.make_node
    # hidden layer: int8 GEMM, +bias, ReLU, requantize to int8 for the second GEMM
    g.initializer.extend([nh.from_array(head["W1"], "W1q"), f("b1", head["b1"]), f("inv_m1", 1 / head["m1"]), f("c127", 127)])
    nodes = [n("DequantizeLinear", ["feat_q", "one", "zero"], ["head_in"]), n("DequantizeLinear", ["W1q", "one", "zero"], ["W1f"]),
             n("MatMul", ["head_in", "W1f"], ["a1"]), n("Add", ["a1", "b1"], ["u0"]),
             # requantize with float ops, then an exact Q/DQ (scale 1) for the int8 GEMM: a QuantizeLinear with scale ~400 right after the
             # first GEMM got fused wrongly by TensorRT 11 (only ~90% of labels matched a numpy emulation, 100% with this form)
             n("Relu", ["u0"], ["u"]), n("Mul", ["u", "inv_m1"], ["uq"]), n("Round", ["uq"], ["ur"]), n("Clip", ["ur", "lo", "c127"], ["hr"]),
             n("QuantizeLinear", ["hr", "one", "zero"], ["hq"]), n("DequantizeLinear", ["hq", "one", "zero"], ["hf"]),
             n("DequantizeLinear", ["Wq", "one", "zero"], ["Wf"]), n("MatMul", ["hf", "Wf"], ["acc"]), n("Add", ["acc", "b"], ["z"]),
              n("ArgMax", ["z"], ["l64"], axis=1, keepdims=0), n("Cast", ["l64"], ["label"], to=T.INT32)]
    g.initializer.append(f("S", head["S"]))  # the head's logits are integers in units of 1/S
    nodes.append(n("Mul", ["z", "S"], ["logits"]))
    out = [h.make_tensor_value_info("label", T.INT32, [None]), h.make_tensor_value_info("logits", T.FLOAT, [None, len(classes)])]
    names = [o for x in g.node for o in x.output] + [i.name for i in g.initializer]
    assert not {o for x in nodes for o in x.output} & set(names), "head names collide with the backbone graph"
    g.node.extend(nodes)
    del g.output[:]
    g.output.extend(out)
    m.metadata_props.add(key="classes", value=json.dumps(list(classes)))
    return m
