"""ONNX Model Zoo ArcFace ResNet100 INT8 (QLinear) converted to QDQ for TensorRT, plus an engine runner.
Input: uint8 NHWC aligned face (112x112 RGB). Output: float [N,512] embedding (not normalized)."""
import subprocess
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from onnx import TensorProto as T, helper as h, numpy_helper as nh

CACHE = Path("cache")
ZOO = CACHE / "arcfaceresnet100-11-int8.onnx"
ZOO_URL = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-11-int8/resolve/main/arcfaceresnet100-11-int8.onnx"
MAXHW = 512
BATCH = 64
SIDE = 112


def front_end():
    """uint8 NHWC `image` of any size -> float NCHW `_mulscalar0` (112x112, (x-127.5)/128), inside the engine."""
    i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
    f32 = lambda n, v: nh.from_array(np.array(v, np.float32), n)
    inits = [f32("pre_sub", 127.5), f32("pre_mul", 1 / 128), i64("pre_s0", [0]), i64("pre_s2", [2]), i64("pre_hw", [SIDE, SIDE])]
    nodes = [h.make_node("Cast", ["image"], ["pre_f0"], to=T.FLOAT), h.make_node("Transpose", ["pre_f0"], ["pre_f"], perm=[0, 3, 1, 2]),
             h.make_node("Shape", ["pre_f"], ["pre_shp"]), h.make_node("Slice", ["pre_shp", "pre_s0", "pre_s2"], ["pre_nc"]),
             h.make_node("Concat", ["pre_nc", "pre_hw"], ["pre_sizes"], axis=0),
             h.make_node("Resize", ["pre_f", "", "", "pre_sizes"], ["pre_r"], mode="linear", coordinate_transformation_mode="half_pixel"),
             h.make_node("Sub", ["pre_r", "pre_sub"], ["pre_c"]), h.make_node("Mul", ["pre_c", "pre_mul"], ["_mulscalar0"])]
    return nodes, inits


def qdq_model():
    """QLinear (uint8 acts) -> QDQ (int8, zero-point 0). Activation scale s -> 2s for zp 0 (post-ReLU) or max(zp, 255-zp)*s/127 otherwise.
    Float ops between the quantized ones (PRelu, BatchNorm) are kept as they are."""
    if not ZOO.exists():
        ZOO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-C", "-", "-o", str(ZOO), ZOO_URL], check=True)
    z = onnx.load(ZOO)
    init = {i.name: i for i in z.graph.initializer}
    val = {k: nh.to_array(v) for k, v in init.items()}
    nodes, inits, par, dqd, alias = [], [], {}, set(), {}
    skip = {"scalar_op2", "scalar_op3"}

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
        if t in init:  # constant operand: fold to float
            assert t not in par
            return t + "_c"
        assert s is None or par[t] == (float(val[s]), int(val[zp])), t
        if t not in dqd:
            dqd.add(t)
            nodes.append(h.make_node("DequantizeLinear", [t + "_q", t + "_s", t + "_z"], [t + "_dq"]))
        return t + "_dq"

    def const(t, s, zp):
        inits.append(nh.from_array(((val[t].astype(np.float32) - float(val[zp])) * float(val[s])).astype(np.float32), t + "_c"))

    for n in z.graph.node:
        o, i, op = n.output[0], list(n.input), n.op_type
        if op in ("Sub", "Mul") and set(i) & skip:
            continue  # replaced by front_end
        if op == "QuantizeLinear":
            reg(o, val[i[1]], val[i[2]]); q(alias.get(i[0], i[0]), o)
        elif op == "DequantizeLinear":
            alias[o] = dq(i[0], i[1], i[2])
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
            ins = []
            for a, s, zp in ((i[0], i[1], i[2]), (i[3], i[4], i[5])):
                if a in init:
                    const(a, s, zp)
                ins.append(dq(a, s, zp))
            nodes.append(h.make_node("Add", ins, [o + "_f"]))
            reg(o, val[i[6]], val[i[7]]); q(o + "_f", o, int(val[i[7]]) == 0)
        elif op == "QLinearMatMul":
            inits.extend([init[i[3]], init[i[4]], init[i[5]]])
            nodes.append(h.make_node("DequantizeLinear", [i[3], i[4], i[5]], [o + "_w"]))
            nodes.append(h.make_node("MatMul", [dq(i[0], i[1], i[2]), o + "_w"], [o + "_f"]))
            reg(o, val[i[6]], val[i[7]]); q(o + "_f", o, int(val[i[7]]) == 0)
        else:  # float op: PRelu, BatchNormalization, Flatten
            inits.extend(init[x] for x in i if x in init)
            nodes.append(h.make_node(op, [alias.get(x, x) for x in i], list(n.output), **{a.name: h.get_attribute_value(a) for a in n.attribute}))
    pre_nodes, pre_inits = front_end()
    inits.extend(pre_inits)
    g = h.make_graph(pre_nodes + nodes, "qdq", [h.make_tensor_value_info("image", T.UINT8, ["N", "H", "W", 3])],
                     [h.make_tensor_value_info("fc1", T.FLOAT, [None, 512])], list({i.name: i for i in inits}.values()))
    return h.make_model(g, opset_imports=[h.make_opsetid("", 13)], ir_version=8)


class Engine:
    """TensorRT engine runner: uint8 [N,H,W,3] (H, W <= MAXHW; numpy, or a torch tensor on the GPU) -> list of outputs
    (numpy, or torch tensors on the GPU if gpu), N <= BATCH."""
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
        K = cudart.cudaMemcpyKind
        on_gpu = isinstance(imgs, torch.Tensor)
        x = imgs.contiguous() if on_gpu else np.ascontiguousarray(imgs)
        n = len(x)
        ys = [torch.empty((n, *sh), dtype=self.TDT[dt], device="cuda") if gpu else np.empty((n, *sh), dt) for _, dt, sh in self.outs]
        self.ctx.set_input_shape("image", tuple(x.shape))
        self.ctx.set_tensor_address("image", int(self.din))
        for (name, _, _), d in zip(self.outs, self.dout):
            self.ctx.set_tensor_address(name, int(d))
        if on_gpu:
            cudart.cudaMemcpy(self.din, x.data_ptr(), x.nelement(), K.cudaMemcpyDeviceToDevice)
        else:
            cudart.cudaMemcpy(self.din, x.ctypes.data, x.nbytes, K.cudaMemcpyHostToDevice)
        assert self.ctx.execute_async_v3(int(self.stream))
        cudart.cudaStreamSynchronize(self.stream)
        for y, d in zip(ys, self.dout):
            if gpu:
                cudart.cudaMemcpy(y.data_ptr(), d, y.nelement() * y.element_size(), K.cudaMemcpyDeviceToDevice)
            else:
                cudart.cudaMemcpy(y.ctypes.data, d, y.nbytes, K.cudaMemcpyDeviceToHost)
        return ys


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x8x8x3",
                        f"--optShapes=image:{BATCH}x{SIDE}x{SIDE}x3", f"--maxShapes=image:{BATCH}x{MAXHW}x{MAXHW}x3"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if "[E]" in l)
