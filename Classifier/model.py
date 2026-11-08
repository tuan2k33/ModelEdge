"""The FP32 model: ONNX Model Zoo ResNet50 (float32) behind the in-engine preprocessing, an engine runner, and the FP32 linear head graph."""
import json, subprocess
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
from cuda.bindings import runtime as cudart
from onnx import TensorProto as T, helper as h, numpy_helper as nh, version_converter

CACHE = Path("cache")
ZOO = CACHE / "zoo-fp32" / "resnet50-v1-12.onnx"
ZOO_URL = "https://huggingface.co/onnxmodelzoo/resnet50-v1-12/resolve/main/resnet50-v1-12.onnx"
POOL = "resnetv17_pool1_fwd"  # float [N,2048,1,1] after the global average pool
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MAXHW = 512  # largest input side the engine accepts (batch x side^2 x float32 temp must fit the GPU)
BATCH = 64


def arr(img):
    return np.asarray(img.convert("RGB"))


def infer(eng, imgs):
    """PIL images -> engine outputs; same-size images go in one batch, others one by one."""
    a = [arr(i) for i in imgs]
    parts = [eng(np.stack(a))] if len({x.shape for x in a}) == 1 else [eng(x[None]) for x in a]
    return [np.concatenate(o) for o in zip(*parts)]


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


def backbone():
    """uint8 image of any size -> in-engine preprocessing -> ResNet50 FP32 up to the global-avg-pool -> float embedding [N,2048]."""
    if not ZOO.exists():
        ZOO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-C", "-", "-o", str(ZOO), ZOO_URL], check=True)
    m = version_converter.convert_version(onnx.load(ZOO), 13)
    g = m.graph
    del g.node[next(i for i, n in enumerate(g.node) if POOL in n.output) + 1:]
    pre, pre_inits = front_end()
    nodes = pre + list(g.node) + [h.make_node("Reshape", [POOL, "shape"], ["emb"])]
    used = {t for n in nodes for t in n.input}
    inits = [i for i in list(g.initializer) + pre_inits + [nh.from_array(np.array([-1, 2048], np.int64), "shape")] if i.name in used]
    graph = h.make_graph(nodes, "fp32", [h.make_tensor_value_info("image", T.UINT8, ["N", "H", "W", 3])],
                         [h.make_tensor_value_info("emb", T.FLOAT, [None, 2048])], inits)
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)


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


def build_model(W, b, scale, classes):
    """Backbone + FP32 head in one graph: logits = emb @ (W / scale) + b, confidence = 100 * max softmax."""
    m = backbone()
    g = m.graph
    g.initializer.extend([nh.from_array((W / scale).astype(np.float32), "W"), nh.from_array(b.astype(np.float32), "b"),
                          nh.from_array(np.array(100, np.float32), "hundred")])
    n = h.make_node
    g.node.extend([n("MatMul", ["emb", "W"], ["acc"]), n("Add", ["acc", "b"], ["z"]),
                   n("ArgMax", ["z"], ["l64"], axis=1, keepdims=0), n("Cast", ["l64"], ["label"], to=T.INT32),
                   n("Softmax", ["z"], ["p"], axis=1), n("ReduceMax", ["p"], ["pm"], axes=[1], keepdims=0),
                   n("Mul", ["pm", "hundred"], ["c"]), n("Cast", ["c"], ["conf"], to=T.INT32)])
    del g.output[:]
    g.output.extend([h.make_tensor_value_info("label", T.INT32, [None]), h.make_tensor_value_info("conf", T.INT32, [None])])
    m.metadata_props.add(key="classes", value=json.dumps(list(classes)))
    return m
