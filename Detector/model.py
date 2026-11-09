"""Ultralytics YOLO26 FP16 ONNX wrapped with an in-engine letterbox, and the TensorRT engine runner.
Input: uint8 NHWC image of any size (batch 1, sides <= MAXHW), letterboxed to 640x640 inside the engine. Output: float [1,300,6] (NMS-free: x1 y1 x2 y2 conf class, in the letterboxed 640x640 frame)."""
import json, subprocess
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart
from onnx import TensorProto as T, helper as h, numpy_helper as nh

CACHE = Path("cache")
SIDE = 640
MAXHW = 2048


def front_end():
    """uint8 NHWC `image` -> float NCHW `images` in 0..1, 640x640, inside the engine: Ultralytics' letterbox (scale to fit, keep the aspect ratio,
    centred, padded with 114). Boxes come out in this 640x640 frame: x = (x_out - left) / r, y = (y_out - top) / r, see letterbox()."""
    f32 = lambda n, v: nh.from_array(np.array(v, np.float32), n)
    i64 = lambda n, v: nh.from_array(np.array(v, np.int64), n)
    inits = [f32("pre_scale", 1 / 255), f32("pre_side", SIDE), f32("pre_half", 0.5), f32("pre_tenth", 0.1), i64("pre_s0", [0]), i64("pre_s2", [2]),
             i64("pre_s3", [3]), i64("pre_s4", [4]), i64("pre_one", [1]), i64("pre_w", [SIDE]), i64("pre_fixed", [1, 3, SIDE, SIDE])]
    n = h.make_node
    nodes = [n("Cast", ["image"], ["pre_f0"], to=T.FLOAT), n("Transpose", ["pre_f0"], ["pre_f"], perm=[0, 3, 1, 2]),
             n("Shape", ["pre_f"], ["pre_shp"]), n("Slice", ["pre_shp", "pre_s0", "pre_s2"], ["pre_nc"]), n("Slice", ["pre_shp", "pre_s2", "pre_s4"], ["pre_hw"]),
             n("Cast", ["pre_hw"], ["pre_hwf"], to=T.FLOAT),                                  # [H, W]
             n("Div", ["pre_side", "pre_hwf"], ["pre_rr"]), n("ReduceMin", ["pre_rr"], ["pre_r"], keepdims=1),   # r = min(640/H, 640/W)
             n("Mul", ["pre_hwf", "pre_r"], ["pre_nf0"]), n("Round", ["pre_nf0"], ["pre_nf"]),                # [newH, newW]
             n("Cast", ["pre_nf"], ["pre_ni"], to=T.INT64), n("Concat", ["pre_nc", "pre_ni"], ["pre_sizes"], axis=0),
             n("Resize", ["pre_f", "", "", "pre_sizes"], ["pre_rs"], mode="linear", coordinate_transformation_mode="half_pixel"),
             n("Sub", ["pre_side", "pre_nf"], ["pre_d0"]), n("Mul", ["pre_d0", "pre_half"], ["pre_d"]),     # d = (640 - new) / 2 for H and W
             n("Sub", ["pre_d", "pre_tenth"], ["pre_b0"]), n("Round", ["pre_b0"], ["pre_b"]),                # top, left   = round(d - 0.1)
             n("Add", ["pre_d", "pre_tenth"], ["pre_e0"]), n("Round", ["pre_e0"], ["pre_e"]),                # bottom, right = round(d + 0.1)
             n("Cast", ["pre_b"], ["pre_bi"], to=T.INT64), n("Cast", ["pre_e"], ["pre_ei"], to=T.INT64),
             # TensorRT 11 fails to build a Pad with dynamic widths: paste the image into a 114 canvas with Concat of ConstantOfShape blocks instead
             n("Slice", ["pre_bi", "pre_s0", "pre_one"], ["pre_top"]), n("Slice", ["pre_bi", "pre_one", "pre_s2"], ["pre_left"]),
             n("Slice", ["pre_ei", "pre_s0", "pre_one"], ["pre_bot"]), n("Slice", ["pre_ei", "pre_one", "pre_s2"], ["pre_right"]),
             n("Slice", ["pre_ni", "pre_s0", "pre_one"], ["pre_nh"]),
             n("Concat", ["pre_nc", "pre_nh", "pre_left"], ["pre_sl"], axis=0), n("ConstantOfShape", ["pre_sl"], ["pre_bl"], value=nh.from_array(np.array([114], np.float32))),
             n("Concat", ["pre_nc", "pre_nh", "pre_right"], ["pre_sr"], axis=0), n("ConstantOfShape", ["pre_sr"], ["pre_br"], value=nh.from_array(np.array([114], np.float32))),
             n("Concat", ["pre_bl", "pre_rs", "pre_br"], ["pre_mid"], axis=3),
             n("Concat", ["pre_nc", "pre_top", "pre_w"], ["pre_st"], axis=0), n("ConstantOfShape", ["pre_st"], ["pre_bt"], value=nh.from_array(np.array([114], np.float32))),
             n("Concat", ["pre_nc", "pre_bot", "pre_w"], ["pre_sb"], axis=0), n("ConstantOfShape", ["pre_sb"], ["pre_bb"], value=nh.from_array(np.array([114], np.float32))),
             n("Concat", ["pre_bt", "pre_mid", "pre_bb"], ["pre_p0"], axis=2),
             n("Reshape", ["pre_p0", "pre_fixed"], ["pre_p"]),  # always 1x3x640x640: tells TensorRT the shape is static (it cannot infer it from the Concat)
             n("Mul", ["pre_p", "pre_scale"], ["images"])]
    return nodes, inits


def letterbox(h, w):
    """(r, left, top) of the in-engine letterbox for an h x w image (same arithmetic as the graph)."""
    r = min(SIDE / h, SIDE / w)
    nh_, nw_ = round(h * r), round(w * r)
    return r, round((SIDE - nw_) / 2 - 0.1), round((SIDE - nh_) / 2 - 0.1)


def with_front_end(path, names):
    """Ultralytics FP16 ONNX (float32 NCHW 0..1 -> float32 [1,300,6]) -> the same model taking a uint8 NHWC image of any size (letterbox inside the engine).
    The class names ride along in the ONNX metadata."""
    m = onnx.load(path)
    g = m.graph
    nodes, inits = front_end()
    nodes[-1].output[0] = "images_f32"
    nodes.append(h.make_node("Cast", ["images_f32"], ["images"], to=T.FLOAT16))
    old = list(g.node)
    del g.node[:], g.input[:]
    g.node.extend(nodes + old)
    g.initializer.extend(inits)
    g.input.append(h.make_tensor_value_info("image", T.UINT8, [1, "H", "W", 3]))
    m.metadata_props.add(key="classes", value=json.dumps(list(names)))
    return m


class Engine:
    """TensorRT engine runner: uint8 [1,H,W,3] (H, W <= MAXHW; numpy, or a torch tensor on the GPU) -> list of numpy arrays."""
    DT = {trt.float32: np.float32, trt.int32: np.int32, trt.int8: np.int8}

    def __init__(self, path):
        self.e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.e.create_execution_context()
        self.stream = cudart.cudaStreamCreate()[1]
        self.din = cudart.cudaMalloc(MAXHW * MAXHW * 3)[1]
        self.outs = [(n, self.DT[self.e.get_tensor_dtype(n)], tuple(self.e.get_tensor_shape(n)))
                     for n in (self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors))
                     if self.e.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dout = [cudart.cudaMalloc(int(np.prod(sh)) * np.dtype(dt).itemsize)[1] for _, dt, sh in self.outs]

    def __call__(self, img):
        on_gpu = isinstance(img, torch.Tensor)
        x = img.contiguous() if on_gpu else np.ascontiguousarray(img)
        ys = [np.empty(sh, dt) for _, dt, sh in self.outs]
        self.ctx.set_input_shape("image", tuple(x.shape))
        self.ctx.set_tensor_address("image", int(self.din))
        for (name, _, _), d in zip(self.outs, self.dout):
            self.ctx.set_tensor_address(name, int(d))
        if on_gpu:
            cudart.cudaMemcpy(self.din, x.data_ptr(), x.nelement(), cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        else:
            cudart.cudaMemcpy(self.din, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        assert self.ctx.execute_async_v3(int(self.stream))
        cudart.cudaStreamSynchronize(self.stream)
        for y, d in zip(ys, self.dout):
            cudart.cudaMemcpy(y.ctypes.data, d, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return ys


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x8x8x3",
                        f"--optShapes=image:1x{SIDE}x{SIDE}x3", f"--maxShapes=image:1x{MAXHW}x{MAXHW}x3"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in r.stdout.splitlines() if "[E]" in l)
