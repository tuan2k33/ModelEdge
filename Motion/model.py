"""Background model for a fixed camera as one TensorRT engine with its memory passed in and out, no trained weights.
Inputs: `image` uint8 [1,H,W,3] RGB, `state` float [1,8,PH,PW] (zeros at the start). Outputs: `state_out` (same layout), `mask` uint8 [1,1,PH,PW] (1 = differs from the background).
State channels: 0-2 background RGB, 3 its weight (updates it has been seen for, at most `cap`), 4-6 candidate RGB, 7 candidate count.
A pixel that matches the background raises its weight; one that does not feeds the candidate, which replaces the background once its count passes that weight (at the start the weight is 0, so the first frame is the background).
When most of the frame stops matching at once, the weights drop to `form` so the background re-forms in that time."""
import math, subprocess
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch
import torch.nn.functional as F
from onnx import TensorProto as T, helper as h, numpy_helper as nh

PH, PW = 180, 320
POOL = 4
MAXHW = 2160
ALPHA = 0.05  # a matched background pixel moves at least this far towards the frame per update, so it follows slow light changes however old it is
FLOOD = 0.5
VER = 2  # part of the engine file name: raise it when the graph changes
CACHE = Path("cache")


class Background(torch.nn.Module):
    def __init__(self, form, cap, thr):
        super().__init__()
        self.form, self.cap, self.thr = float(form), float(cap), float(thr)

    def forward(self, image, state):
        x = image.float().permute(0, 3, 1, 2)
        x = F.avg_pool2d(F.interpolate(x, size=(PH * POOL, PW * POOL), mode="bilinear", align_corners=False), POOL)
        bg, w, cand, cnt = state[:, 0:3], state[:, 3:4], state[:, 4:7], state[:, 7:8]
        hit = ((x - bg).abs().amax(1, keepdim=True) < self.thr) & (w > 0)
        hit_c = ((x - cand).abs().amax(1, keepdim=True) < self.thr) & (cnt > 0)

        flood = (~hit).float().mean() > FLOOD  # most of the frame changed at once: the light, not objects, so the background is re-formed
        w = torch.where(flood, w.clamp(max=self.form), w)
        w1 = torch.where(hit, (w + 1).clamp(max=self.cap), w)
        bg1 = torch.where(hit, bg + (x - bg) * (1 / w1.clamp(min=1)).clamp(min=ALPHA), bg)
        new = ~hit & ~hit_c & (cnt <= 1)  # the candidate loses a count each time something else shows and is replaced at zero: what is there most of the time wins, even on a busy path
        cnt1 = torch.where(hit_c & ~hit, cnt + 1, torch.where(new, torch.ones_like(cnt), (cnt - 1).clamp(min=0)))
        cand1 = torch.where(hit_c & ~hit, cand + (x - cand) / cnt1, torch.where(new, x, cand))
        swap = cnt1 > w1  # the candidate has been there longer than the background: they trade places, the old background stays remembered as the candidate
        out = torch.cat([torch.where(swap, cand1, bg1), torch.where(swap, cnt1, w1), torch.where(swap, bg1, cand1), torch.where(swap, w1, cnt1)], 1)

        m = (~hit).float()
        m = -F.max_pool2d(-m, 3, 1, 1)  # drop specks
        m = F.max_pool2d(m, 5, 1, 2)
        return out, m.to(torch.uint8)


def export(path, form, cap, thr, opset=17):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Background(form, cap, thr), (torch.zeros(1, 540, 960, 3, dtype=torch.uint8), torch.zeros(1, 8, PH, PW)), str(path), dynamo=False, opset_version=opset,
                      input_names=["image", "state"], output_names=["state_out", "mask"], dynamic_axes={"image": {1: "H", 2: "W"}})


def fuse(det_onnx, path, form, cap, thr, boost):
    """One ONNX from the Detector's model.onnx and the background model: `image`, `state` -> `detections` [1,300,6], `state_out`, `mask`.
    A box whose centre is on the mask gets `boost` added to the logit of its score (the rows stay in the detector's order, no longer sorted by score)."""
    det = onnx.load(det_onnx)
    export(path, form, cap, thr, det.opset_import[0].version)
    bg, g = onnx.load(path), det.graph
    keep = {"image", "state", "state_out", "mask"}
    name = lambda n: n if n in keep or not n else "bg/" + n
    for n in bg.graph.node:
        n.name = name(n.name)
        n.input[:], n.output[:] = [name(i) for i in n.input], [name(o) for o in n.output]
    for i in bg.graph.initializer:
        i.name = name(i.name)
    def c(n, v, t=np.float32):
        g.initializer.append(nh.from_array(np.array(v, t), n))
        return n

    n = h.make_node
    two, four, five, six = (c(f"mo_{k}", [k], np.int64) for k in (2, 4, 5, 6))
    zero, swap, ax = c("mo_0", [0], np.int64), c("mo_swap", [1, 0], np.int64), c("mo_ax", [2], np.int64)
    out = g.output[0].name
    maskf = next(x for x in bg.graph.node if "mask" in x.output).input[0]  # the mask before its cast to uint8: TensorRT 11.4 returns a wrong `mask` when that output is also read inside the graph
    nodes = [n("Slice", [out, zero, two, ax], ["mo_p1"]), n("Slice", [out, two, four, ax], ["mo_p2"]), n("Slice", [out, zero, four, ax], ["mo_box"]),
             n("Slice", [out, four, five, ax], ["mo_conf"]), n("Slice", [out, five, six, ax], ["mo_cls"]),
             n("Add", ["mo_p1", "mo_p2"], ["mo_sum"]), n("Mul", ["mo_sum", c("mo_half", 0.5)], ["mo_c"]),            # box centre, x y in the letterboxed frame
             n("Cast", ["pre_bi"], ["mo_tl"], to=T.FLOAT), n("Gather", ["mo_tl", swap], ["mo_lt"]),                   # left, top of the letterbox
             n("Sub", ["mo_c", "mo_lt"], ["mo_c1"]), n("Div", ["mo_c1", "pre_r"], ["mo_xy"]),                         # centre in image pixels
             n("Gather", ["pre_hwf", swap], ["mo_wh"]), n("Div", ["mo_xy", "mo_wh"], ["mo_n"]),
             n("Mul", ["mo_n", c("mo_two", 2.0)], ["mo_n2"]), n("Sub", ["mo_n2", c("mo_one", 1.0)], ["mo_g"]), n("Unsqueeze", ["mo_g", ax], ["mo_grid"]),
             n("GridSample", [maskf, "mo_grid"], ["mo_s"], mode="nearest", padding_mode="zeros", align_corners=0),
             n("Reshape", ["mo_s", c("mo_shape", [1, -1, 1], np.int64)], ["mo_fg"]),
             # sigmoid(logit(s) + boost) = s k / (1 + s (k - 1)) with k = e^boost
             n("Mul", ["mo_fg", c("mo_k", math.exp(boost) - 1)], ["mo_k1"]), n("Add", ["mo_k1", "mo_one"], ["mo_kk"]),
             n("Mul", ["mo_conf", "mo_kk"], ["mo_num"]), n("Mul", ["mo_conf", "mo_k1"], ["mo_d"]), n("Add", ["mo_d", "mo_one"], ["mo_den"]),
             n("Div", ["mo_num", "mo_den"], ["mo_new"]), n("Concat", ["mo_box", "mo_new", "mo_cls"], ["detections"], axis=2)]
    g.node.extend(list(bg.graph.node) + nodes)
    g.initializer.extend(bg.graph.initializer)
    g.input.append(bg.graph.input[1])
    del g.output[:]
    g.output.extend([h.make_tensor_value_info("detections", T.FLOAT, [1, 300, 6])] + list(bg.graph.output))
    onnx.save(det, path)


def compile_engine(onnx_path, engine_path, top=(MAXHW, MAXHW * 2)):
    """top: the largest frame, height and width (the fused engine needs it smaller: the detector's letterbox works on the full frame)."""
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x90x160x3",
                        "--optShapes=image:1x1080x1920x3", f"--maxShapes=image:1x{top[0]}x{top[1]}x3"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "[E]" in l)


class Engine:
    """uint8 [1,H,W,3] cuda tensor -> mask uint8 [PH,PW] cuda tensor. The state stays on the GPU; update=False leaves it as it was (mask only)."""

    def __init__(self, path):
        e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = e.create_execution_context()
        self.state = torch.zeros(1, 8, PH, PW, device="cuda")
        self.next = torch.zeros_like(self.state)
        self.mask = torch.zeros(1, 1, PH, PW, dtype=torch.uint8, device="cuda")
        self.ctx.set_tensor_address("mask", self.mask.data_ptr())
        self.det = torch.zeros(1, 300, 6, device="cuda") if "detections" in [e.get_tensor_name(i) for i in range(e.num_io_tensors)] else None  # fuse()
        if self.det is not None:
            self.ctx.set_tensor_address("detections", self.det.data_ptr())

    def __call__(self, image, update=True):
        image = image.contiguous()
        self.ctx.set_input_shape("image", tuple(image.shape))
        self.ctx.set_tensor_address("image", image.data_ptr())
        self.ctx.set_tensor_address("state", self.state.data_ptr())
        self.ctx.set_tensor_address("state_out", self.next.data_ptr())
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        if update:
            self.state, self.next = self.next, self.state
        return self.mask[0, 0]

    @property
    def background(self):
        return self.state[0, :3].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
