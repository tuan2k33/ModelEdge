"""Background model for a fixed camera as one TensorRT engine with its memory passed in and out, no trained weights.
Inputs: `image` uint8 [1,H,W,3] RGB, `state` float [1,8,PH,PW] (zeros at the start). Outputs: `state_out` (same layout), `mask` uint8 [1,1,PH,PW] (1 = differs from the background).
State channels: 0-2 background RGB, 3 its weight (updates it has been seen for, at most `cap`), 4-6 candidate RGB, 7 candidate count.
A pixel that matches the background raises its weight; one that does not feeds the candidate, which replaces the background once its count passes both `form` and that weight."""
import subprocess
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn.functional as F

PH, PW = 180, 320
POOL = 4
MAXHW = 2160
ALPHA = 0.05  # a matched background pixel moves at least this far towards the frame per update, so it follows slow light changes however old it is
FLOOD = 0.5
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
        swap = cnt1 > w1.clamp(min=self.form)  # the candidate has been there longer than the background: they trade places, the old background stays remembered as the candidate
        out = torch.cat([torch.where(swap, cand1, bg1), torch.where(swap, cnt1, w1), torch.where(swap, bg1, cand1), torch.where(swap, w1, cnt1)], 1)

        m = (~hit).float()
        m = -F.max_pool2d(-m, 3, 1, 1)  # drop specks
        m = F.max_pool2d(m, 5, 1, 2)
        return out, m.to(torch.uint8)


def export(path, form, cap, thr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Background(form, cap, thr), (torch.zeros(1, 540, 960, 3, dtype=torch.uint8), torch.zeros(1, 8, PH, PW)), str(path), dynamo=False, opset_version=17,
                      input_names=["image", "state"], output_names=["state_out", "mask"], dynamic_axes={"image": {1: "H", 2: "W"}})


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--minShapes=image:1x90x160x3",
                        "--optShapes=image:1x1080x1920x3", f"--maxShapes=image:1x{MAXHW}x{MAXHW * 2}x3"], capture_output=True, text=True)
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
