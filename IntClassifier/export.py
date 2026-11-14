"""Backbones as ready-to-run INT8 Q/DQ ONNX (uint8 NHWC `image` [N,H,W,3] in, any size; int8 `feat_q` [N,D] out, feat = feat_q * metadata feat_scale),
built once from the upstream weights and pushed to the Hugging Face Hub, so training a head only needs the file:
    python export.py resnet50            # ONNX Model Zoo resnet50-v1-12-int8 (QLinear) -> weight/backbone/resnet50-qdq-int8.onnx, D=2048
    python export.py convnext            # facebook/dinov3-convnext-tiny-pretrain-lvd1689m -> weight/backbone/dinov3-convnext-tiny-qdq-int8.onnx, D=768
    python export.py push [resnet50|convnext]   # into the private HF repo model.HUB, folder <file stem>/, with the license and a model card
ConvNeXt: Q/DQ only where int8 math runs, the activation input and the weight of the stem/downsample convs and the pointwise Linears (40 of 58).
Depthwise convs stay FP16 (the most sensitive layers, and int8 depthwise was slower on TensorRT: layout changes around it); LayerNorm, GELU and the
residual stream stay FP16 with no Q/DQ around them (TensorRT fuses GELU into the first pointwise GEMM and the following Q into LayerNorm).
The HF channels-last layout is kept: a channels-first rewrite (1x1 convs, LayerNorm over channels) removed the transposes but was slower at every batch.
Calibration: per-tensor activation amax = 99.99th percentile of |x| over 256 COCO val2017 images (data/coco/val2017), weights per output channel."""
import json, os, sys
from pathlib import Path

import numpy as np, onnx, torch, torch.nn.functional as F
from onnx import helper as h, numpy_helper as nh

from model import BACKBONES as NAMES, BATCH, HUB, MEAN, STD, qdq_backbone

OUT = Path("weight/backbone")
CONVNEXT = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
PCT, NCAL = 99.99, 256


def resnet50():
    m = qdq_backbone()
    s = next(float(nh.to_array(i)) for i in m.graph.initializer if i.name == "resnetv17_pool1_fwd_quantized_s")
    return m, dict(feat_scale=s, source="https://huggingface.co/onnxmodelzoo/resnet50-v1-12-int8", license="apache-2.0")


class Wrap(torch.nn.Module):
    """uint8 NHWC image -> resize 224 (bilinear, no antialias, as the ResNet50 backbone) -> ImageNet normalize -> HF pooler_output (LayerNorm of the avg pool)."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("scale", torch.tensor(1 / (255 * STD), dtype=torch.float16).view(1, 3, 1, 1))
        self.register_buffer("shift", torch.tensor(-MEAN / STD, dtype=torch.float16).view(1, 3, 1, 1))

    def forward(self, image):
        x = image.half().permute(0, 3, 1, 2)  # cast first: TensorRT has no uint8 transpose
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False) * self.scale + self.shift
        return self.m(pixel_values=x).pooler_output


def convnext():
    from PIL import Image
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(CONVNEXT, token=os.environ.get("HF_TOKEN")).eval().cuda().half()
    net = Wrap(hf).cuda().eval()
    files = sorted(Path("data/coco/val2017").glob("*.jpg"))[:NCAL]
    assert len(files) == NCAL, "calibration needs COCO val2017 in data/coco/val2017"
    imgs = torch.stack([torch.as_tensor(np.asarray(Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR))) for f in files]).cuda()
    mods = {"Conv": [m for m in hf.modules() if isinstance(m, torch.nn.Conv2d)], "MatMul": [m for m in hf.modules() if isinstance(m, torch.nn.Linear)]}
    samples = {id(m): [] for v in mods.values() for m in v}
    hooks = [m.register_forward_pre_hook(lambda mod, a: samples[id(mod)].append(a[0].detach().abs().flatten().float()[::97].cpu())) for v in mods.values() for m in v]
    feats = []
    with torch.no_grad():
        for i in range(0, NCAL, 32):
            feats.append(net(imgs[i:i + 32]).float().abs().flatten().cpu())
    for k in hooks:
        k.remove()
    pct = lambda xs: torch.quantile(torch.cat(xs)[:2**24], PCT / 100).item()
    amax = {k: pct(v) for k, v in samples.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "convnext-fp16.onnx"
    torch.onnx.export(net, (imgs[:2],), tmp, input_names=["image"], output_names=["feat"], dynamo=True, external_data=False,
                      dynamic_shapes={"image": {0: torch.export.Dim("N", max=BATCH), 1: torch.export.Dim("H", max=512), 2: torch.export.Dim("W", max=512)}})
    m = onnx.load(tmp); tmp.unlink()
    g = m.graph
    init = {i.name: i for i in g.initializer}
    f16 = lambda name, v: nh.from_array(np.asarray(v, np.float16), name)
    z8 = lambda name, n=None: nh.from_array(np.array(0, np.int8) if n is None else np.zeros(n, np.int8), name)
    out, seen = [], {"Conv": 0, "MatMul": 0}
    for node in g.node:
        if node.op_type in seen:  # ONNX Conv/MatMul nodes come in the order of the torch modules: matched by order, checked by weight value
            mod = mods[node.op_type][seen[node.op_type]]; k = f"{node.op_type}{seen[node.op_type]}"; seen[node.op_type] += 1
            w = nh.to_array(init[node.input[1]]).astype(np.float32)
            ref = mod.weight.detach().float().cpu().numpy()
            ref = ref.T if node.op_type == "MatMul" else ref  # Linear exports as MatMul with the weight [in, out]
            assert w.shape == ref.shape and np.abs(w - ref).max() < 1e-2, k
            if not (node.op_type == "Conv" and mod.groups > 1):  # depthwise stays FP16
                ax = 1 if node.op_type == "MatMul" else 0  # output channel
                ws = np.maximum(np.abs(w).max(axis=tuple(i for i in range(w.ndim) if i != ax)), 1e-8) / 127
                shp = [-1 if i == ax else 1 for i in range(w.ndim)]
                g.initializer.extend([nh.from_array(np.clip(np.round(w / ws.reshape(shp)), -127, 127).astype(np.int8), k + "_wq"),
                                      f16(k + "_ws", ws), f16(k + "_as", amax[id(mod)] / 127), z8(k + "_z"), z8(k + "_wz", len(ws))])
                out += [h.make_node("QuantizeLinear", [node.input[0], k + "_as", k + "_z"], [k + "_xq"]),
                        h.make_node("DequantizeLinear", [k + "_xq", k + "_as", k + "_z"], [k + "_x"]),
                        h.make_node("DequantizeLinear", [k + "_wq", k + "_ws", k + "_wz"], [k + "_w"], axis=ax)]
                node.input[0], node.input[1] = k + "_x", k + "_w"
        out.append(node)
    assert seen == {k: len(v) for k, v in mods.items()}, seen
    fs = pct(feats) / 127  # int8 embedding out, like the ResNet50 backbone
    g.initializer.extend([f16("feat_s", fs), z8("feat_z")])
    out.append(h.make_node("QuantizeLinear", ["feat", "feat_s", "feat_z"], ["feat_q"]))
    dim = g.output[0].type.tensor_type.shape.dim[1].dim_value
    del g.node[:]; g.node.extend(out)
    del g.output[:]; g.output.append(h.make_tensor_value_info("feat_q", onnx.TensorProto.INT8, ["N", dim]))
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]; g.initializer.extend(keep)
    return m, dict(feat_scale=fs, source=f"https://huggingface.co/{CONVNEXT}", license="DINOv3 License (LICENSE.md)",
                   calibration=f"COCO val2017, first {NCAL} images; activation amax = {PCT}th percentile of |x|")


def save(which):
    m, meta = globals()[which]()
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v if isinstance(v, str) else json.dumps(v))
    onnx.checker.check_model(m)
    path = OUT / f"{NAMES[which]}.onnx"
    onnx.save(m, path)
    print(path, f"{path.stat().st_size / 2**20:.1f} MB", meta)


CARD = """# {name}

INT8 Q/DQ ONNX backbone for [ModelEdge IntClassifier](https://github.com/tuan2k33/ModelEdge): a frozen feature extractor that TensorRT builds into one INT8 engine.

- Input `image`: uint8 [N, H, W, 3] RGB, any size; resize to 224x224 (bilinear) and ImageNet normalization run inside the graph.
- Output `feat_q`: int8 [N, {dim}]; the embedding is `feat_q * feat_scale` (`feat_scale` = {feat_scale:.6g}, also in the ONNX metadata).
- Source: {source}
{notes}
Build: `trtexec --onnx={name}.onnx --minShapes=image:1x8x8x3 --optShapes=image:64x64x64x3 --maxShapes=image:64x512x512x3 --saveEngine=model.engine` (TensorRT 11, strongly typed).

## License
{license_text}
"""
NOTES = {
    "resnet50": "- Conversion: the Model Zoo QLinear graph to Q/DQ (int8, zero-point 0, what TensorRT accepts); weights and biases keep their exact values. "
                "Every activation stays int8 (convs, residual adds, pools); `feat_q` is the global-average-pool tensor (>= 0).\n",
    "convnext": "- DINOv3 ConvNeXt-Tiny distilled from DINOv3 ViT-7B; `feat_q` is the pooled, LayerNorm-ed embedding (the HF `pooler_output`).\n"
                "- Quantization: int8 activation and weight on the stem/downsample convs and the 36 pointwise Linears (per-channel weights, per-tensor "
                "activations, 99.99th percentile on 256 COCO val2017 images); depthwise convs, LayerNorm, GELU and the residual stream stay FP16.\n",
}


def push(which):
    from huggingface_hub import HfApi, hf_hub_download
    path = OUT / f"{NAMES[which]}.onnx"
    meta = {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}
    dim = onnx.load(path).graph.output[0].type.tensor_type.shape.dim[1].dim_value
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HUB, private=True, exist_ok=True)
    if which == "convnext":
        lic = Path(hf_hub_download(CONVNEXT, "LICENSE.md", token=os.environ["HF_TOKEN"]))
        files, license_, license_text = {"LICENSE.md": lic}, "other", "DINOv3 License, see [LICENSE.md](LICENSE.md): this model is a derivative of the DINO Materials and is distributed under that Agreement."
    else:
        from dataset import fetch
        lic = fetch("https://www.apache.org/licenses/LICENSE-2.0.txt", OUT / "LICENSE-apache-2.0.txt")
        files, license_, license_text = {"LICENSE": lic}, "apache-2.0", "Apache-2.0 (see [LICENSE](LICENSE)), as the source model in the ONNX Model Zoo."
    card = OUT / f"{NAMES[which]}.README.md"
    card.write_text(CARD.format(name=NAMES[which], dim=dim, feat_scale=float(meta["feat_scale"]), source=meta["source"], notes=NOTES[which],
                                license=license_, license_text=license_text))
    for dst, src in {f"{NAMES[which]}.onnx": path, "README.md": card, **files}.items():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{NAMES[which]}/{dst}", repo_id=HUB)
    print(f"https://huggingface.co/{HUB}/tree/main/{NAMES[which]}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["resnet50"]
    if cmd == "push":
        for w in rest or list(NAMES):
            push(w)
    else:
        save(cmd)
