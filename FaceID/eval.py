"""Face verification accuracy of the TensorRT engine (and optionally the ONNX models on the CPU, slow): 10-fold, cosine threshold picked on 9 folds, tested on the 10th.
Decoding (nvJPEG), engine, normalization, cosine and folds all run on the GPU. Embeddings are cached in cache/emb-<set>-<model>.npz.
    python eval.py [--sets lfw,cfp_fp,agedb_30] [--models engine,qdq,orig] [--pairs N]"""
import argparse, functools, json, time

import numpy as np
import torch

import dataset
from model import BATCH, CACHE, Engine, compile_engine, qdq_model

ORIG = CACHE / "arcfaceresnet100-11-int8.onnx"
QDQ, ENG = CACHE / "face-qdq.onnx", CACHE / "face.engine"
SRC = {"engine": ENG, "qdq": QDQ, "orig": ORIG}  # an embedding cache older than its model is recomputed


def runner(name):
    """uint8 [n,112,112,3] cuda tensor -> float embeddings (cuda tensor)."""
    if name == "engine":
        if not ENG.exists():
            if not QDQ.exists():
                import onnx
                CACHE.mkdir(exist_ok=True)
                onnx.save(qdq_model(), QDQ)
            compile_engine(QDQ, ENG)
        e = Engine(ENG)
        return lambda x: torch.cat([e(x[i:i + BATCH], gpu=True)[0] for i in range(0, len(x), BATCH)])
    import onnxruntime as ort
    if name == "qdq":
        s = ort.InferenceSession(str(QDQ))
        f = lambda x: s.run(None, {"image": x.cpu().numpy()})[0]
    else:  # the zoo model: batch 1, raw 0..255 NCHW float
        s = ort.InferenceSession(str(ORIG))
        f = lambda x: np.concatenate([s.run(None, {"data": x[i:i + 1].cpu().numpy().astype(np.float32).transpose(0, 3, 1, 2)})[0] for i in range(len(x))])
    return lambda x: torch.as_tensor(np.concatenate([f(x[i:i + 16]) for i in range(0, len(x), 16)]), device="cuda")


def kfold_acc(sim, same, k=10):
    """Mean and std of the accuracy over k folds; the threshold is the best one on the other folds."""
    idx = torch.arange(len(sim), device="cuda").reshape(k, -1)
    ths = sim.sort().values[::20]
    accs = []
    for f in range(k):
        tr, te = idx[torch.arange(k, device="cuda") != f].ravel(), idx[f]
        best = ths[((sim[tr, None] > ths) == same[tr, None]).float().mean(0).argmax()]
        accs.append(((sim[te] > best) == same[te]).float().mean())
    accs = torch.stack(accs)
    return accs.mean().item(), accs.std(unbiased=False).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="lfw,cfp_fp,agedb_30")
    p.add_argument("--models", default="engine")
    p.add_argument("--pairs", type=int, help="use only the first N pairs (a multiple of 10)")
    a = p.parse_args()
    runs = {m: functools.cache(lambda m=m: runner(m)) for m in a.models.split(",")}  # a model is only loaded if some embeddings are missing
    out = {}
    for d in a.sets.split(","):
        enc, same = dataset.load(d)
        if a.pairs:
            enc, same = enc[:2 * a.pairs], same[:a.pairs]
        same = torch.as_tensor(same, device="cuda")
        for m, run in runs.items():
            t = time.time()
            c = CACHE / f"emb-{d}-{m}.npz"
            if not a.pairs and c.exists() and c.stat().st_mtime > SRC[m].stat().st_mtime:
                e = torch.as_tensor(np.load(c)["emb"], device="cuda").double()
            else:
                e = torch.cat([run()(dataset.decode(enc[i:i + BATCH])) for i in range(0, len(enc), BATCH)])
                if not a.pairs:
                    np.savez(c, emb=e.float().cpu().numpy())
                e = e.double()
            e /= e.norm(dim=1, keepdim=True)
            acc, std = kfold_acc((e[0::2] * e[1::2]).sum(1), same)
            out[f"{d}/{m}"] = dict(acc=round(100 * acc, 2), std=round(100 * std, 2), pairs=len(same), seconds=round(time.time() - t, 1))
            print(d, m, out[f"{d}/{m}"], flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
