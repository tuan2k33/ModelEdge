"""Does the Motion mask help a YOLO that is not retrained? Boxes from collect.py are re-scored in logit space and compared with the plain scores against the reference boxes.
A box is `fg` when it holds the centre of a Motion blob (P), when its own centre is on the mask (C), or by the share of its area on the mask (F).
The added constants are log likelihood ratios measured on the other video: a = log P(fg | true) / P(fg | false), b = log P(not fg | true) / P(not fg | false).
    python fuse.py cache/traffic.npz cache/pedestrians.npz [--any-class]"""
import sys

import cv2, numpy as np


def iou(a, b):
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    return inter / ((a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter + 1e-9)


def features(d, name):
    """Per box of detector `name`: P, C, F and whether it matches a reference box (same class, IoU >= 0.5, each reference box once, best score first)."""
    det, ref, (h, w) = d[name], d["yolo26x"], d["shape"]
    mh, mw = d["masks"].shape[1:]
    sx, sy = mw / w, mh / h
    P, C, F, tp = (np.zeros(len(det), bool) for _ in range(4))
    F = np.zeros(len(det), np.float32)
    nref = 0
    for k, m in enumerate(d["masks"]):
        idx = np.nonzero(det[:, 0] == k)[0]
        idx = idx[np.argsort(-det[idx, 5])]
        b, r = det[idx, 1:5], ref[ref[:, 0] == k]
        nref += len(r)
        n, _, stats, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        cen = cen[1:][stats[1:, 4] >= 12] / [sx, sy]
        if len(cen):
            P[idx] = ((cen[None, :, 0] >= b[:, None, 0]) & (cen[None, :, 0] <= b[:, None, 2]) & (cen[None, :, 1] >= b[:, None, 1]) & (cen[None, :, 1] <= b[:, None, 3])).any(1)
        cx, cy = ((b[:, 0] + b[:, 2]) / 2 * sx).astype(int).clip(0, mw - 1), ((b[:, 1] + b[:, 3]) / 2 * sy).astype(int).clip(0, mh - 1)
        C[idx] = m[cy, cx] > 0
        ii = np.pad(m.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        x1, y1 = (b[:, 0] * sx).astype(int).clip(0, mw - 1), (b[:, 1] * sy).astype(int).clip(0, mh - 1)
        x2, y2 = np.maximum((b[:, 2] * sx).astype(int).clip(0, mw), x1 + 1), np.maximum((b[:, 3] * sy).astype(int).clip(0, mh), y1 + 1)
        F[idx] = (ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]) / ((x2 - x1) * (y2 - y1))
        if len(r) and len(b):
            o = iou(b, r[:, 1:5]) * ((det[idx, 6][:, None] == r[None, :, 6]) | ANY_CLASS)
            used = np.zeros(len(r), bool)
            for j in range(len(b)):
                o[j, used] = 0
                q = o[j].argmax()
                if o[j, q] >= 0.5:
                    used[q] = tp[idx[j]] = True
    return {"P": P, "C": C, "F": F >= 0.5}, tp, nref


def ratios(f, tp):
    e = 0.5
    return (np.log(((f & tp).sum() + e) / (tp.sum() + 2 * e)) - np.log(((f & ~tp).sum() + e) / ((~tp).sum() + 2 * e)),
            np.log(((~f & tp).sum() + e) / (tp.sum() + 2 * e)) - np.log(((~f & ~tp).sum() + e) / ((~tp).sum() + 2 * e)))


def score(s, tp, nref):
    """AP@0.5, then precision and recall at score 0.25, then recall at precision 0.9."""
    o = np.argsort(-s)
    c = np.cumsum(tp[o])
    prec, rec = c / np.arange(1, len(o) + 1), c / nref
    env = np.maximum.accumulate(prec[::-1])[::-1]
    ap = np.sum(np.diff(np.concatenate([[0], rec])) * env)
    k = (s >= 0.25).sum()
    ok = np.nonzero(prec >= 0.9)[0]
    return ap, (prec[k - 1] if k else 1), (rec[k - 1] if k else 0), (rec[ok[-1]] if len(ok) else 0)


ANY_CLASS = "--any-class" in sys.argv  # a box on a reference box counts whatever its class
data = {p: np.load(p) for p in sys.argv[1:3]}
for name in ["yolo26n", "yolo26m"]:
    feat = {p: features(d, name) for p, d in data.items()}
    for test, other in [sys.argv[1:3], sys.argv[2:0:-1]]:
        f, tp, nref = feat[test]
        s = data[test][name][:, 5].clip(1e-6, 1 - 1e-6)
        logit = np.log(s / (1 - s))
        print(f"\n{name} on {test} ({nref} reference boxes; constants from {other})")
        print(f"  {'':28s} AP50   P@.25  R@.25  R@P.9")
        print("  %-28s %.3f  %.3f  %.3f  %.3f" % ("plain", *score(s, tp, nref)))
        for key in "PCF":
            a, b = ratios(feat[other][0][key], feat[other][1])
            fg = f[key]
            for label, add in [(f"{key} boost a={a:+.2f}", a * fg), (f"{key} both  a={a:+.2f} b={b:+.2f}", a * fg + b * ~fg)]:
                print("  %-28s %.3f  %.3f  %.3f  %.3f" % (label, *score(1 / (1 + np.exp(-(logit + add))), tp, nref)))
