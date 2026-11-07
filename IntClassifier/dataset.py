"""Datasets. A loader returns (train PIL images, train labels, test PIL images, test labels, class names);
per / test_per cap the images per class (None = all). To add a dataset: write a loader and register it in DATASETS.
A directory with one subfolder per class also works (no registration)."""
import collections, io, pickle, tarfile, urllib.request, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("data")


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def cifar10(per, test_per):
    tgz = DATA / "cifar-10-python.tar.gz"
    if not tgz.exists():
        fetch("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tgz)
        tarfile.open(tgz).extractall(DATA, filter="data")
    d = DATA / "cifar-10-batches-py"
    ld = lambda f: pickle.load(open(d / f, "rb"), encoding="bytes")
    cap = lambda y, k: np.concatenate([np.where(y == c)[0][:k] for c in range(10)])
    parts = [ld(f"data_batch_{i}") for i in range(1, 6)]
    X = np.concatenate([p[b"data"] for p in parts]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.concatenate([p[b"labels"] for p in parts])
    t = ld("test_batch")
    Xt, yt = t[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(t[b"labels"])
    i, j = cap(y, per), cap(yt, test_per)
    classes = [c.decode() for c in ld("batches.meta")[b"label_names"]]
    return [Image.fromarray(a) for a in X[i]], y[i], [Image.fromarray(a) for a in Xt[j]], yt[j], classes


def tiny_imagenet(per, test_per):  # reads the zip directly: 110k tiny files would be slow to extract on /mnt/c
    z = zipfile.ZipFile(fetch("http://cs231n.stanford.edu/tiny-imagenet-200.zip", DATA / "tiny-imagenet-200.zip"))
    r = "tiny-imagenet-200/"
    classes = z.read(r + "wnids.txt").decode().split()
    idx = {c: i for i, c in enumerate(classes)}
    load = lambda n: Image.open(io.BytesIO(z.read(n))).convert("RGB")
    cnt, tr, y = collections.Counter(), [], []
    for n in z.namelist():
        c = n.split("/")[2] if n.startswith(r + "train/") else None
        if c and n.endswith(".JPEG") and (per is None or cnt[c] < per):
            cnt[c] += 1; tr.append(load(n)); y.append(idx[c])
    cnt, te, yt = collections.Counter(), [], []
    for line in z.read(r + "val/val_annotations.txt").decode().splitlines():
        f, c = line.split("\t")[:2]
        if test_per is None or cnt[c] < test_per:
            cnt[c] += 1; te.append(load(r + "val/images/" + f)); yt.append(idx[c])
    return tr, np.array(y), te, np.array(yt), classes


def folder(root, per, test_per=None):  # every 5th image of each class is held out as test (test_per is ignored)
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    tr, te = ([], []), ([], [])
    for k, c in enumerate(classes):
        for n, f in enumerate(sorted((root / c).iterdir())[:per]):
            s = te if n % 5 == 4 else tr
            s[0].append(Image.open(f)); s[1].append(k)
    return tr[0], np.array(tr[1]), te[0], np.array(te[1]), classes


DATASETS = {"cifar10": cifar10, "tiny-imagenet": tiny_imagenet}


def load(name, per=None, test_per=None):
    return DATASETS[name](per, test_per) if name in DATASETS else folder(name, per)
