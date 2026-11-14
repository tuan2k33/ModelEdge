"""Datasets. A loader returns (train PIL images, train labels, test PIL images, test labels, class names);
per / test_per cap the images per class (None = all). To add a dataset: write a loader and register it in DATASETS.
A directory <root>/{train,test}/<class>/<image> also works (no registration)."""
import collections, gzip, io, pickle, subprocess, tarfile, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("data")


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-C", "-", "-o", str(path), url], check=True)  # curl, not urllib: in this environment urlretrieve crawled (10 MB in over 90 s) while curl fetched 102 MB in 11 s
    return path


def _cifar(name, per, test_per, train_files, test_file, labels, meta, names):
    tgz = DATA / f"{name}-python.tar.gz"
    if not tgz.exists():
        fetch(f"https://www.cs.toronto.edu/~kriz/{name}-python.tar.gz", tgz)
        tarfile.open(tgz).extractall(DATA, filter="data")
    d = DATA / ("cifar-10-batches-py" if name == "cifar-10" else "cifar-100-python")
    ld = lambda f: pickle.load(open(d / f, "rb"), encoding="bytes")
    parts, t = [ld(f) for f in train_files], ld(test_file)
    img = lambda p: p[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X, y = np.concatenate([img(p) for p in parts]), np.concatenate([p[labels] for p in parts])
    Xt, yt = img(t), np.array(t[labels])
    classes = [c.decode() for c in ld(meta)[names]]
    cap = lambda y, k: np.concatenate([np.where(y == c)[0][:k] for c in range(len(classes))])
    i, j = cap(y, per), cap(yt, test_per)
    return [Image.fromarray(a) for a in X[i]], y[i], [Image.fromarray(a) for a in Xt[j]], yt[j], classes


def cifar10(per, test_per):
    return _cifar("cifar-10", per, test_per, [f"data_batch_{i}" for i in range(1, 6)], "test_batch", b"labels", "batches.meta", b"label_names")


def cifar100(per, test_per):
    return _cifar("cifar-100", per, test_per, ["train"], "test", b"fine_labels", "meta", b"fine_label_names")


def mnist(per, test_per):  # 28x28 grey digits, shown to the network as RGB
    def read(split, n):
        raw = lambda f, off: np.frombuffer(gzip.decompress(fetch(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", DATA / f).read_bytes()), np.uint8, offset=off)
        X, y = raw(f"{split}-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28), raw(f"{split}-labels-idx1-ubyte.gz", 8)
        i = np.concatenate([np.where(y == c)[0][:n] for c in range(10)])
        return [Image.fromarray(a).convert("RGB") for a in X[i]], y[i]
    tr, y = read("train", per)
    te, yt = read("t10k", test_per)
    return tr, y, te, yt, [str(c) for c in range(10)]


def emnist(per, test_per):  # balanced split: 47 classes (digits + letters, look-alike lower/upper merged), 28x28 grey stored transposed
    z = zipfile.ZipFile(fetch("https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", DATA / "emnist-gzip.zip"))
    raw = lambda f, off: np.frombuffer(gzip.decompress(z.read(f"gzip/emnist-balanced-{f}")), np.uint8, offset=off)
    classes = [chr(int(l.split()[1])) for l in z.read("gzip/emnist-balanced-mapping.txt").decode().split("\n") if l.strip()]
    def read(split, n):
        X, y = raw(f"{split}-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28).transpose(0, 2, 1), raw(f"{split}-labels-idx1-ubyte.gz", 8)
        i = np.concatenate([np.where(y == c)[0][:n] for c in range(len(classes))])
        return [Image.fromarray(a).convert("RGB") for a in X[i]], y[i]
    tr, y = read("train", per)
    te, yt = read("test", test_per)
    return tr, y, te, yt, classes


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


def folder(root, per, test_per=None):  # <root>/train/<class>/*.jpg and <root>/test/<class>/*.jpg (like torchvision's ImageFolder, twice)
    def read(split, n):
        d = Path(root) / split
        classes = sorted(p.name for p in d.iterdir() if p.is_dir())
        ims = [[Image.open(f).convert("RGB") for f in sorted((d / c).iterdir())[:n]] for c in classes]  # convert loads the pixels and closes the file
        return [i for g in ims for i in g], np.array([k for k, g in enumerate(ims) for _ in g]), classes
    tr, y, classes = read("train", per)
    te, yt, c2 = read("test", test_per)
    assert classes == c2, "train and test must have the same class folders"
    return tr, y, te, yt, classes


DATASETS = {"cifar10": cifar10, "cifar100": cifar100, "mnist": mnist, "emnist": emnist, "tiny-imagenet": tiny_imagenet}


def load(name, per=None, test_per=None):
    return DATASETS[name](per, test_per) if name in DATASETS else folder(name if Path(name).exists() else DATA / name, per, test_per)
