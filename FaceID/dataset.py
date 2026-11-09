"""Face verification sets. load(name) -> (JPEG-encoded images [2N] as uint8 tensors, same bool [N]); pair k = images 2k and 2k+1.
decode(encoded) -> uint8 [n,112,112,3] RGB on the GPU (nvJPEG).
lfw, cfp_fp, agedb_30: InsightFace .bin files (aligned 112x112, 6000 / 7000 / 6000 pairs), mirror gaunernst/face-recognition-eval."""
import pickle, subprocess
from pathlib import Path

import numpy as np
import torch
from torchvision.io import decode_jpeg

DATA = Path("data")
HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
DATASETS = {"lfw", "cfp_fp", "agedb_30"}


class NumpyOnly(pickle.Unpickler):
    """The .bin files are pickles of (list of encoded images, array of bool); only numpy array classes may be loaded."""
    def find_class(self, module, name):
        if module.startswith("numpy") and name in ("ndarray", "dtype", "_reconstruct"):
            return getattr(np, name) if name != "_reconstruct" else np._core.multiarray._reconstruct
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def fetch(repo, file, dst):
    if not dst.exists():
        dst.parent.mkdir(exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-C", "-", "-o", str(dst), HF.format(repo, file)], check=True)
    return dst


def decode(encoded):
    return torch.stack(decode_jpeg(encoded, device="cuda")).permute(0, 2, 3, 1)


def load(name):
    assert name in DATASETS, f"unknown dataset {name}: {sorted(DATASETS)}"
    with open(fetch("gaunernst/face-recognition-eval", name + ".bin", DATA / (name + ".bin")), "rb") as f:
        enc, same = NumpyOnly(f, encoding="bytes").load()
    return [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in enc], np.asarray(same, bool)
