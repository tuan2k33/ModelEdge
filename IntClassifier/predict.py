"""python predict.py weight/<dataset>/model.engine image.jpg  (any image size up to 512x512, everything runs in the engine; confidence = sigmoid of the top logit)"""
import json, sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

from model import Engine, infer

eng = Engine(sys.argv[1])
label, logits = infer(eng, [Image.open(sys.argv[2])])
classes = json.loads({p.key: p.value for p in onnx.load(Path(sys.argv[1]).with_suffix(".onnx"), load_external_data=False).metadata_props}["classes"])
print(classes[label[0]], f"{100 / (1 + np.exp(-logits[0, label[0]])):.0f}%")
