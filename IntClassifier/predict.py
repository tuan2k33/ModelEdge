"""python predict.py weight/run-<time>/model.engine image.jpg  (any image size up to 512x512, everything runs in the engine)"""
import json, sys
from pathlib import Path

import onnx
from PIL import Image

from train import Engine, infer

eng = Engine(sys.argv[1])
label, conf = infer(eng, [Image.open(sys.argv[2])])
classes = json.loads({p.key: p.value for p in onnx.load(Path(sys.argv[1]).with_suffix(".onnx"), load_external_data=False).metadata_props}["classes"])
print(classes[label[0]], f"{conf[0]}%")
