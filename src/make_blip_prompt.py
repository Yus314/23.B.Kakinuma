import json
import os

import numpy as np
import torch
from clip_interrogator import Config, Interrogator
from PIL import Image

from utile import load_image

image_prompt = []

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "../Data/in/imagenet_1k"
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
for image_file in os.listdir(image_dir):
    image_path: str = image_dir + "/" + image_file
    txt_image = Image.open(image_path)
    txt = ci.interrogate(txt_image)
    pair = {"image": image_path, "prompt": txt}
    image_prompt.append(pair)
json_file = "BSDS500_prompt.json"
with open(json_file, "w") as file:
    json.dump(image_prompt, file, indent=4)
