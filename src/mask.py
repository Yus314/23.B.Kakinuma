import os

import numpy as np
from PIL import Image

for image in os.listdir("../imagenet1k/"):
    if not image.endswith(".png"):
        continue
    im = np.array(Image.open(f"../imagenet1k/{image}"))
    im[84:172, 84:172, :] = 0
    Image.fromarray(im).save(f"../rekka_imagenet1k/{image}")
