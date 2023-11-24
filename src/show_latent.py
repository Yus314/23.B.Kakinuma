import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utile import load_image, model_load

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)

# 最終の出力
out = torch.zeros(1, 4, 64, 64, dtype=torch.float32)
out = out.to(torch_device)

# 使った画像の枚数
count = 0
cur = "/work/Data/in/BSDS300/images/"
cur = cur + "test"
image_dir: str = cur
image_path: str = image_dir + "/" + "291000.jpg"
y: torch.tensor = load_image(image_path, torch_device)
y_masked = y
y_masked[:, 210:305, 210:305] = 0
y = torch.unsqueeze(y, dim=0)
y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean
y_masked = torch.unsqueeze(y_masked, dim=0)
y_masked = 0.1825 * vae.encode(2 * y_masked - 1).latent_dist.mean
y_check = y_masked.to("cpu").detach().numpy().copy()  #!!!! 以下を追加
y_check = np.squeeze(y_check)
y_check = y_check[0:3, :, :]
y_check = y_check.transpose(1, 2, 0)

yma = np.max(y_check)
ymi = np.min(y_check)
y_check = (y_check - ymi) / (yma - ymi) * 255
pil_image = Image.fromarray(np.uint8(y_check))
pil_image.save("latent_check.png")


# out_check = out.to("cpu").detach().numpy().copy()
# fig, ax = plt.subplots()
# im = ax.imshow(out_check[0, 0, :, :])
# plt.savefig("mask_check1.jpg")
