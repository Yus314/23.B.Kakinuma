import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

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
dir = ["test", "train"]
for pl_dir in dir:
    cur = "/work/Data/in/BSDS300/images/"
    cur = cur + pl_dir
    files = os.listdir(cur)
    count += len(files)
    for image_file in files:
        with torch.no_grad():
            image_dir: str = cur
            image_path: str = image_dir + "/" + image_file
            y: torch.tensor = load_image(image_path, torch_device)
            y_masked = y.detach().clone()
            y_masked[:, 205:306, 205:306] = 0
            y = torch.unsqueeze(y, dim=0)
            y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean
            y_masked = torch.unsqueeze(y_masked, dim=0)
            y_masked = 0.1825 * vae.encode(2 * y_masked - 1).latent_dist.mean
            out += (y - y_masked) ** 2
out /= count


out_check = out.to("cpu").detach().numpy().copy()
# sns.heatmap(out_check[0, 3, :, :])
# plt.savefig("mask_check4.jpg")
fin_out = out_check
fin_out[0][0] = out_check[0][0] < 1.5
fin_out[0][1] = out_check[0][1] < 0.8
fin_out[0][2] = out_check[0][2] < 0.8
fin_out[0][3] = out_check[0][3] < 0.9
out_check = out_check < 3
fin_out = fin_out.astype(np.float32)
sns.heatmap(out_check[0, 2, :, :])
# plt.savefig("mask_check2_l.jpg")
# out_check = out_check.astype(np.int)
# out_check = out_check.astype(np.float32)

# np.save("latent_mask.np", out_check)
np.save("latent_mask.np", fin_out)
# sns.heatmap(out_check[0, 0, :, :])]
# plt.savefig("mask_check1.jpg")
