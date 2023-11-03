import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

from utile import load_image, model_load

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)

# 最終の出力
out = np.array(Image.open("/work/Data/out/castle/my_castle.png"))
out = out.astype("float32")

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
            y = torch.unsqueeze(y, dim=0)
            y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean
            y_masked = y.detach().clone()
            mask = torch.ones(1, 4, 64, 64).to(torch_device)
            mask[:, :, 32:38, 26:38] = 0
            y_masked *= mask

            # 画像の拡大縮小とVAEによる復号化
            y = y.float()
            y = 1 / 0.18215 * y

            with torch.no_grad():
                y = vae.decode(y)

            # 生成画像の表示
            y = (y.sample / 2 + 0.5).clamp(0, 1)
            y = y.detach().cpu().permute(0, 2, 3, 1).numpy()
            y = (y * 255).round().astype("uint8")

            # 画像の拡大縮小とVAEによる復号化
            y_masked = y_masked.float()
            y_masked = 1 / 0.18215 * y_masked

            with torch.no_grad():
                y_masked = vae.decode(y_masked)

            # 生成画像の表示
            y_masked = (y_masked.sample / 2 + 0.5).clamp(0, 1)
            y_masked = y_masked.detach().cpu().permute(0, 2, 3, 1).numpy()
            y_masked = (y_masked * 255).round().astype("uint8")

            y = np.squeeze(y)
            y_masked = np.squeeze(y_masked)

            out += (y - y_masked) ** 2
out /= count
out = out.astype("uint8")
sns.heatmap(out[:, :, 0])
plt.savefig("pic_mask_check_heat.jpg")
pil_image = Image.fromarray(out)
pil_image.save("pic_mask_check.jpg")
