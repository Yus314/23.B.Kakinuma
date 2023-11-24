import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance

# ディレクトリのパスを指定
handbook_dir = "/work/Data/in/BSDS300/images/all"
# generated_dir = "/work/Data/out/all"
generated_dir = "/work/DDNM/exp/image_samples/imagenet_inpainting"
generated_dir2 = "/work/DDNM/exp/image_samples/imagenet_inpainting2"

# KernelInceptionDistanceのインスタンスを作成
kid_metric = KernelInceptionDistance(subset_size=100)

# 手本画像の読み込みとメトリックの更新
for filename in os.listdir(handbook_dir):
    img_path = os.path.join(handbook_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img) / 255.0  # 画像のピクセル値を[0, 1]にスケーリング
    img_tensor = torch.unsqueeze(
        torch.transpose(torch.tensor(img_np, dtype=torch.uint8), 0, 2), 0
    )
    kid_metric.update(img_tensor, real=True)

# 生成画像の読み込みとメトリックの更新
for filename in os.listdir(generated_dir):
    img_path = os.path.join(generated_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img) / 255.0  # 画像のピクセル値を[0, 1]にスケーリング
    img_tensor = torch.unsqueeze(
        torch.transpose(torch.tensor(img_np, dtype=torch.uint8), 0, 2), 0
    )
    kid_metric.update(img_tensor, real=False)

# 生成画像の読み込みとメトリックの更新
for filename in os.listdir(generated_dir2):
    img_path = os.path.join(generated_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img) / 255.0  # 画像のピクセル値を[0, 1]にスケーリング
    img_tensor = torch.unsqueeze(
        torch.transpose(torch.tensor(img_np, dtype=torch.uint8), 0, 2), 0
    )
    kid_metric.update(img_tensor, real=False)

# KIDを計算
kid_mean, kid_std = kid_metric.compute()

# 結果を表示
print("Kernel Inception Distance (KID) 平均:", kid_mean.item())
print("Kernel Inception Distance (KID) 標準偏差:", kid_std.item())
