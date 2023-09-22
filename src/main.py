import os

import torch
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm

from utile import (
    gen_text_embeddings,
    load_image,
    model_load,
    save_masked_and_origin_image,
)

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)

# 画像読み込み
image_dir: str = "/work/Data/in/BSDS300/images/test"
image_file: str = "19021.jpg"
image_path: str = image_dir + "/" + image_file
y: torch.tensor = load_image(image_path, torch_device)

# 設定
prompt = ["cactus"]
left = 0
up = 0
height = 512  # Stable Diffusion標準の出力画像サイズ (高さ)
width = 512  # Stable Diffusion標準の出力画像サイズ (幅)
num_inference_steps = 30  # ノイズ除去のステップ数
guidance_scale = 7.5  # ガイダンスの強さ
generator = torch.manual_seed(11)  # 潜在空間のノイズ生成のためのシード生成
batch_size = 1

save_dir = "/work/Data/out/"
if not os.path.exists(save_dir + prompt[0]):
    os.makedirs(save_dir + prompt[0])

save_masked_and_origin_image(image_path, prompt, save_dir, left, up)

# テキストの準備
text_embeddings = gen_text_embeddings(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    torch_device=torch_device,
    batch_size=batch_size,
    prompt=prompt,
)


# スケジューラの準備
scheduler.set_timesteps(num_inference_steps)

# 潜在ベクトルの準備
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)
latents = latents * scheduler.init_noise_sigma

# torchvision.transforms.ToTensorを使う
to_tensor_tfm = tfms.ToTensor()


# インペインティング
mask = torch.ones(1, 4, 64, 64).to(torch_device)
mask[:, :, 28:35, 28:36] = 0


def A(z):
    return z * mask


Ap = A

with torch.no_grad():
    y = torch.cat([y[:, :, left:], y[:, :, :left]], 2)
    y = torch.cat([y[:, up:, :], y[:, :up, :]], 1)
    y[206:306, 206:306] = 0
    y = torch.unsqueeze(y, dim=0)
    y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean


# ループの実行
with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # 分類器不要のガイダンスを行う場合は、2回のフォワードパスを行わないように潜在ベクトルを拡張する
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # ノイズの残差を予測する
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

        # ガイダンスを実行する
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # ひとつ前のサンプルを計算する x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents = latents - Ap(A(latents) - y)


# 画像の拡大縮小とVAEによる復号化
latents = latents.float()
latents = 1 / 0.18215 * latents


with torch.no_grad():
    image = vae.decode(latents)

# 生成画像の表示
image = (image.sample / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save(save_dir + prompt[0] + "/my_" + prompt[0] + ".png")
