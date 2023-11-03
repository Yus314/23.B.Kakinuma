import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

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
# 潜在空間を画像空間にデコードするためのVAEモデルを読み込む

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="vae",
    token="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",
)

# トークナイズとテキストのエンコード用に、tokenizerと、text_encoderを読み込む

tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="tokenizer",
    token="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",
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="text_encoder",
    token="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",
)

# 潜在空間を生成するためのU-Netモデルの指定

unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="unet",
    token="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",
)

# ノイズスケジューラの指定

scheduler = DDIMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="scheduler",
    token="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",
)

# モデルをGPUへ移す

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# 画像読み込み
image_dir: str = "/work/Data/in/BSDS300/images/test"
image_file: str = "291000.jpg"
image_path: str = image_dir + "/" + image_file
y: torch.tensor = load_image(image_path, torch_device)
# y: torch.tensor = load_image("/work/src/my_jizo.png", torch_device)

# 設定
prompt = ["horse"]
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
# mask[:, :, 26:30, 26:38] = 0
# mask[:, :, 34:38, 26:38] = 0
# mask[:, :, 29:35, 26:38] = 0

# mask[:, :, 26:38, 26:38] = 0
# mask[:, :, 27:37, 27:37] = 1

# mask[:, :, 32:38, 26:38] = 0

mask[:, :, 26:38, 26:38] = 0
# mask[:, :, 28:36, 28:36] = 0  # 　昔
# mask = np.load("/work/src/latent_mask.np.npy")
# mask = torch.from_numpy(mask.astype(np.float32)).clone()
mask = mask.to(torch_device)


def A(z):
    return z * mask


Ap = A

with torch.no_grad():
    y = torch.cat([y[:, :, left:], y[:, :, :left]], 2)
    y = torch.cat([y[:, up:, :], y[:, :up, :]], 1)
    y[:, 210:305, 210:305] = 0
    # y[:, 240:305, 210:305] = 0
    # y[:, 260:305, 210:305] = 0
    # y[:, 210:305, 210:305] = ddnm_y[:, 210:305, 210:305]

    # for i in range(3):
    #   y[i, 210:305, 210:305] = torch.mean(y[i])
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
        # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 1. get previous step value (=t-1)
        prev_timestep = (
            t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )

        # 2. compute alphas, betas
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(torch_device)
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            latents - beta_prod_t ** (0.5) * noise_pred
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = noise_pred

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = scheduler._get_variance(t, prev_timestep)
        std_dev_t = 0.0 * variance ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # pred_original_sample = pred_original_sample - Ap(A(pred_original_sample) - y)

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        # latents = (prev_sample,)
        latents = prev_sample

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
pil_images[0].save(save_dir + prompt[0] + "/my_" + prompt[0] + "2.4" + ".png")
