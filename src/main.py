import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm

from sch_plus import randn_tensor
from utile import (
    gen_text_embeddings,
    load_image,
    model_load,
)


parser = argparse.ArgumentParser()
parser.add_argument("--mask_size", type=int, default=168)
parser.add_argument("--use_BLIP", type=bool, default=True)
parser.add_argument("--image_dir", type=str, default="./ImageNet_o_prompt.json")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--save_dir_option", type=str, default="")
parser.add_argument("--num_inference_steps", type=int, default=30)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--eta", type=float, default=0.85)
parser.add_argument("--latent_mask_min", type=int, default=15)
parser.add_argument("--latent_mask_max", type=int, default=49)


args = parser.parse_args()
mask_size = args.mask_size
use_BLIP = args.use_BLIP
image_dir = args.image_dir
seed = args.seed
generator = torch.manual_seed(seed)
height = args.height
width = args.width
save_dir_option = args.save_dir_option
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
batch_size = args.batch_size
eta = args.eta
latent_mask_min = args.latent_mask_min
latent_mask_max = args.latent_mask_max


# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)


to_tensor_rfm = tfms.ToTensor()


def save_latent(latent, name):
    image = latent_to_pil(latent).squeeze(0)
    print(image.shape)
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    images = (image * 255).round().astype("uint8")
    Image.fromarray(images).save(name + ".png")


def latent_to_pil(latents):
    latents = (1 / 0.18215) * latents
    # with torch.no_grad():
    #    image = vae.decode(latents)
    image = vae.decode(latents)
    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
    images = (image * 255).round().astype("uint8")
    images = to_tensor_rfm(images).to(torch_device)
    images = images.unsqueeze(0)
    return images


with open(image_dir, "r") as f:
    data = json.load(f)
for item in data:
    image_path = item["image"]
    y = load_image(image_path, torch_device)

    if use_BLIP:
        prompt = [item["prompt"]]
    else:
        prompt = [""]

    save_dir = f"../Data/out/{os.path.basename(os.path.dirname(image_path))}_{mask_size}_{'BLIP' if use_BLIP else 'NOBLIP'}_{latent_mask_min}-{latent_mask_max}_{save_dir_option}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # torchvision.transforms.ToTensorを使う
    to_tensor_tfm = tfms.ToTensor()

    # インペインティング
    mask = torch.ones(1, 4, height // 8, width // 8).to(torch_device)
    mask[:, :, latent_mask_min:latent_mask_max, latent_mask_min:latent_mask_max] = 0
    mask = mask.to(torch_device)

    def A(z):
        return z * mask

    Ap = A

    with torch.no_grad():
        y[
            :,
            height // 2 - mask_size // 2 : height // 2 + mask_size // 2 - 1,
            width // 2 - mask_size // 2 : width // 2 + mask_size // 2 - 1,
        ] = 0
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

            # 1. get previous step value (=t-1)
            prev_timestep = (
                t
                - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
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
            std_dev_t = eta * variance ** (0.5)

            # 6. compute "direction pointing to x_t" of formula (12)
            # from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
                0.5
            ) * pred_epsilon

            pred_original_sample = pred_original_sample - Ap(
                A(pred_original_sample) - y
            )

            # save_latent(y, "./masked/y" + str(i))
            # 7. compute x_t without "random noise" of formula (12)
            # from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            variance_noise = randn_tensor(
                noise_pred.shape,
                generator=generator,
                device=noise_pred.device,
                dtype=noise_pred.dtype,
            )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

            latents = prev_sample

    # 画像の拡大縮小とVAEによる復号化
    latents = latents.float()
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents)

    # 生成画像の表示
    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    final_pil = np.array(Image.open(image_path).resize((height, width)), np.uint8)
    iimages = Image.fromarray(images[0])
    final_pil[
        height // 2 - mask_size // 2 : height // 2 + mask_size // 2 - 1,
        width // 2 - mask_size // 2 : width // 2 + mask_size // 2 - 1,
        :,
    ] = images[
        0,
        height // 2 - mask_size // 2 : height // 2 + mask_size // 2 - 1,
        width // 2 - mask_size // 2 : width // 2 + mask_size // 2 - 1,
        :,
    ]
    ffinal_pil = Image.fromarray(final_pil)
    ffinal_pil.save(save_dir + "/" + os.path.basename(image_path))
