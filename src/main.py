import json
import os

import numpy as np
import torch
from clip_interrogator import Config, Interrogator
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm

from sch_plus import randn_tensor
from utile import (calc_metrics, gen_text_embeddings, load_image, model_load,
                   save_masked_and_origin_image)

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)
# ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))


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


with open("./BSDS500_prompt.json", "r") as f:
    data = json.load(f)
# 画像読み込み
# image_dir: str = "../Data/in/BSDS500"
# for image_file in os.listdir(image_dir):
for item in data:

    # image_path: str = image_dir + "/" + image_file
    # image_path = "./00000.png"
    # test_image_path = "../Data/out//" + image_file
    # y = load_image(image_path, torch_device)
    y = load_image(item["image"], torch_device)
    image_path = item["image"]

    # BLIP でプロンプト作成
    # txt_image = Image.open(image_path)
    # txt_image = txt_image.resize((512, 512))
    # txt_image = np.array(txt_image)
    # txt_image[172:339, 172:339, :] = 0
    # txt_image = Image.fromarray(txt_image)
    # txt = ci.interrogate(txt_image)
    # print(txt)
    # 設定
    # prompt = [txt]
    txt = item["prompt"]
    prompt = [txt]

    left = 0
    up = 0
    height = 512  # Stable Diffusion標準の出力画像サイズ (高さ)
    width = 512  # Stable Diffusion標準の出力画像サイズ (幅)
    # num_inference_steps = 30  # ノイズ除去のステップ数
    num_inference_steps = 30  # ノイズ除去のステップ数
    guidance_scale = 7.5  # ガイダンスの強さ プロンプトなしなら0
    # guidance_scale = 0  # ガイダンスの強さ プロンプトなしなら0
    generator = torch.manual_seed(11)  # 潜在空間のノイズ生成のためのシード生成
    # generator = torch.manual_seed(18)  # 潜在空間のノイズ生成のためのシード生成
    batch_size = 1

    save_dir = "../Data/out/BSDS500_168_BLIP"
    save_z_dir = save_dir + "_z"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_z_dir):
        os.makedirs(save_z_dir)

    # save_masked_and_origin_image(image_path, prompt, save_dir, left, up)

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
    mask = torch.ones(1, 4, 64, 64).to(torch_device)
    # mask[:, :, 15:49, 15:49] = 0
    mask[:, :, 21:43, 21:43] = 0
    # mask[:, :, 19:45, 19:45] = 0
    # mask[:, :, 25:38, 25:38] = 0

    # save_latent(mask, "./masked/mask")

    mask = mask.to(torch_device)

    def A(z):
        return z * mask

    Ap = A

    with torch.no_grad():
        y = torch.cat([y[:, :, left:], y[:, :, :left]], 2)
        y = torch.cat([y[:, up:, :], y[:, :up, :]], 1)
        # y[:, 188:323, 188:323] = 0
        y[:, 172:340, 172:340] = 0
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

            eta = 0.85
            # eta = 0.5

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

        # save_latent(latents, "./changes/latent_" + str(i))

    # 画像の拡大縮小とVAEによる復号化
    latents = latents.float()
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents)

    # 生成画像の表示
    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    final_pil = np.array(Image.open(image_path).resize((512, 512)), np.uint8)
    # final_pil = np.array(Image.open(image_path).resize((256, 256)), np.uint8)
    iimages = Image.fromarray(images[0])
    # iimages.save(
    #     save_dir
    #     + "_z/"
    #     + image_file
    # + prompt[0][0 : min(len(prompt[0]), 20)]:
    # + "/my_"
    # + prompt[0][0 : min(len(prompt[0]), 20)]
    # + ".png"
    # )
    final_pil[172:340, 172:340, :] = images[0, 172:340, 172:340, :]
    ffinal_pil = Image.fromarray(final_pil)
    # pil_images = [Image.fromarray(image) for image in images]
    # pil_images[0].save(
    ffinal_pil.save(
        save_dir
        + "/"
        + os.path.basename(item["image"])
        # + image_file
        # + prompt[0][0 : min(len(prompt[0]), 20)]:
        # + "/my_"
        # + prompt[0][0 : min(len(prompt[0]), 20)]
        # + ".png"
    )
# calc_metrics(save_dir, prompt
