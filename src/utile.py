import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms as tfms
from transformers import CLIPTextModel, CLIPTokenizer


def model_load(device: str):
    # 潜在空間を画像空間にデコードするためのVAEモデルを読み込む
    vae = AutoencoderKL.from_pretrained("models/vae")

    # トークナイズとテキストのエンコード用に、tokenizerと、text_encoderを読み込む
    tokenizer = CLIPTokenizer.from_pretrained("models/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("models/text_encoder")

    # 潜在空間を生成するためのU-Netモデルの指定
    unet = UNet2DConditionModel.from_pretrained("models/unet")

    # ノイズスケジューラの指定
    scheduler = DDIMScheduler.from_pretrained("models/scheduler")

    # モデルをGPUに移す
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    return vae, tokenizer, text_encoder, unet, scheduler


def load_image(path: str, device: str):
    y: Image = Image.open(path).resize((512, 512))
    y = tfms.functional.to_tensor(y)
    y = y.to(device)
    return y


def gen_text_embeddings(tokenizer, text_encoder, torch_device, batch_size, prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def save_masked_and_origin_image(image_path, prompt, save_dir, left, up):
    zz: Image = Image.open(image_path).resize((512, 512))
    zz = np.array(zz)
    zz = np.concatenate([zz[:, left:], zz[:, :left]], 1)
    zz = np.concatenate([zz[up:, :], zz[:up, :]], 0)
    zzz = Image.fromarray(zz)
    zzz.save(
        save_dir
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/"
        + "GT"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )
    zz[210:305, 210:305, :] = 0
    zz = Image.fromarray(zz)
    zz.save(
        save_dir
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/"
        + "masked"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )


def calculate_ssim(save_dir, prompt, C1=0.01, C2=0.03):
    # Load the images
    x = np.array(
        Image.open(
            save_dir
            + "/"
            + prompt[0][0 : min(len(prompt[0]), 20)]
            + "/my_"
            + prompt[0][0 : min(len(prompt[0]), 20)]
            + ".png"
        )
    )
    y = np.array(
        Image.open(
            save_dir
            + "/"
            + prompt[0][0 : min(len(prompt[0]), 20)]
            + "/GT"
            + prompt[0][0 : min(len(prompt[0]), 20)]
            + ".png"
        )
    )

    # Calculate SSIM for each color channel
    ssim_r = ssim(x[:, :, 0], y[:, :, 0], data_range=255)
    ssim_g = ssim(x[:, :, 1], y[:, :, 1], data_range=255)
    ssim_b = ssim(x[:, :, 2], y[:, :, 2], data_range=255)

    # Calculate the mean SSIM
    mean_ssim = (ssim_r + ssim_g + ssim_b) / 3.0

    return mean_ssim


def calc_metrics(save_dir, prompt):
    img_my = cv2.imread(
        save_dir
        + "/"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/my_"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )
    img_GT = cv2.imread(
        save_dir
        + "/"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/GT"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )
    img_my_PL = Image.open(
        save_dir
        + "/"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/my_"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )
    img_GT_PL = Image.open(
        save_dir
        + "/"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + "/GT"
        + prompt[0][0 : min(len(prompt[0]), 20)]
        + ".png"
    )
    img_my_PL = (tfms.functional.to_tensor(img_my_PL) - 0.5) * 2
    img_my_PL.unsqueeze(0)
    img_GT_PL = (tfms.functional.to_tensor(img_GT_PL) - 0.5) * 2
    img_GT_PL.unsqueeze(0)
    print("PSNR: ", cv2.PSNR(img_my, img_GT))
    loss_fn_alex = lpips.LPIPS(net="alex")
    d = loss_fn_alex(img_my_PL, img_GT_PL)
    print("LPIPS: ", float(d[0][0][0][0]))
    ssim_value = calculate_ssim(save_dir, prompt)
    print("SSIM:", ssim_value)
