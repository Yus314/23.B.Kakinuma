
import os
import copy

import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm

from sch_plus import randn_tensor
from utile import (
    load_image,
    gen_text_embeddings,
    model_load,
    save_masked_and_origin_image,
)
# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# load image
#image_path: str = "./42049.jpg"
#y = load_image(image_path, torch_device)

image_path: str = "./document.png"
y = load_image(image_path, torch_device)
y = y[0:3,:,:]

#import pdb; pdb.set_trace()


y2 = y.clone().detach().cpu().permute(1, 2, 0).numpy()
y2 = (y2 * 255).round().astype("uint8")
y3 = Image.fromarray(y2)
y3.save("input.png")

#import pdb; pdb.set_trace()

# モデルを読み込む
vae, tokenizer, text_encoder, unet, scheduler = model_load(torch_device)

# yをエンコード
y = torch.unsqueeze(y, dim=0)

y_old = y.clone().detach()

y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean

latents = y.clone().detach()

latents = latents.float()
latents = 1 / 0.18215 * latents

#latents=torch.randn_like(latents)

with torch.no_grad():
    image = vae.decode(latents)
image = (image.sample / 2 + 0.5).clamp(0, 1)

y_new = image.clone().detach()


image = image.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
image = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
image.save("output.png")

#print( 10*torch.log10(1/torch.mean((y_old-y_new)**2)))



for param in vae.decoder.parameters():
    param.requires_grad=False
latents.requires_grad=True

optimizer = torch.optim.Adam(params=[latents],lr=0.1)

print(10*torch.log10(1/torch.nn.functional.mse_loss(y_old, y_new)))

#import pdb; pdb.set_trace()
for i in tqdm(range(600)):
    optimizer.zero_grad()
    outputs = vae.decode(latents)
    outputs = (outputs.sample / 2 + 0.5).clamp(0, 1)
    loss = torch.nn.functional.mse_loss(y_old, outputs)
    loss.backward()
    optimizer.step()
    #print(loss.item())

print(10*torch.log10(1/torch.nn.functional.mse_loss(y_old, outputs)))

#y_new2 = outputs.clone().detach()
image = outputs.detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
image = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
image.save("output_2.png")