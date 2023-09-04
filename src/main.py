
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler,DDIMScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms

# For video display:
from IPython.display import HTML
from base64 import b64encode

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


# 潜在空間を画像空間にデコードするためのVAEモデルを読み込む
vae = AutoencoderKL.from_pretrained('vae')

# トークナイズとテキストのエンコード用に、tokenizerと、text_encoderを読み込む
tokenizer=CLIPTokenizer.from_pretrained('tokenizer')
text_encoder=CLIPTextModel.from_pretrained('text_encoder')

# 潜在空間を生成するためのU-Netモデルの指定
unet = UNet2DConditionModel.from_pretrained('unet')

# ノイズスケジューラの指定
scheduler = DDIMScheduler.from_pretrained('scheduler')
# モデルをGPUへ移す
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# 画像読み込み
y =Image.open("/work/Div2k/BSDS300/images/test/69020.jpg").resize((512,512))
y = tfms.functional.to_tensor(y)
y = y.to(torch_device)

# 設定
prompt = ["kangaroo"]
height = 512                        # Stable Diffusion標準の出力画像サイズ (高さ)
width = 512                         # Stable Diffusion標準の出力画像サイズ (幅)
num_inference_steps = 30           # ノイズ除去のステップ数
guidance_scale = 7.5                # ガイダンスの強さ
generator = torch.manual_seed(11)   # 潜在空間のノイズ生成のためのシード生成
batch_size = 1

# テキストの準備
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# スケジューラの準備
scheduler.set_timesteps(num_inference_steps)

# 潜在ベクトルの準備
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)
#latents = latents * scheduler.sigmas[0] # Need to scale to match k
latents = latents * scheduler.init_noise_sigma

# torchvision.transforms.ToTensorを使う
to_tensor_tfm = tfms.ToTensor()

"""


# カラー化
def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)
def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)  
A = lambda z: color2gray(z)
Ap = lambda z: gray2color(z)

"""

# インペインティング
import numpy as np
#loaded = np.load("/work/latent_mask.npy")
mask = torch.ones(1,4,64,64).to(torch_device)
mask[:,:,24:39,24:40]=1//2
mask[:,:,28:35,28:36]=0
print(mask)

A = lambda z: z*mask
Ap = A

with torch.no_grad():
    y[512//2-50:512//2+50,512//2-50:512//2+50]=0
    y = torch.unsqueeze(y,dim=0)
    y =0.1825* vae.encode(2*y-1).latent_dist.mean



# ループの実行
with autocast("cuda"):
  for i, t in tqdm(enumerate(scheduler.timesteps)):
    # 分類器不要のガイダンスを行う場合は、2回のフォワードパスを行わないように潜在ベクトルを拡張する
    latent_model_input = torch.cat([latents] * 2)
    #sigma = scheduler.sigmas[i]
    #latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # ノイズの残差を予測する
    with torch.no_grad():
      noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    # ガイダンスを実行する
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # ひとつ前のサンプルを計算する x_t -> x_t-1
    latents = scheduler.step(noise_pred,t,latents,return_dict=False)[0]
    #latents = latents - Ap(A(latents)-y)

""""
    with torch.no_grad():
        imgs = vae.decode(1/0.1825*latents).sample/2+0.5   
        imgs = imgs - Ap(A(imgs) - y)
        #imgs=imgs.to(torch.float16)
        latents =0.1825* vae.encode(2*imgs-1).latent_dist.mean
"""


# 画像の拡大縮小とVAEによる復号化
latents=latents.float()
latents = 1 / 0.18215 * latents


with torch.no_grad():
  image = vae.decode(latents)
  
# 生成画像の表示
image = (image.sample / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("my_kangaroo.png")