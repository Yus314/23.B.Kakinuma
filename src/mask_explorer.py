
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
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
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", token="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")

# トークナイズとテキストのエンコード用に、tokenizerと、text_encoderを読み込む
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 潜在空間を生成するためのU-Netモデルの指定
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", token="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")

# ノイズスケジューラの指定
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# モデルをGPUへ移す
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)




import os
input_dir = "/work/Div2k/BSDS300/images/train"
input_list = os.listdir(input_dir)
print(len(input_list))
out = torch.zeros(1,64,64)
out = out.to(torch_device)
with torch.no_grad():
    for one_im in input_list:
        y = Image.open("/work/Div2k/BSDS300/images/train/"+one_im).resize((512,512))
        y = tfms.functional.to_tensor(y)
        y = y.to(torch_device)
        y = y.unsqueeze(dim=0)
        x=y.clone()
        x[:,:,512//2-50:512//2+50,512//2-50:512//2+50]=0
        x = 0.1825* vae.encode(2*x-1).latent_dist.mean
        y = 0.1825* vae.encode(2*y-1).latent_dist.mean
        out = out + torch.mean((y-x)**2,dim=1)
input_dir = "/work/Div2k/BSDS300/images/test"
input_list = os.listdir(input_dir)    
with torch.no_grad():
    for one_im in input_list:
        y = Image.open("/work/Div2k/BSDS300/images/test/"+one_im).resize((512,512))
        y = tfms.functional.to_tensor(y)
        y = y.to(torch_device)
        y = y.unsqueeze(dim=0)
        x=y.clone()
        x[:,:,512//2-50:512//2+50,512//2-50:512//2+50]=0
        x = 0.1825* vae.encode(2*x-1).latent_dist.mean
        y = 0.1825* vae.encode(2*y-1).latent_dist.mean
        out = out + torch.mean((y-x)**2,dim=1)
out = out/300
show_num = out.cpu().numpy().copy()
def threshold_array(arr, threshold=0.5):
    """
    与えられた配列を指定された閾値で1または0に変換します。
    
    Parameters:
        arr (numpy.ndarray): 入力のNumPy配列。
        threshold (float, optional): 閾値。デフォルトは0.5です。
    
    Returns:
        numpy.ndarray: 与えられた配列を閾値で変換した結果のNumPy配列。
    """
    return (arr < threshold).astype(int)

true_mask = threshold_array(show_num,threshold=0.85)
print(out)
from pylab import *
im = imshow(true_mask.squeeze(),interpolation='nearest')
np.save('latent_mask',true_mask)
colorbar(im)
grid(True)
plt.savefig('view_mask.jpg')
show()
