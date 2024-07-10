import subprocess


mask_size = 168
use_BLIP = False
image_dir = "./prompt/BSDS500_prompt.json"
seed = 11
height = 512
width = 512
save_dir_option = "moveSeed"
num_inference_steps = 30
guidance_scale = 7.5
batch_size = 1
eta = 0.85
latent_mask_min = 21
latent_mask_max = 43

# 実行するPythonスクリプト
script_name = "main.py"

# 引数リストの作成
args = [
    "python",
    script_name,
    "--mask_size",
    str(mask_size),
    "--use_BLIP",
    str(use_BLIP),
    "--image_dir",
    image_dir,
    "--seed",
    str(seed),
    "--height",
    str(height),
    "--width",
    str(width),
    "--save_dir_option",
    save_dir_option,
    "--num_inference_steps",
    str(num_inference_steps),
    "--guidance_scale",
    str(guidance_scale),
    "--batch_size",
    str(batch_size),
    "--eta",
    str(eta),
    "--latent_mask_min",
    str(latent_mask_min),
    "--latent_mask_max",
    str(latent_mask_max),
]

# コマンドの実行
subprocess.run(args)
