from clip_interrogator import Config, Interrogator
from PIL import Image

image = Image.open("/work/Data/in/BSDS300/images/test/19021.jpg").convert("RGB")
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
print(ci.interrogate(image))
