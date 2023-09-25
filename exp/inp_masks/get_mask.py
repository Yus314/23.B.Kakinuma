# import cv2
import numpy as np
from PIL import Image

# mask = cv2.imread("./tmp.png")
mask = Image.open("./tmp.png")
mask = np.array(mask)
print(mask.shape)
mask = mask[:, :, 0]
mask = (mask == 255) * 1
print(mask.shape)
np.save("mask.npy", mask)
