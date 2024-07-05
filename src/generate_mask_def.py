import numpy as np

mask = np.ones((3, 64, 64), dtype=np.float32)
mask[:, 84:172, 84:172] = 0
np.save("mask.npy", mask)
