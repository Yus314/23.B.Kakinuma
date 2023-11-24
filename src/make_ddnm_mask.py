import numpy as np

# 256x256のndarrayを作成
array = np.ones((256, 256))

# 条件に基づき一部を0にする
array[105:153, 105:153] = 0

# 作成したndarrayをnpyファイルに保存
np.save("output_array.npy", array)
