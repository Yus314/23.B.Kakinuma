import statistics

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data_num = range(1, 11)


font_path = "/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf"
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams["font.family"] = font_prop.get_name()

word = [
    0.17119653522968292,
    0.05856965854763985,
    0.11851149797439575,
    0.1484045386314392,
    0.06574076414108276,
    0.10801710933446884,
    0.13221167027950287,
    0.11064456403255463,
    0.08979301154613495,
    0.15613576769828796,
]
print(statistics.mean(word))
blip = [
    0.16983668506145477,
    0.06480726599693298,
    0.11455740034580231,
    0.15294601023197174,
    0.06905383616685867,
    0.10686492919921875,
    0.13169458508491516,
    0.10992741584777832,
    0.08979301154613495,
    0.15942911803722382,
]
noword = [
    0.16882635653018951,
    0.06439760327339172,
    0.11631052941083908,
    0.147477388381958,
    0.06561542302370071,
    0.10742662847042084,
    0.13178136944770813,
    0.10936391353607178,
    0.09057319909334183,
    0.15575645864009857,
]
print(statistics.mean(blip))
print(statistics.mean(noword))
pic = [
    "horse",
    "airplane",
    "bird",
    "sea anemone",
    "mountain",
    "camel",
    "cactus",
    "racing car",
    "church",
    "rock wall",
]
plt.plot(data_num, word, marker="o", label="word")
plt.plot(data_num, blip, marker="x", label="BLIP")
plt.plot(data_num, noword, marker=".", label="noword")

plt.xticks(data_num, pic)
plt.xlim(0, 11)
plt.ylim(0.0, 0.2)

plt.title("LPIPS", fontsize=20)
plt.xlabel("target", fontsize=16)
plt.ylabel("LPIPS", fontsize=16)
plt.tick_params(labelsize=14)
plt.grid(True)
plt.legend()
plt.savefig("lpips_graph.jpg")
