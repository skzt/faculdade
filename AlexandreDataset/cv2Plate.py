# 1106200195976514
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.backend import clear_session
from tqdm.auto import tqdm
from random import randint

TEST_DIR = "./alexandre-real-20-GrayScale-(8, 5)-test.npy"
MODEL_NAME = "alexandre03-80"
ROW_SIZE = 8
COL_SIZE = 5

model = load_model(f"{MODEL_NAME}.h5")

erros = 0
acertos = 0
total = 0

fig = plt.figure()
num = 0

testing_data = np.load(TEST_DIR, allow_pickle=True)

for img_data in testing_data:
    img = img_data[0]
    orig = img
    img = img.reshape(-1, ROW_SIZE, COL_SIZE, 1)

    y = fig.add_subplot(4, 5, num + 1)
    mdl = model.predict([img])[0]

    str_label = f"Numero: {np.argmax(mdl)} ({img_data[1]})"

    print(str_label)
    print(["{0:.2f}%".format(i * 100) for i in mdl])

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    num += 1
plt.show()
clear_session()
