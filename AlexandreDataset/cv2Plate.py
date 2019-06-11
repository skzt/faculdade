# 1106200195976514
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tqdm.auto import tqdm
from random import randint

TEST_DIR = "../Datasets/alexandre/test"
MODEL_NAME = "alexandre02"
ROW_SIZE = 8
COL_SIZE = 5

model = load_model(f"{MODEL_NAME}.h5")
erros = 0
acertos = 0
total = 0

fig = plt.figure()
num = 0

for num_folder in range(10):
    for img_name in os.listdir(os.path.join(TEST_DIR, str(num_folder))):
        img = cv2.imread(os.path.join(TEST_DIR, str(num_folder), img_name), cv2.IMREAD_GRAYSCALE)
        orig = img

        img = cv2.resize(img, (ROW_SIZE, COL_SIZE))
        img = img.reshape(-1, ROW_SIZE, COL_SIZE, 1)

        y = fig.add_subplot(4, 5, num + 1)

        mdl = model.predict([img])[0]

        str_label = f"Numero: {np.argmax(mdl)} ({num_folder})"

        print(str_label)
        print(["{0:.2f}%".format(i * 100) for i in mdl])

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
        num += 1
plt.show()
os._exit(0)
