import cv2
import os
import numpy as np
from keras.models import load_model
from tkinter.filedialog import askopenfilenames, askopenfilename
from tkinter import Tk

ROW_SIZE = 50
COLUMN_SIZE = 50
Tk().withdraw()

#Seleciona a imagem a ser testada.
files = askopenfilenames()

#seleciona o modelo já treinado.
model_path = askopenfilename()

model = load_model(model_path)

if files == '':
    exit()

for img_path in files:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original = img
    img = cv2.resize(img, (ROW_SIZE, COLUMN_SIZE))
    img = img.reshape(-1, ROW_SIZE, COLUMN_SIZE, 1)

    mdl_out = model.predict([img])
    print(mdl_out)
    label = 'CAT' if np.argmax(mdl_out[0] ) == 0 else 'DOG'

    cv2.imshow(label, original)
    cv2.waitKey(0)

cv2.destroyAllWindows()








