from time import strftime

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tqdm.auto import tqdm

testing_data = np.load("hlp-dataset-TESTE-5726-GrayScale-(64,64).npy")

images = np.array([data[0] for data in testing_data])
labels = np.array([data[1] for data in testing_data])
model = load_model()

testing_data = []
for index, img in tqdm(enumerate(images)):
    numbers_array = [[], [], [], []]
    for linha in range(64):
        tmp = [[], [], [], []]  # [ [IMG_NUM1], [IMG_NUM2], [IMG_NUM3], [IMG_NUM4]]
        for coluna in range(5, 64):
            if coluna <= 13:
                tmp[0].append(img[linha][coluna])
            elif 24 >= coluna > 15:
                tmp[1].append(img[linha][coluna])
            elif 35 >= coluna > 26:
                tmp[2].append(img[linha][coluna])
            elif 46 >= coluna > 37:
                tmp[3].append(img[linha][coluna])
            else:
                continue
        numbers_array[0].append(tmp[0])
        numbers_array[1].append(tmp[1])
        numbers_array[2].append(tmp[2])
        numbers_array[3].append(tmp[3])
    numbers_array[0] = np.array(numbers_array[0])
    numbers_array[1] = np.array(numbers_array[1])
    numbers_array[2] = np.array(numbers_array[2])
    numbers_array[3] = np.array(numbers_array[3])

    predicted_plate = []
    lb = LabelBinarizer()
    lb.fit(list("0123456789"))
    for img_to_predict in numbers_array:
        model_out = model.predict(img_to_predict)
        predicted_index = model_out.argmax()
        predicted_plate.append((model_out[predicted_index]*100, lb.classes_[predicted_index]))
    plate = ''+predicted_plate[0][1]+predicted_plate[1][1]+predicted_plate[2][1]+predicted_plate[3][1]
    print(f"Original Plate: {labels[index]}\n"
          f"Predicte Plate: {plate}\n"
          f"Percentages: {predicted_plate[0][0]}% "
          f"{predicted_plate[1][0]}%"
          f"{predicted_plate[2][0]}%"
          f"{predicted_plate[3][0]}%")
