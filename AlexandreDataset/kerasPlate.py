import os
from random import shuffle, randint
from time import strftime

import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm.auto import tqdm

# DIR_TREINO = "./realImages/Train"
DIR_TREINO = "C:/Dataset/train"
LR = 0.001
ROW_SIZE = 50  # 9
COL_SIZE = 50  # 6
EPOCHS = 10


def lprNet():
    new_model = Sequential()

    new_model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(ROW_SIZE, COL_SIZE, 1)))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(5))

    new_model.add(Conv2D(64, kernel_size=(5, 5)))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(5))

    new_model.add(Conv2D(128, kernel_size=(1, 1)))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(1))

    new_model.add(Conv2D(64, kernel_size=(1, 1)))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(1))

    new_model.add(Conv2D(32, kernel_size=(1, 1)))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(1))

    new_model.add(Flatten())
    new_model.add(Dense(1024))
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.2))

    new_model.add(Dense(2, activation="softmax"))

    return new_model


def generateData():
    training_data = []
    testing_data = []

    # Algoritimo para Dataset Alexandre
    # for num_folder in range(10):
    #     test_labels = os.listdir(os.path.join(DIR_TREINO, str(num_folder)))
    #     train_labels = []
    #     # Loop responsavel por separar as labels de treino e test
    #     for i in range(15):
    #         index = randint(0, 17 - i)
    #         train_labels.append(test_labels[index])
    #         test_labels = test_labels[:index] + test_labels[index + 1:]
    #
    #     for img_name in train_labels:
    #             img = cv2.imread(os.path.join(DIR_TREINO, str(num_folder), img_name),
    #                              cv2.IMREAD_GRAYSCALE)
    #             img = cv2.resize(img, (9, 6))
    #             training_data.append([img, to_categorical(str(num_folder), 10, dtype='uint8')])
    #
    #     for img_name in test_labels:
    #             img = cv2.imread(os.path.join(DIR_TREINO, str(num_folder), img_name),
    #                              cv2.IMREAD_GRAYSCALE)
    #             img = cv2.resize(img, (9, 6))
    #             testing_data.append([img, to_categorical(str(num_folder), 10, dtype='uint8')])

    # Algoritmo para Dataset CadDog
    tmp = np.load("CatDog-50x50.npy")

    data = {}
    for i in tmp:
        data[i[1]] = i[0]
    #
    test_labels = os.listdir(DIR_TREINO)
    train_labels = []

    def binirizer(data, classes, dtype):
        label = np.zeros((classes,), dtype=dtype)
        # [1,0] -> CAT
        # [0, 1] -> DOG
        if data.upper() == 'CAT':
            label[0] = 1
        elif data.upper() == 'DOG':
            label[1] = 1
        return label

    # Loop responsavel por separar as labels de treino e test.
    # test_labels = []
    # for i in range(25000):

    print("\n\n\n\n\nIniciando randomização.")
    for i in tqdm(range(20000)):
        index = randint(0, 24999 - i)
        train_labels.append(test_labels[index])
        test_labels = test_labels[:index] + test_labels[index + 1:]
    print(f"Randomização concluida:\n"
          f"    train_labels: {len(train_labels)}\n"
          f"    test_labels: {len(test_labels)}\n\n\n")

    print("Iniciando leitura das imagens.")
    for img_name in tqdm(train_labels):
        training_data.append([data[img_name],
                             binirizer(
                                 img_name.split('.')[0],
                                 2,
                                 dtype='uint8')
                             ])

    for img_name in tqdm(test_labels):
        testing_data.append([data[img_name],
                             binirizer(
                                 img_name.split('.')[0],
                                 2,
                                 dtype='uint8')
                             ])
    print(f"Leitura concluida:\n"
          f"    training_data: {len(training_data)}\n"
          f"    testing_data: {len(testing_data)}\n\n\n")
    # TODO SHUFFLE THE ARRAY!!!!!!!!
    shuffle(training_data)
    shuffle(testing_data)

    trainX = np.array([img[0] for img in training_data])
    trainX = trainX.reshape(-1, ROW_SIZE, COL_SIZE, 1)
    trainX = trainX / 255.0

    trainY = np.array([img[1] for img in training_data])
    print(trainY[:12])
    print(trainY[0].shape)

    testX = np.array([img[0] for img in testing_data])
    testX = testX.reshape(-1, ROW_SIZE, COL_SIZE, 1)
    testX = testX / 255.0

    testY = np.array([img[1] for img in testing_data])

    return trainX, trainY, testX, testY


def treinar(epochs, tensorboard):
    for _ in range(int(epochs/10)):
        X, Y, test_x, test_y = generateData()
        model.fit(X, Y, batch_size=100, epochs=int(10), validation_data=(test_x, test_y), callbacks=[tensorboard])
        # TODO: testar com os test_x antes do proximo fit
    model.save("CatDog03.h5")


model = lprNet()

optimizer = Adam(lr=LR)

tensorBoard = TensorBoard(log_dir=f"./logs/{strftime('%d-%m-%Y')}-{strftime('%H-%M-%S')}")

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

treinar(EPOCHS, tensorBoard)
