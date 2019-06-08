import os
from random import shuffle, randint
from time import strftime

import numpy as np
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm.auto import tqdm

# DIR_TREINO = "./realImages/Train"
DIR_TREINO = "C:/Dataset/train"
LR = 0.001
ROW_SIZE = 50  # 9
COL_SIZE = 50  # 6
EPOCHS = 10


def lprNet():
    # new_model = input_data(shape=[None, ROW_SIZE, COL_SIZE, 1], name='input')
    #
    # new_model = conv_2d(new_model, 32, 2, activation='relu')
    # new_model = max_pool_2d(new_model, 2)
    #
    # new_model = conv_2d(new_model, 64, 2, activation='relu')
    # new_model = max_pool_2d(new_model, 2)
    #
    # new_model = fully_connected(new_model, 32, activation='relu')
    # new_model = dropout(new_model, 0.8)
    #
    # new_model = fully_connected(new_model, 2, activation='softmax')
    # new_model = regression(new_model,
    #                        optimizer='adam',
    #                        learning_rate=LR,
    #                        loss='categorical_crossentropy',
    #                        name='targets')

    new_model = input_data(shape=[None, ROW_SIZE, COL_SIZE, 1], name='input')

    new_model = conv_2d(new_model, 32, 5, activation='relu')
    new_model = max_pool_2d(new_model, 5)

    new_model = conv_2d(new_model, 64, 5, activation='relu')
    new_model = max_pool_2d(new_model, 5)

    new_model = conv_2d(new_model, 128, 5, activation='relu')
    new_model = max_pool_2d(new_model, 5)

    new_model = conv_2d(new_model, 64, 5, activation='relu')
    new_model = max_pool_2d(new_model, 5)

    new_model = conv_2d(new_model, 32, 5, activation='relu')
    new_model = max_pool_2d(new_model, 5)

    new_model = fully_connected(new_model, 1024, activation='relu')
    new_model = dropout(new_model, 0.8)

    new_model = fully_connected(new_model, 2, activation='softmax')
    new_model = regression(new_model, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

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
    # trainX = trainX / 255.00

    trainY = np.array([img[1] for img in training_data])
    print(trainY[:12])

    testX = np.array([img[0] for img in testing_data])
    testX = testX.reshape(-1, ROW_SIZE, COL_SIZE, 1)
    # testX = testX / 255.0

    testY = np.array([img[1] for img in testing_data])

    return trainX, trainY, testX, testY


def treinar():
    MODEL_NAME = "CatDog03"
    for _ in range(int(EPOCHS / 10)):
        X, Y, test_x, test_y = generateData()
        model.fit({'input': X},
                  {'targets': Y},
                  n_epoch=10,
                  validation_set=({'input': test_x}, {'targets': test_y}),
                  show_metric=True, run_id=MODEL_NAME)
        # TODO: testar com os test_x antes do proximo fit
    model.save(f"{MODEL_NAME}.model")


model = DNN(lprNet(), tensorboard_dir=f"./logs/{strftime('%d-%m-%Y')}-{strftime('%H-%M-%S')}")

treinar()
