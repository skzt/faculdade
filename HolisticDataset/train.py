from time import strftime

import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from lprnet import LPRNet

# from HolisticDataset.lprnet import LPRNet
# Comentario pra vida: numpay array shape = (LINHAS, COLUNAS)

IMG_SIZE = 64  # 64 x 64
CLASSES_SIZE = 10  # 0 1 2 3 4 5 6 7 8 9
CLASSES = '0123456789'
OUTPUTS = 1  # Número de saidas
RANDOM_SEED = 8495

LEARNING_RATE = 0.001
EPOCHS = 1
BS = 32

training_data = np.load("hlp-training_data-22800-Labeled-GrayScale-(64,9).npy")
# 20% Para teste (4560)
# 80% Para treino (18240)
train = training_data[:-4560]
test = training_data[-4560:]
del training_data

trainX = np.array([data[0] for data in train])
trainY = np.array([data[1] for data in train])

testX = np.array([data[0] for data in test])
testY = np.array([data[1] for data in test])

del train
del test

trainX = trainX.reshape(-1, IMG_SIZE, trainX[0].shape[1], 1)
testX = testX.reshape(-1, IMG_SIZE, testX[0].shape[1], 1)

trainX = trainX / 255.0
testX = testX / 255.0
print(f"Iniciando Treino:\n Train data shape = {trainX.shape}"
      f"\n Train label shape = {trainY.shape}"
      f"\n Test data shape = {testX.shape}"
      f"\n Test label shape = {testY.shape}")

model = LPRNet.build(trainX.shape[1], trainX.shape[2], 1, CLASSES_SIZE, OUTPUTS)

optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{strftime('%d-%m-%Y')}-{strftime('%H-%M-%S')}")

model.fit(trainX,
          trainY,
          epochs=EPOCHS,
          verbose=1,
          callbacks=[tensorboard],
          validation_data=(testX, testY)
          )

model.save(f"LR-LPR-("
           f"{trainX.shape[0] + testX.shape[0]}, {trainX.shape[1]}, {trainX.shape[2]}"
           f")-{LEARNING_RATE}"
           f"-HLP-dataset-"
           f"{strftime('%d-%m-%Y')}"
           f".h5")
