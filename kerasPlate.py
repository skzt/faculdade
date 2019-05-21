import numpy as np
import cv2
import os
from random import shuffle
from tqdm.auto import tqdm
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callback import Tensorboard


LR = 0.001
#IMG_SIZE = 64
ROW_SIZE = 9
COL_SIZE = 6
EPOCHS = 100

data = np.load("alexandre-real-180-GrayScale-(9,6) 2.npy")
train = data[:144]
test = data[144:]

trainX = np.array([img[0] for img in train])
trainX = trainX.reshape(-1, ROW_SIZE, COL_SIZE, 1)
trainX = trainX/255.00

trainY = np.array([img[1] for img in train])

testX = np.array([img[0] for img in test])
testX = testX.reshape(-1, ROW_SIZE, COL_SIZE, 1)
testX = testX/255.0

testY = np.array([img[1] for img in test])

print(train[0][1])
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(ROW_SIZE, COL_SIZE, 1)))
model.add(Activation('relu'))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))

model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(10, activation="softmax"))

optimizer = Adam(lr = LR)
Tensorboard(log_dir='./logs')
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs = EPOCHS, validation_data=(testX, testY))

model.save("Teste06.h5")


