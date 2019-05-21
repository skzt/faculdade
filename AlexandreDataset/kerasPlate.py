import numpy as np
import cv2
import os
from time import strftime
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
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import os
from tqdm.auto import tqdm
from keras.utils import to_categorical
from keras.models import load_model
from random import shuffle

DIR_TREINO = "./realImages/Train"

data = []
for num_folder in range(10):
	for img_label in tqdm(os.os.listdir(DIR_TREINO)):
	    for img_name in tqdm(os.listdir(os.path.join(DIR_TREINO, img_label))):
	        img = cv2.imread(os.path.join(DIR_TREINO, img_label, img_name),
	                         cv2.IMREAD_GRAYSCALE)
	        img = cv2.resize(img, (9, 6))
	        data.append([img, to_categorical(img_label, 10, dtype='uint8')])
#%%
#TODO SHUFFLE THE ARRAY!!!!!!!!
shuffle(data)
data = np.array(data)

#%%
np.save(f"alexandre-real-{data.shape[0]}-GrayScale-(9,6).npy", data)



LR = 0.001
#IMG_SIZE = 64
ROW_SIZE = 9
COL_SIZE = 6
EPOCHS = 10

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
tensorBoard = TensorBoard(log_dir=f"./logs/{strftime('%d-%m-%Y')}-{strftime('%H-%M-%S')}")
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs = EPOCHS, validation_data=(testX, testY), callbacks=[tensorBoard])

model.save("Teste06.h5")


