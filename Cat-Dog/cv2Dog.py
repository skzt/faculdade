# 1106200195976514
import matplotlib.pyplot as plt
import numpy as np
from keras.backend import clear_session
from keras.models import load_model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from time import strftime

TEST_DIR = "./cat-dog-12500-GrayScale-(50, 50)-test.npy"
MODEL_NAME = "cat-dog-3-5"
ROW_SIZE = 50
COL_SIZE = 50
LR = 0.001

convnet = input_data(shape=[None, ROW_SIZE, COL_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

modelo = tflearn.DNN(convnet, tensorboard_dir=f"../logs/alexandre/{MODEL_NAME}-{strftime('%d-%m-%Y')}-{strftime('%H-%M-%S')}")

model = modelo.load(f"{MODEL_NAME}.model")
print(model)

# model = load_model(f"{MODEL_NAME}.model")

fig = plt.figure()
num = 0

testing_data = np.load(TEST_DIR, allow_pickle=True)

for img_data in testing_data[:12]:
	img = img_data[0]
	orig = img
	img = img.reshape(-1, ROW_SIZE, COL_SIZE, 1)
	y = fig.add_subplot(3, 4, num + 1)
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
