import matplotlib.pyplot as plt
import numpy as np
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

ROW_SIZE = 50
COL_SIZE = 50
LR = 1e-3


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
    new_model = regression(new_model, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                           name='targets')

    return new_model

model = DNN(lprNet()) 
MODEL_NAME = ("CatDog03.model")
test_data = np.load('test_data2.npy')
model.load(MODEL_NAME)

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(ROW_SIZE,COL_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
    print(model_out, str_label) 
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()