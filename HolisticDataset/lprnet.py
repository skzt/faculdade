import numpy as np
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dropout


class LPRNet:

    @staticmethod
    def convolutional_layers(inputLayer, inputShape):
        # Layers
        #  Argawal et al.{64, 64, 128,128, 256,256, 512, 512}
        # Spanhel et al. {32, 32, 32, 64, 64, 64, 128, 128, 128

        # Primeiro Conv Layer (3x[Conv => ReLu => BN])
        model = Conv2D(32, (3, 3), padding='same', input_shape=inputShape)(inputLayer)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(32, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(32, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        # Max Pooling (2 x 2)
        model = MaxPooling2D(strides=1)(model)

        # Segundo Conv Layer (3x[Conv => ReLu => BN])
        model = Conv2D(64, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(64, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(64, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        # Max Pooling (2 x 2)
        model = MaxPooling2D(strides=1)(model)

        # Terceiro Conv Layer (3x[Conv => ReLu => BN])
        model = Conv2D(128, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(128, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)
        model = Conv2D(128, (3, 3), padding='same', input_shape=inputShape)(model)
        model = Activation('relu')(model)
        model = BatchNormalization()(model)

        # Max Pooling (2 x 2)
        model = MaxPooling2D(strides=1)(model)

        return model

    @staticmethod
    def build(linha, coluna, canais, numClasses, numOutputs=1):
        inputShape = (linha, coluna, canais)
        inputLayer = Input(shape=inputShape)

        if numOutputs > 1:
            numbersOutput = []
            for number in range(numOutputs):
                output = LPRNet.convolutional_layers(inputLayer, inputShape)
                output = Flatten()(output)
                # Primeiro FC layer
                output = Dense(128)(output)
                output = Activation('relu')(output)
                output = Dropout(0.5)(output)

                # Segundo FC layer
                output = Dense(128)(output)
                output = Activation('relu')(output)
                output = Dropout(0.5)(output)

                # Softmax layer
                output = Dense(numClasses)(output)
                output = Activation("softmax", name=f"num{number+1}_output")(output)

                numbersOutput.append(output)
            return Model(inputs=inputLayer, outputs=numbersOutput, name="lprnet")
        else:
            output = LPRNet.convolutional_layers(inputLayer, inputShape)
            output = Flatten()(output)
            # Primeiro FC layer
            output = Dense(128)(output)
            output = Activation('relu')(output)
            output = Dropout(0.5)(output)

            # Segundo FC layer
            output = Dense(128)(output)
            output = Activation('relu')(output)
            output = Dropout(0.5)(output)

            # Softmax layer
            output = Dense(numClasses)(output)
            output = Activation("softmax", name=f"number_output")(output)

            return Model(inputs=inputLayer, outputs=output, name="lprnet")









