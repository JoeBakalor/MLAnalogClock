import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
import coremltools

MODEL_H5_NAME = "time.h5"
print('keras version ', keras.__version__)

INCLUDE_SECONDS_HAND = True
IMG_SIZE = [640,480,1]

def get_cnn_model(num_color_channels, img_width, img_height):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),input_shape=(img_width, img_height, num_color_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(12 + 60 + 60))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12 + 60 + 60))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model


def get_ann_model(num_color_channels, img_width, img_height):
    model = Sequential()
    model.add(Flatten(input_shape=(num_color_channels, img_width, img_height)))
    model.add(Dense(12 + 60 + 60))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model
