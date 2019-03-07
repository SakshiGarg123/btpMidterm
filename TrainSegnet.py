from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils


def preprocess_input(X):
    return imagenet_utils.preprocess_input(X)


def to_categorical(y, nb_classes):
    num_samples = len(y)
    Y = np_utils.to_categorical(y.flatten(), nb_classes)
    return Y.reshape((num_samples, y.size / num_samples, nb_classes))


def SegNet(input_shape=(512, 512,1), classes=1):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(classes, 1, 1, border_mode="valid")(x)
    x = Reshape((input_shape[0] * input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model
from PIL import Image
def load_train():
    fil1 = "C:\\Users\\sakshigarg\\Desktop\\Prerna_BTP_Liver disease_2018\\Dataset\\dataset\\"
    X = []
    y = []
    for i in range(0, 131):
        sample = []
        img = Image.open(fil1 + 'enlarged_vol' + str(i) + 'a.png')
        arr = np.array(img)
        X.append(arr)
        # sample.append(arr)
        img = Image.open(fil1 + 'enlarged_vol' + str(i) + 'c.png')
        arr = np.array(img)
        X.append(arr)
        # sample.append(arr)
        img = Image.open(fil1 + 'enlarged_vol' + str(i) + 's.png')
        arr = np.array(img)
        # sample.append(arr)
        X.append(arr)

        sample = []
        img = Image.open(fil1 + 'enlarged_liver' + str(i) + 'a.png')
        arr = np.array(img)
        y.append(arr)
        img = Image.open(fil1 + 'enlarged_liver' + str(i) + 'c.png')
        arr = np.array(img)
        y.append(arr)
        img = Image.open(fil1 + 'enlarged_liver' + str(i) + 's.png')
        arr = np.array(img)
        y.append(arr)

    X = np.array(X)
    y = np.array(y)
    X_new = np.zeros((393, 512, 512, 1))
    y_new = np.zeros((393, 512, 512, 1))
    for i in range(X.shape[0]):
        X_new[i] = X[i].reshape((512, 512, 1))
        y_new[i] = y[i].reshape((512, 512, 1))
    print(X_new.shape)
    print(y_new.shape)
    return X_new,y_new

import os
import glob
import numpy as np
import cv2

input_shape = (512, 512,3)
nb_classes = 1
nb_epoch = 100
batch_size = 4

X, y = load_train() # need to implement, y shape is (None, 360, 480, nb_classes)
#X = preprocess_input(X)
Y = to_categorical(y, nb_classes)
model = SegNet(input_shape=input_shape, classes=nb_classes)
model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch)