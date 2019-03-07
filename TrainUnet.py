import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


# def unet(pretrained_weights=None, input_size=(200, 200, 1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(input=inputs, output=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model
def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

def unet(sz=(512, 512, 1)):
    x = Input(sz)
    inputs = x

    # down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64

    # bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1

    # upsampling
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1

        # classification
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[mean_iou])

    return model

from PIL import Image
import numpy as np
model=unet()
fil1 = "C:\\Users\\sakshigarg\\Desktop\\Prerna_BTP_Liver disease_2018\\Dataset\\dataset\\"
X = []
y = []
for i in range(0,131):
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

X=np.array(X)
y=np.array(y)
X_new = np.zeros((393,512,512,1))
y_new = np.zeros((393,512,512,1))
for i in range(X.shape[0]):
    X_new[i] = X[i].reshape((512,512,1))
    y_new[i] = y[i].reshape((512,512,1))
print(X_new.shape)
print(y_new.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, y_new, test_size=0.1, random_state=42)

print(X_train.shape)
model.fit(X_train, Y_train, batch_size=1, nb_epoch=1000,validation_data=(X_test, Y_test))