
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal, ndimage, misc

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Dropout, GaussianNoise
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Convolution3D, UpSampling3D, MaxPooling3D
from keras.models import Model
from keras import regularizers
from keras import backend as K_backend
from keras import objectives
from keras.datasets import mnist
from keras.utils import np_utils

# Need this because otherwise the progbar freezes my Jupyter
from keras_tqdm import TQDMCallback, TQDMNotebookCallback


class SimpleAE3D(object):
    def __init__(self, input_shape=(48, 48, 48, 1), n_classes=10, n_filters=4):
        """
        5D tensor with shape: (samples, channels, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th' or
        5D tensor with shape: (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.
        """
        #         input_img = Input(shape=(1, 28, 28, 28)) # th =(nChan, nFrames, xPix, yPix) or (nChan, z, x, y)
        input_img = Input(shape=input_shape)  # th =(nChan, nFrames, xPix, yPix) or (nChan, z, x, y)
        img_chans = 1

        x = Convolution3D(n_filters, 3, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
        x = Convolution3D(n_filters * 8, 3, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling3D((2, 2, 2), border_mode='same')(x)
        x = Convolution3D(n_filters * 16, 3, 3, 3, activation='relu', border_mode='same')(x)
        self.z = MaxPooling3D((2, 2, 2), border_mode='same')(x)

        # at this point the representation is (4, 4, 4, 8) i.e. 512-dimensional

        flat = Flatten()(self.z)
        classer_base = Dense(n_classes, init='normal', activation='softmax', name='classer_output')(flat)

        x = Convolution3D(n_filters * 16, 3, 3, 3, activation='relu', border_mode='same')(self.z)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(n_filters * 8, 3, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        #         x = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='valid')(x)
        x = UpSampling3D((2, 2, 2))(x)
        # smash the extra channels down
        x = Convolution3D(img_chans, 3, 3, 3, activation='sigmoid', border_mode='same')(x)
        #         print(input_shape)
        #         x = Dense(input_shape)(x) # fudge to fix output shape
        self.decoded = x
        self.autoencoder = Model(input_img, self.decoded)

        self.classifier = Model(input_img, classer_base)
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

