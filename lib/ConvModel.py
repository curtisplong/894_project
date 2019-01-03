import os
import sys
import glob
import time
import imageio
import itertools
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imresize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint

from lib import Dataset 
from lib import utils


class ConvModel(object):
    convolutions = [32, 64]
    pooling = [0, 2]
    mlp = [100]
    name = "AnonymousModel"
    epochs = 30
    optimizer = 'adam'
    activation = 'relu'
    dropout = 0.25

    def __init__(self, data, convolutions = [32, 64], pooling = [0,2], mlp = [100], batch = 8, name = 'default', epochs = 30, optimizer = 'adam', activation = 'relu', dropout = 0.25):
        self.ds = data 
        self.convolutions = convolutions
        self.mlp = mlp
        self.batch_size = batch
        self.name = name
        self.epochs = epochs 
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout 

        self.train_gen = ImageDataGenerator(rescale = 1. / 255,
                                            horizontal_flip=True, 
                                            rotation_range=20, 
                                            vertical_flip = True )
        self.train_gen.fit(self.ds.X_train)

        self.val_gen = ImageDataGenerator(rescale = 1. / 255 )
        self.val_gen.fit(self.ds.X_val)

        self.test_gen = ImageDataGenerator(rescale = 1. / 255 )
        self.test_gen.fit(self.ds.X_test)

        self.train_generator = self.train_gen.flow(np.array(self.ds.X_train), self.ds.y_train, batch_size=self.batch_size) 
        self.validation_generator = self.val_gen.flow(np.array(self.ds.X_val), self.ds.y_val, batch_size=self.batch_size)
        self.test_generator = self.test_gen.flow(np.array(self.ds.X_test), self.ds.y_test, batch_size=1)
        self.build()


    def build(self):
        # Model Building
        self.model = Sequential()
        for i,j in enumerate(self.convolutions):
            self.model.add(Conv2D(j, activation=self.activation, kernel_size=3,
                         input_shape=(self.ds.img_height, self.ds.img_width, 3), padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros'))
            # if pooling specified for this layer
            if self.pooling[i]:
                self.model.add(MaxPooling2D(pool_size=(self.pooling[i],self.pooling[i])))

        self.model.add(Flatten())
        for y in self.mlp:
            self.model.add(Dense(y, activation=self.activation, kernel_initializer='TruncatedNormal', bias_initializer='zeros'))

        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(4, activation='softmax'))

        weightsfile = self.name + '-best.hdf5'
        if(os.path.isfile(weightsfile)):
            print("Loading weights...")
            self.model.load_weights(weightsfile)

        #opt = RMSprop(lr=0.01)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())


    def train(self):
        self.checkpoint = ModelCheckpoint(filepath = self.name + "-best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.hist = self.model.fit_generator(
            self.train_generator, 
            validation_data=self.validation_generator,
            steps_per_epoch=len(self.ds.X_train) // self.batch_size,
            validation_steps=len(self.ds.X_val) // self.batch_size,
            epochs=self.epochs,
            class_weight = self.ds.class_weight,
            callbacks = [self.checkpoint]
        )

        #Plot Testing and Validation MSE and Loss
        plt.figure()
        utils.plot_val_loss(self.hist.history)
        plt.savefig(self.name + "-val_loss.png")
        plt.close()

        plt.figure()
        utils.plot_val_acc(self.hist.history)
        plt.savefig(self.name + "-val_acc.png")
        plt.close()


    def test(self):
        y_pred = self.model.predict_generator(self.test_generator, steps = len(self.ds.X_test))
        #y_pred = self.model.predict(self.ds.X_test)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.ds.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        utils.plot_confusion_matrix(cnf_matrix, classes=self.ds.categories,
                              title='Confusion matrix, without normalization')
        plt.savefig(self.name + "-confusion.png")
        plt.close()

        #self.model.save_weights('curtis300.h5')


#############################################################################################


