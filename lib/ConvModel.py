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
        self.pooling = pooling

        # for backwards compat
        self.train_generator = self.ds.train_generator
        self.validation_generator = self.ds.validation_generator
        self.test_generator = self.ds.test_generator

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
        self.checkpoint = ModelCheckpoint(filepath = os.path.join("weights", self.name + "-best.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        self.hist = self.model.fit_generator(
            self.ds.train_generator, 
            validation_data=self.ds.validation_generator,
            steps_per_epoch=len(self.ds.X_train) // self.batch_size,
            validation_steps=len(self.ds.X_val) // self.batch_size,
            epochs=self.epochs,
            class_weight = self.ds.class_weight,
            callbacks = [self.checkpoint]
        )

        #Plot Testing and Validation MSE and Loss
        utils.plot_val_loss(self.hist.history, filename=os.path.join("results", self.name + "-val_loss.png"))
        utils.plot_val_acc(self.hist.history, filename=os.path.join("results", self.name + "-val_acc.png"))


    def test(self):
        y_pred = self.model.predict_generator(self.ds.test_generator, steps = len(self.ds.X_test))
        #y_pred = self.model.predict(self.ds.X_test)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.ds.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        utils.plot_confusion_matrix(cnf_matrix, classes=self.ds.categories,
                              title='Confusion matrix, without normalization')
        plt.savefig(os.path.join("results", self.name + "-confusion.png"))
        plt.close()

        #self.model.save_weights('curtis300.h5')


#############################################################################################


