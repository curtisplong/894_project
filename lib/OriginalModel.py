PYTHONHASHSEED=0.5

import os
import sys
import glob
import time
import imageio

import numpy as np
np.random.seed(1)

import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imresize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint 

from lib import utils
from lib import Dataset

RANDOM_SEED=159


class OriginalModel(object):

    optimizer = 'rmsprop'
    epochs = 500

    def __init__(self, data, batch = 8, name = 'default', epochs = 500, optimizer = 'rmsprop', activation = 'relu', dropout = 0.35):
        self.ds = data 
        self.batch_size = batch
        self.name = name
        self.epochs = epochs 
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout 

        train_gen = ImageDataGenerator(rescale=1. / 255, 
                                       horizontal_flip=True, 
                                       rotation_range=20,
                                       vertical_flip = True
                                      )

        val_gen = ImageDataGenerator(rescale=1. / 255,
                                    horizontal_flip=True, 
                                       rotation_range=20,
                                       vertical_flip = True)

        test_gen = ImageDataGenerator(rescale=1. / 255,
                                    horizontal_flip=True, 
                                       rotation_range=20,
                                       vertical_flip = True)

        self.train_generator = train_gen.flow(np.array(self.ds.X_train), 
                                         self.ds.y_train, 
                                         batch_size=self.batch_size) 
        self.validation_generator = val_gen.flow(np.array(self.ds.X_val), self.ds.y_val)
        self.test_generator = test_gen.flow(np.array(self.ds.X_test), self.ds.y_test, batch_size=1)
        self.build()


    def build(self):
        # Model Building
        self.model = Sequential()
        self.model.add(Conv2D(32, activation='relu', kernel_size=3,
                         input_shape=(self.ds.img_height, self.ds.img_width, 3), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(3,3)))
        self.model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same'))
        self.model.add(MaxPooling2D(pool_size=(3,3)))
        self.model.add(Conv2D(64, activation='relu', kernel_size=3, padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(4, activation='softmax'))

        #opt = RMSprop(lr=0.01)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())


    def train(self):
        self.checkpoint = ModelCheckpoint(filepath = os.path.join("weights", self.name + "-best.hdf5"), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        self.hist = self.model.fit_generator(
            self.train_generator, 
            validation_data=self.validation_generator,
            steps_per_epoch=len(self.ds.X_train) // self.batch_size,
            validation_steps=len(self.ds.X_test) // self.batch_size,
            epochs=self.epochs,
            class_weight = [1.5, 2.0, 0.5, 2.0],
            callbacks = [self.checkpoint]
        )

        #Plot Testing and Validation MSE and Loss
        utils.plot_val_loss(self.hist.history, filename=os.path.join("results", self.name + "-val_loss.png"))
        utils.plot_val_acc(self.hist.history, filename=os.path.join("results", self.name + "-val_acc.png"))


    def test(self):
        y_pred = self.model.predict_generator(self.test_generator, steps = len(self.ds.X_test))
        #y_pred = self.model.predict(self.ds.X_test)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.ds.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        utils.plot_confusion_matrix(cnf_matrix, classes=self.ds.categories,
                              title='Confusion matrix, without normalization')
        filename = self.name + "-confusion.png"
        plt.savefig(os.path.join("results", filename))
        plt.close()

#############################################################################################

