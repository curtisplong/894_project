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

from keras.applications.vgg16 import VGG16
from keras import models
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint

from lib import Dataset 
from lib import utils


class VggModel(object):
    mlp = [100]
    name = "AnonymousModel"
    epochs = 30
    optimizer = 'adam'
    activation = 'relu'
    dropout = 0.5

    def __init__(self, data, mlp = [100], batch = 8, name = 'default', epochs = 3, optimizer = 'adam', activation = 'relu', dropout = 0.5):
        self.ds = data 
        self.mlp = mlp
        self.batch_size = batch
        self.name = name
        self.epochs = epochs 
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout 

        # for backwards compat
        self.train_generator = self.ds.train_generator
        self.validation_generator = self.ds.validation_generator
        self.test_generator = self.ds.test_generator

        self.build()


    def build(self):
        # Model Building
        self.vgg = VGG16(weights = 'imagenet', include_top=False)

        for layer in self.vgg.layers:
            layer.trainable = False

        self.vgg.summary()

        x = self.vgg.output
        x = GlobalAveragePooling2D()(x)
        for n in self.mlp:
            x = Dense(n, activation=self.activation)(x)
        x = Dropout(self.dropout)(x)
        predictions = Dense(4, activation='softmax')(x)

        # create new model composed of pre-trained network and new final layers
        # if you want to change the input size, you can do this with the input parameter below
        self.model = models.Model(input=self.vgg.input, output=predictions)

        self.model.summary()

        weightsfile = os.path.join("weights", self.name + '-best.hdf5')
        if(os.path.isfile(weightsfile)):
            print("Loading weights...")
            self.model.load_weights(weightsfile)

        #opt = RMSprop(lr=0.01)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())


    def train(self):
        self.checkpoint = ModelCheckpoint(filepath = os.path.join("weights", self.name + "-best.hdf5"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
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
        plt.savefig(os.path.join("results", self.name + "-confusion.png"))
        plt.close()

        #self.model.save_weights('curtis300.h5')


#############################################################################################


