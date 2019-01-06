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


class Dataset(object):
    imagedir = "dataset-master/JPEGImages/"
    labelsfile = "dataset-master/labels.csv"
    img_width = 160
    img_height = 120
    categories = []
    labels = pd.DataFrame()

    def __init__(self, img_width=160, img_height=120):
        self.labels = pd.read_csv(self.labelsfile)
        self.labels = self.labels.loc[:, ['Image', 'Category']]
        self.labels['Image'] = pd.to_numeric(self.labels['Image'])
        self.labels = self.labels.dropna()
        self.labels = self.labels[~self.labels.Category.str.contains(",")]
        self.labels = self.labels[~self.labels.Category.str.contains("BASOPHIL")]
        self.class_weight = compute_class_weight('balanced', np.unique(self.labels['Category']), self.labels['Category']) 
        self.labels = pd.get_dummies(self.labels, columns=['Category'])
        #print(class_weight)

        self.labels['filename'] = self.labels.Image.apply(lambda x: self.imagedir + "BloodImage_" + str(x).zfill(5) + ".jpg")
        # Filter for only image files that actually exist
        self.labels = self.labels[self.labels['filename'].apply(lambda x: os.path.isfile(x))]

        # Load and resize self.images
        self.images = []
        for x in self.labels['filename']:
            img = imageio.imread(x, as_gray = False)
            img = imresize(img, (self.img_height,self.img_width))
            self.images.append(img)

        self.images = np.array(self.images)
        print(self.images.shape)
        #print(self.images[0])

        #plt.imshow(self.images[0])
        #plt.show()

        # Reduce to only categories
        self.categories = ['Category_EOSINOPHIL', 'Category_LYMPHOCYTE', 'Category_MONOCYTE', 'Category_NEUTROPHIL']
        self.y = self.labels.loc[:,self.categories].values

        #print(str(images.shape) + ', ' + str(y.shape))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.y, test_size=0.2, stratify=self.y, random_state=159)
        #print(str(X_split.shape) + ', ' + str(y_split.shape))
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, stratify=self.y_train, random_state=159)


    def stats(self):
        # Category_BASOPHIL  Category_EOSINOPHIL  Category_LYMPHOCYTE  Category_MONOCYTE  Category_NEUTROPHIL
        #basophil = self.labels['Category_BASOPHIL'].sum()
        eosinophil = self.labels['Category_EOSINOPHIL'].sum()
        lymphocyte = self.labels['Category_LYMPHOCYTE'].sum()
        monocyte = self.labels['Category_MONOCYTE'].sum()
        neutrophil = self.labels['Category_NEUTROPHIL'].sum()
        count = eosinophil + lymphocyte + monocyte + neutrophil 
        print(self.labels.describe())
        #Basophil   {bas}
        print("""
        Eosinophil {eos}
        Lymphocyte {lym}
        Neutrophil {neu}
        Monocyte   {mono}
        """.format(
            #bas=basophil, 
            eos=eosinophil, lym=lymphocyte, neu=neutrophil,
                   mono=monocyte))

        #train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        print("Train:      " + str(self.X_train.shape) + ', ' + str(self.y_train.shape))
        print("Validation: " + str(self.X_val.shape) + ', ' + str(self.y_val.shape))
        print("Test:       " + str(self.X_test.shape) + ', ' + str(self.y_test.shape))

