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


def plot_val_acc(h):
    # Prepare plotting
    #fig_size = plt.rcParams["figure.figsize"]
    #plt.rcParams["figure.figsize"] = [xsize, ysize]
    #fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

   #  summarize history for accuracy
    #plt.subplot(211)
    plt.plot(h['acc'])
    plt.plot(h['val_acc'])
    plt.title('Training vs Validation MSE using Scaled Data')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.draw()


def plot_val_loss(h):
    # summarize history for loss
    #plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss using Scaled Data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.savefig("val_loss.png")
    plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return


class Dataset(object):
    imagedir = "dataset-master/JPEGImages/"
    labelsfile = "dataset-master/labels.csv"
    img_width = 160 
    img_height = 120
    categories = []
    labels = pd.DataFrame()

    def __init__(self):
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


class ConvModel(object):
    convolutions = [128, 64, 64]
    mlp = [200, 50]
    name = "AnonymousModel"
    epochs = 75
    optimizer = 'adam'
    activation = 'relu'

    def __init__(self, data, c, m, b, name, epochs = 75, optimizer = 'adam', activation = 'relu'):
        self.ds = data 
        self.convolutions = c
        self.mlp = m
        self.batch_size = b
        self.name = name
        self.epochs = epochs 
        self.optimizer = optimizer
        self.activation = activation

        self.train_gen = ImageDataGenerator(rescale = 1. / 255,
                                            featurewise_center=True, 
                                            featurewise_std_normalization=True, 
                                            horizontal_flip=True, 
                                            rotation_range=20, 
                                            vertical_flip = True )
        self.train_gen.fit(self.ds.X_train)

        self.val_gen = ImageDataGenerator(rescale = 1. / 255,
                                          featurewise_center=True, 
                                          featurewise_std_normalization=True, 
                                          horizontal_flip=True, 
                                          rotation_range=20, 
                                          vertical_flip = True)
        self.val_gen.fit(self.ds.X_val)

        self.test_gen = ImageDataGenerator(rescale = 1. / 255,
                                           featurewise_center=True, 
                                           featurewise_std_normalization=True, 
                                           horizontal_flip=True, 
                                           rotation_range=20, 
                                           vertical_flip = True)
        self.test_gen.fit(self.ds.X_test)

        self.train_generator = self.train_gen.flow(np.array(self.ds.X_train), self.ds.y_train, batch_size=self.batch_size) 
        self.validation_generator = self.val_gen.flow(np.array(self.ds.X_val), self.ds.y_val, batch_size=self.batch_size)
        self.test_generator = self.test_gen.flow(np.array(self.ds.X_test), self.ds.y_test, batch_size=1)
        self.build()


    def build(self):
        # Model Building
        self.model = Sequential()
        for x in self.convolutions:
            self.model.add(Conv2D(x, activation=self.activation, kernel_size=3,
                         input_shape=(self.ds.img_height, self.ds.img_width, 3), padding='same', kernel_initializer='TruncatedNormal', bias_initializer='zeros'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        for y in self.mlp:
            self.model.add(Dense(y, activation=self.activation, kernel_initializer='TruncatedNormal', bias_initializer='zeros'))

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
        plot_val_loss(self.hist.history)
        plt.savefig(self.name + "-val_loss.png")
        plt.close()

        plt.figure()
        plot_val_acc(self.hist.history)
        plt.savefig(self.name + "-val_acc.png")
        plt.close()


    def test(self):
        y_pred = self.model.predict_generator(self.test_generator, steps = len(self.ds.X_test))
        #y_pred = self.model.predict(self.ds.X_test)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.ds.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes=self.ds.categories,
                              title='Confusion matrix, without normalization')
        plt.savefig(self.name + "-confusion.png")
        plt.close()

        #self.model.save_weights('curtis300.h5')


#############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Model Training", action="store_true")
#parser.add_argument("--test", "Model Testing")
args = parser.parse_args()

RANDOM_SEED=159
pd.set_option('display.max_colwidth', -1)

md = Dataset()
md.stats()

m = ConvModel(data = md, c = [64,32], m = [100,20], b = 8, name="conv64-20")
if args.train:
    m.train()
else:
    m.test()

m = ConvModel(data = md, c = [128,64,32], m = [100,20], b = 8, name="conv128-20")
if args.train:
    m.train()
else:
    m.test()

m = ConvModel(data = md, c = [256,128,64], m = [100,20], b = 8, name="conv256-20")
if args.train:
    m.train()
else:
    m.test()
