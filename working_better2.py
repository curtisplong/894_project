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

def plot_hist(h, xsize=3, ysize=3):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

   #  summarize history for accuracy
    plt.subplot(211)
    plt.plot(h['acc'])
    plt.plot(h['val_acc'])
    plt.title('Training vs Validation MSE using Scaled Data')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss using Scaled Data')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return


RANDOM_SEED=159
pd.set_option('display.max_colwidth', -1)

imagedir = "dataset-master/JPEGImages/"
labelsfile = "dataset-master/labels.csv"
img_width = 160
img_height = 120

labels = pd.read_csv(labelsfile)
labels = labels.loc[:, ['Image', 'Category']]
labels['Image'] = pd.to_numeric(labels['Image'])
labels = labels.dropna()
labels = labels[~labels.Category.str.contains(",")]
labels = labels[~labels.Category.str.contains("BASOPHIL")]

labels = pd.get_dummies(labels, columns=['Category'])
# Category_BASOPHIL  Category_EOSINOPHIL  Category_LYMPHOCYTE  Category_MONOCYTE  Category_NEUTROPHIL
#basophil = labels['Category_BASOPHIL'].sum()
eosinophil = labels['Category_EOSINOPHIL'].sum()
lymphocyte = labels['Category_LYMPHOCYTE'].sum()
neutrophil = labels['Category_NEUTROPHIL'].sum()
monocyte = labels['Category_MONOCYTE'].sum()
print(labels.describe())
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

labels['filename'] = labels.Image.apply(lambda x: imagedir + "BloodImage_" + str(x).zfill(5) + ".jpg")
# Filter for only image files that actually exist
labels = labels[labels['filename'].apply(lambda x: os.path.isfile(x))]

# Load and resize images
images = []
for x in labels['filename']:
    img = imageio.imread(x)
    img = imresize(img, (img_height,img_width))
    images.append(img)

images = np.array(images)
print(images.shape)

#plt.imshow(images[0])
#plt.show()

# Reduce to only categories
y = labels.loc[:,[
    #'Category_BASOPHIL',
    'Category_EOSINOPHIL',
                  'Category_LYMPHOCYTE', 'Category_MONOCYTE',
                  'Category_NEUTROPHIL']].values

#X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y)
#print(str(images.shape) + ', ' + str(y.shape))
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y)
#print(str(X_split.shape) + ', ' + str(y_split.shape))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)


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

train_generator = train_gen.flow(np.array(X_train), 
                                 y_train, 
                                 batch_size=8) 
validation_generator = val_gen.flow(np.array(X_val), y_val)
test_generator = test_gen.flow(np.array(X_test), y_test, batch_size=1)


# Model Building
model = Sequential()
model.add(Conv2D(32, activation='relu', kernel_size=3,
                 input_shape=(img_height, img_width, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, activation='relu', kernel_size=3, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(4, activation='softmax'))


#opt = RMSprop(lr=0.01)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Checkpoint to save optimal weights
checkpoint = ModelCheckpoint(filepath = "weights/third_try.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

historyAll = model.fit_generator(
    train_generator, 
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 8,
    validation_steps=len(X_test) // 8,
    epochs=500,
    class_weight = [1.5, 2.0, 0.5, 2.0],
    callbacks = [checkpoint]
)

# load the optimal weights before testing
model.load_weights("weights/third_try.h5")

#print(model.evaluate(X_test, y_test))


#Plot Testing and Validation MSE and Loss
plot_hist(historyAll.history, xsize=8, ysize=12)

# Plot Confusion Matrix
categories = ['Category_EOSINOPHIL', 'Category_LYMPHOCYTE', 'Category_MONOCYTE', 'Category_NEUTROPHIL']
y_pred = model.predict_generator(test_generator, steps = len(X_test))
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
utils.plot_confusion_matrix(cnf_matrix, classes=categories, title="Confusion Matrix (not normalized)" )
plt.savefig("results/working_better2-confusion.png")
plt.close()


