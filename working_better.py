import os
import sys
import glob
import time
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imresize 
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop

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

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y)

train_gen = ImageDataGenerator(rescale=1. / 255, 
                               horizontal_flip=True, 
                               rotation_range=20,
                               vertical_flip = True
                              )

test_gen = ImageDataGenerator(rescale=1. / 255,
                            horizontal_flip=True, 
                               rotation_range=20,
                               vertical_flip = True)

train_generator = train_gen.flow(np.array(X_train), 
                                 y_train, 
                                 batch_size=8) 
validation_generator = test_gen.flow(np.array(X_test), y_test)
test_generator = test_gen.flow(np.array(X_test), y_test)

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

model.load_weights('good_try_v2.h5')

#opt = RMSprop(lr=0.01)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())



historyAll = model.fit_generator(
    train_generator, 
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 8,
    validation_steps=len(X_test) // 8,
    epochs=100,
    class_weight = [1.5, 2.0, 0.5, 2.0]
)

#model.fit(X_train, y_train, validation_split=0.2, epochs=10)

print(model.evaluate(X_test, y_test))

model.save_weights('first_try.h5')


#Plot Testing and Validation MSE and Loss
plot_hist(historyAll.history, xsize=8, ysize=12)

