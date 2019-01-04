
from keras.applications.vgg16 import VGG16
from keras import models
from keras.layers import Dense, Dropout

vgg = VGG16(weights = 'imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

vgg.summary()

model = models.Sequential()
model.add(vgg)
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

model.summary()

