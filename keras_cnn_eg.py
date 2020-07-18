import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


from keras import backend as K

import h5py
from keras import utils
import matplotlib.pyplot as plt

# To get the images and labels from file
with h5py.File('Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.15)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

img_rows = 69
img_cols = 69
	
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 3, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 3)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)





model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, kernel_size=(3, 3),
                 activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, 
          batch_size=128, nb_epoch=50, verbose=1, validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


