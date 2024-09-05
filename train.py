# Training the CNN Deep Learning Model

# Imports
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop

# Reproducibility
np.random.seed(2222)  

# Load scaled data, both pixels and labels
X_train = np.load('data/Scaled.bin.npy')
Y_tr_labels = np.load('data/labels.bin.npy')

# Reshape pixel data to 48 X 48 images
shapex, shapey = 48, 48
X_train = X_train.reshape(X_train.shape[0], shapex, shapey, 1)

# Convert labels to One-Hot-Encoding
Y_tr_labels = np_utils.to_categorical(Y_tr_labels)

# Define the model with 32 filters in the first convolutional layer followed by a max pooling layer and dense layer with dropout (50%)
model = Sequential()
model.add(Convolution2D(name="convolution2d_1", filters=32, kernel_size=(3, 3), padding="valid", input_shape=(48, 48, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(name="maxpooling2d_1", pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(name="dense_1", units=128, kernel_initializer="lecun_uniform"))
model.add(Dropout(trainable=True, name="dropout_1", rate=0.4))
model.add(Activation('relu'))
model.add(Dense(name="dense_2", units=7))
model.add(Activation('softmax'))

# Training the model with SGD and Nesterov momentum
sgd = SGD(lr=0.055, decay=1e-6, momentum=0.9, nesterov=True)
# optm = RMSprop(lr=0.004, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Fit the model
model.fit(X_train, Y_tr_labels, batch_size=128, epochs=15)

# Save the model and weights
import h5py
json_string = model.to_json()
model.save_weights('models/Face_model_weights.h5')
open('models/Face_model_architecture.json', 'w').write(json_string)
model.save_weights('models/Face_model_weights.h5')
