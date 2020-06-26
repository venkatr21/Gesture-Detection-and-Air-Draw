import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(6, activation='softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['acc'])

traindatapath = 'dataset/train'
testdatapath = 'dataset/test'
datatrain = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30)
datatest = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30)

traindata = datatrain.flow_from_directory(
    traindatapath,
    target_size=(300,300),
    class_mode = 'categorical',
    batch_size = 64)

testdata = datatest.flow_from_directory(
    testdatapath,
    target_size = (300,300),
    batch_size = 64,
    class_mode = 'categorical')

model.fit_generator(traindata,
                    steps_per_epoch = 140,
                    epochs = 20,
                    validation_data=testdata,
                    validation_steps = 60,
                    verbose = 1)
model.save('finger_model1.h5')
