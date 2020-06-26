import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
ytraindata1 = []
ytraindata2 = []
xtraindata = []
dataset = "dataset/test/"
for root,dirs,files in os.walk(dataset):
    for imgname in files:
        img = cv2.imread(dataset+imgname, cv2.IMREAD_GRAYSCALE)
        name = imgname.split('.')
        img = np.asarray(img).reshape(300,300)
        y1,y2 = list(map(int,name[0].strip().split(',')))
        xtraindata.append(img.astype('float32'))
        ytraindata1.append(y1)
        ytraindata2.append(y2)
    xtraindata = np.asarray(xtraindata)
    xtraindata = np.expand_dims(xtraindata, -1)
    ytraindata1 = np.asarray(ytraindata1)
    ytraindata2 = np.asarray(ytraindata2)
print("Input data shape:",xtraindata.shape)
print("Output data shape:",ytraindata1.shape)
ytraindata1 = ytraindata1/300
ytraindata2 = ytraindata2/300
data_generator = ImageDataGenerator(rescale = 1./255)
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(xtraindata, ytraindata1,test_size=0.2,shuffle=True)
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(xtraindata, ytraindata2,test_size=0.2,shuffle=True)
trainingdata1 = data_generator.flow(xtrain1, ytrain1, batch_size = 128)
trainingdata2 = data_generator.flow(xtrain2, ytrain2, batch_size = 128)

model = Sequential()
model.add(Conv2D(16,(3,3), input_shape = (300,300,1), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit_generator(trainingdata1,
                    steps_per_epoch = int(len(xtrain1)/128),
                    epochs = 25,
                    validation_data = (xtest1, ytest1),
                    verbose = 1)
model.save('index_point_x4.h5')
model.fit_generator(trainingdata2,
                    steps_per_epoch = int(len(xtrain2)/128),
                    epochs = 25,
                    validation_data = (xtest2, ytest2),
                    verbose = 1)
model.save('index_point_y4.h5')

