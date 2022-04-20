import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

# Function to rescaling the training images from pixel range of (0,255) to a range (0,1). This function helps rescaling all the 
# images to the similar pixel range so that model can be trained in a better way.
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

batchSize= 32        # number of training sample to work at a time.
targetSize=(24,24)  

train_batch= generator('data/train',shuffle=True, batch_size=batchSize,target_size=targetSize)   
valid_batch= generator('data/valid',shuffle=True, batch_size=batchSize,target_size=targetSize)

SPE= len(train_batch.classes)//batchSize
VS = len(valid_batch.classes)//batchSize
print(SPE,VS)


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),   
    Conv2D(32,(3,3),activation='relu'), 
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# here we have defined the loss function, optimizer and the metrices.
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Here we are training our NN for 20 epochs
model.fit_generator(train_batch, validation_data=valid_batch,epochs=20,steps_per_epoch=SPE ,validation_steps=VS)

# we are saving the trained model in the models folder.
model.save('models/model4.h5', overwrite=True)