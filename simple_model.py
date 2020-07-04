# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:10:49 2020

@author: Chandravaran K V
"""

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.metrics import confusion_matrix,classification_report
import cv2
import os, pathlib, argparse


mapping = {'defect': 0,'non_defect': 1}
class_weight = {0: 1., 1: 1.}
EPOCHS=25
file = open('C:\\Users\\Chandravaran K V\\Documents\\internship\\test_full.txt', 'r')
testfiles = file.readlines()
print(len(testfiles))



# Initialising the CNN
model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense( activation = 'relu',units= 2585))

model.add(Dense(activation = 'sigmoid',units = 1))

model.summary()
lr=0.001
opt = Adam(learning_rate=lr, amsgrad=True)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])




#inputing the images 


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=(0.9, 1.1),
                fill_mode='constant',
                cval=0.,)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:\\Users\\Chandravaran K V\\Documents\\internship\\cropped_images\\train',
                                                 target_size = (50, 50),
                                                 batch_size = 4,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:\\Users\\Chandravaran K V\\Documents\\internship\\cropped_images\\test',
                                            target_size = (50, 50),
                                            batch_size = 4,
                                            class_mode = 'binary')
print(test_set.labels)
H = model.fit_generator(training_set,steps_per_epoch=10,epochs=EPOCHS,validation_data=test_set,validation_steps=10,class_weight=class_weight)




#parameters 

print(H.history)
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

from PIL import Image
y_test = []
pred = []
line = testfiles[0].split()

for i in range(len(testfiles)):
    line = testfiles[i].split()
    x = cv2.imread(os.path.join('C:\\Users\\Chandravaran K V\\Documents\\internship\\validation', line[0]))
    x = x.astype('float32') / 255.0
    #print(mapping[line[1]])
    y_test.append(mapping[line[1]])
    #print(model.predict_classes(np.expand_dims(x, axis=0)))
    #print(np.array(model.predict_classes(np.expand_dims(x, axis=0))))
    #print(np.array(model.predict_classes(np.expand_dims(x, axis=0))).argmax(axis=1))
    pred.append(np.array(model.predict_classes(np.expand_dims(x, axis=0))).argmax(axis=1))
y_test = np.array(y_test)
pred = np.array(pred)

matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
print(matrix)
class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens defect: {0:.3f}, non_defect: {1:.3f}'.format(class_acc[0],class_acc[1]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV defect: {0:.3f}, non_defect {1:.3f}'.format(ppvs[0],ppvs[1]))

print(classification_report(y_test,pred,target_names=['defect','normal']))