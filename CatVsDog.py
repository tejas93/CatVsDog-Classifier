# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:11:48 2020

@author: Tejas Chaturvedi
"""

#IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib as plt
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import random 
import os

# DEFINING IMAGE PROPERTIES------ 
Image_width = 128
Image_height = 128
Image_size = (Image_width, Image_height)
Image_channels = 3

# PREPARING DATA FOR TRAINING MODEL -----------
filename = os.listdir('C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\train') #listdir() returns a list containing the names of the entries in the directory given by path
categories = []
for f_name in filename:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filenames': filename, 'category': categories}) # A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns

#NEURAL NETWORK------------
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Activation, Flatten

model = Sequential()

model.add(Conv2D(32, (3,3), activation= 'relu', input_shape = (Image_width, Image_height, Image_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation=('relu')))

model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation=('softmax'))) 


model.compile(loss = ('categorical_crossentropy'), optimizer=('adam'),
              metrics = ['accuracy'])

#DEFINE CALLBACKS AND PRAMS AS LEARNING RATE------------------
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#EarlyStopping:Stop training when a monitored metric has stopped improving.
#ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
earlystop = EarlyStopping(patience= 10) 
#PATIENCE:Number of epochs with no improvement after which training will be stopped.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience= 2 , 
                                            verbose = 1, factor= 0.5, min_lr= 0.00001)
callbacks = [earlystop, learning_rate_reduction]

#DATA MANAGMENT, TRAIN SET
df['category']= df['category'].replace({0:'cats',1:'dogs'})
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 42)
train_df = train_df.reset_index(drop=True)
validate_df= validate_df.reset_index(drop =True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

#DATA AGUMENTATION 
train_datagen = ImageDataGenerator(rotation_range= 15, rescale = 1./255,
                                  height_shift_range = 0.1, width_shift_range =0.1 ,
                                  shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True)
train_gen = train_datagen.flow_from_dataframe(train_df,'C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\train',
                                              x_col = 'filenames', y_col = 'category', target_size = Image_size, 
                                              class_mode = 'categorical', batch_size = batch_size)
#flow_from_dataframe: Takes the dataframe and the path to a directory and generates batches of augmented/normalized data.
validate_datagen = ImageDataGenerator(rescale = 1./255)
validate_gen = validate_datagen.flow_from_dataframe(validate_df,'C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\train',
                                                    x_col = 'filenames', y_col = 'category', target_size = Image_size,
                                                    batch_size = batch_size)

#MODEL TRAINING
epochs = 10
history = model.fit_generator(train_gen, epochs=epochs, validation_data= validate_gen,
                              validation_steps= total_validate//batch_size,
                              steps_per_epoch=total_train//batch_size, callbacks=callbacks)
#fit_generator: Fits the model on data yielded batch-by-batch by a Python generator. (deprecated)

#TEST DATA PREP
test_filename = os.listdir('C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\test1')
test_df = pd.DataFrame({'filenames':test_filename})
nb_samples = test_df.shape[0]
test_datagen = ImageDataGenerator(rescale = 1./255)
test_gen = test_datagen.flow_from_dataframe(test_df,'C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\test1',
                                              x_col = 'filenames', y_col = None, target_size = Image_size, 
                                              class_mode = 'categorical', batch_size = batch_size)

#MAKE CATEGORICAL PREDICTION
predict = model.predict_generator(test_gen, steps = np.ceil(nb_samples/batch_size))
#TEST_GENERATOR KO DEFINE HI NAHI HI KIYA?, STEPS KYA H? 

#CONVERT LABELS TO CATEGORIES
test_df['category'] = np.argmax(predict, axis=-1)  #Returns the indices of the maximum values along an axis.
label = dict((v,k) for k,v in train_gen.class_indices.items())
test_df['category'] = test_df['category'].replace(label)
test_df['category'] = test_df['category'].replace({'dog':1,'cat':0})

#VISULAIZING THE RESULTS
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("C:\\Users\\Tejas Chaturvedi\\Downloads\\dogs-vs-cats\\test1"+54, target_size=Image_size)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()