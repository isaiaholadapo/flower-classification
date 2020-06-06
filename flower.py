# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:52:40 2020

@author: Isaiah
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising
classifier = Sequential()

# convolution
classifier.add(Convolution2D(32, 3, 3, input_shape =( 64, 64, 3), activation = 'relu'))

# pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a second layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifier.add(Flatten())

#fully connected 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'sigmoid'))

#compling 
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#image preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Creating the Test set
test_set = test_datagen.flow_from_directory('data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                          steps_per_epoch = 3130,
                          epochs = 25,
                          validation_data = test_set,
                          validation_steps = 540)