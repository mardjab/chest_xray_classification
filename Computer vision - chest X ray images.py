#!/usr/bin/env python
# coding: utf-8

# # Classifying pulmonary diseases using chest X rays

# <b>Contact info</b>
# <br>
# <b>Name:</b> Mardja Bueno, Ph.D.
# <br>
# <b>email:</b> mardja.bueno@gmail.com
# <br>

# ## 1. Overview and goals

# ## 2. The dataset

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from PIL import Image
from pathlib import Path
import os


# In[2]:


# create path for the folders where the training, test and validation images are
train_folders = 'Chest X ray images/train/' 
test_folders = 'Chest X ray images/test/'
val_folders = 'Chest X ray images/val/'

# create empty lists for train, test and val
total_train = {}
total_test = {}
total_val = {}

# iterate through the length of the list
for folder in os.listdir(train_folders):
    if not folder.startswith('.'): # so it doesn't open the .DS Store folder on Mac
        total_train[folder] = len(os.listdir(train_folders + folder))
    
for folder in os.listdir(test_folders):
    if not folder.startswith('.'):
        total_test[folder] = len(os.listdir(test_folders + folder))
    
for folder in os.listdir(val_folders):
    if not folder.startswith('.'):
        total_val[folder] = len(os.listdir(val_folders + folder))

# sum the number of images in each list
quantity_train = pd.DataFrame(list(total_train.items()), index = range(0,len(total_train)), columns = ['class','count'])
quantity_test = pd.DataFrame(list(total_test.items()), index = range(0,len(total_test)), columns = ['class','count'])
quantity_val = pd.DataFrame(list(total_val.items()), index = range(0,len(total_val)), columns = ['class','count'])

# print how many images we have in each dataset
print("Number of images in the training dataset : ", sum(total_train.values()))
print("Number of images in the testing dataset : ",sum(total_test.values()))
print("Number of images in the validation dataset : ",sum(total_val.values()))


# The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Normal/Pneumonia/Covid-19/Tuberculosis). A total of 7135 x-ray images are present.

# ## 3. Exploratory data analysis

# In[3]:


# create bar plots with the number of training and testing
# images in each folder per pulmonary disease
figure, ax = plt.subplots(1,2, figsize = (15,8))

sns.set_palette("Set2")

sns.barplot(x = 'class',y = 'count',data = quantity_train,ax = ax[0])
sns.barplot(x = 'class',y = 'count',data = quantity_test,ax = ax[1])


# set titles and labels
ax[0].set_title('Number of training images per pulmonary disease')
ax[0].set(xlabel = 'Pulmonary disease', ylabel = 'Number of images')

ax[1].set_title('Number of testing images per pulmonary disease')
ax[1].set(xlabel = 'Pulmonary disease', ylabel = 'Number of images')

plt.show()


# ## 4. Data preprocessing

# In[4]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# rescale: rescales images by the given factor
# horizontal flip: randomly flip images horizontally.
# validation_split: percentage of images reserved for validation.

# In[5]:


# use tensorflow for the data augmentation and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# use ImageDataGenerator to rescale and horizontally flip the images
train_image_gen = ImageDataGenerator(rescale=1./255, 
                                     horizontal_flip = True, 
                                     validation_split=0.2)

test_image_gen  = ImageDataGenerator(rescale=1./255)

valid_image_gen = ImageDataGenerator(rescale=1./255)



# transform the images from RGB to grayscale - is going to speed up model training later
# image size = 331 -> because of the pre trained model we're going to use
# flow training images in batches of 20 using train_datagen generator
train_generator = train_image_gen.flow_from_directory(directory=train_folders, 
                                                    target_size=(224, 224),
                                                    color_mode="grayscale",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=42)


valid_generator = valid_image_gen.flow_from_directory(directory=val_folders, 
                                                    target_size=(224, 224),
                                                    color_mode="grayscale",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,seed=42)

test_generator = test_image_gen.flow_from_directory(directory=test_folders,
                                                  target_size=(224, 224),
                                                  color_mode="grayscale",
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False,
                                                  seed=42)


# ## 5. Building the model

# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# creating the model
model= Sequential()

model.add(Conv2D(filters=128,strides= 1, kernel_size = (5,5), activation='relu', input_shape=(224,224,1,)))
model.add(MaxPool2D(3,3))
model.add(Conv2D(filters=64,kernel_size = (5,5),activation='relu'))
model.add(MaxPool2D(3,3))

model.add(Conv2D(filters=30,kernel_size = (3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(.1))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(.1))
model.add(Dense(32,activation='relu'))
model.add(Dense(4,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics= ['accuracy'])
model.summary()


# In[7]:


# hyperparameters
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

# stop the training when there is no improvement after 3 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# fit the model
model.fit(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=valid_generator,verbose= 1,
          validation_steps=STEP_SIZE_VALID,
          epochs=15, callbacks=early_stop)


# In[8]:


# evaluate the model
model.evaluate(valid_generator,steps=STEP_SIZE_VALID)


# In[16]:


# save the model
model.save("ChestXrayModel.h5")


# In[ ]:





# In[ ]:





# ## 6. Using a pre-trained model

# In[30]:


from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

# first pretrained model using Xception
base_pt_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_pt_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_pt_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
pt_model = keras.Model(inputs, outputs)

pt_model.summary()


# In[32]:


# train the top layer
pt_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
pt_model.fit(train_generator, epochs=epochs, validation_data=valid_generator)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


# second pretrained model using InceptionResNetV2
# load the InceptionResNetV2 architecture with imagenet weights as base
base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(331,331,3)
                     )


# In[23]:


# freeze the layer so that the base_model's internal state will not change during training
base_model.trainable=False


# In[24]:


pt_model = tf.keras.Sequential([
        base_model,  
        tf.keras.layers.BatchNormalization(renorm=True), #speeds up training
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(120, activation='softmax')
    ])


# In[25]:


# compile the model
pt_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[26]:


# print the summary of the model
pt_model.summary()


# In[27]:


# early stopping call back
early = tf.keras.callbacks.EarlyStopping( patience=10,
                                          min_delta=0.001,
                                          restore_best_weights=True)


# In[28]:


# fit model
pt_model.fit(train_generator, 
             steps_per_epoch=STEP_SIZE_TRAIN, 
             validation_data=valid_generator, 
             validation_steps=STEP_SIZE_VALID, 
             epochs=15, 
             callbacks=[early])


# In[19]:


# evaluate the model
pt_model.evaluate(valid_generator,steps=STEP_SIZE_VALID)


# In[ ]:


# save the model
pt_model.save("ChestXrayModel_pretrained.h5")


# In[ ]:




