# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
 
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
 
import warnings
warnings.filterwarnings('ignore')

# ## 4. Data preprocessing

# In[9]:


# folders paths
train_path = 'Chest X ray images/train/'


# In[10]:


# use tensorflow for the data augmentation and preprocessing
# split data into 80% training and 20% validation
gen = ImageDataGenerator(
                  rescale=1./255.,
                  horizontal_flip = True,
                  validation_split=0.2 
                 )
 
train_generator = gen.flow_from_directory(
    directory = train_path, 
    subset="training",
    color_mode="rgb",
    target_size = (331,331), 
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
)
 
validation_generator = gen.flow_from_directory(
    directory = train_path, 
    subset="validation",
    color_mode="rgb",
    target_size = (331,331),
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=42,
)


# In[11]:


x,y = next(train_generator)
x.shape


# In[12]:


a = train_generator.class_indices

# store class names in a list
class_names = list(a.keys()) 

def plot_images(img, labels):
    plt.figure(figsize=[15, 10])
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(img[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')
 
plot_images(x,y)
