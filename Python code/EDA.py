# ## 2. The dataset

# In[1]:


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


# In[4]:


# load images with the Keras API
from tensorflow.keras.utils import load_img, img_to_array

def sample_image(sample_image_path):
    img = load_img(sample_image_path)
    print((img_to_array(img)).shape)
    plt.imshow(img)
    plt.show()


# In[5]:


sample_image_path = train_folders+"/COVID19/COVID19(22).jpg"
sample_image(sample_image_path)


# In[6]:


sample_image_path = train_folders+"/NORMAL/IM-0162-0001.jpeg"
sample_image(sample_image_path)


# In[7]:


sample_image_path = train_folders+"/PNEUMONIA/person3_bacteria_10.jpeg"
sample_image(sample_image_path)


# In[8]:


sample_image_path = train_folders+"/TURBERCULOSIS/Tuberculosis-12.png"
sample_image(sample_image_path)
