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
