# ## 5. Building the model with transfer learning

# performance of this from-scratch model was drastically limited.
# 
# This model had to first learn how to detect generic features in the images, such as edges and blobs of color, before detecting more complex features.
# 
# In real-world applications, this can take days of training and millions of images to achieve high performance. It would be easier for us to download a generic pretrained model and retrain it on our own dataset. This is what Transfer Learning entails.
# 
# In this way, Transfer Learning is an approach where we use one model trained on a machine learning task and reuse it as a starting point for a different job. Multiple deep learning domains use this approach, including Image Classification, Natural Language Processing, and even Gaming! The ability to adapt a trained model to another task is incredibly valuable.

# In[13]:


# load the InceptionResNetV2 architecture
base_model = tf.keras.applications.InceptionResNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=(331,331,3)
                     )

# freeze the layer = its internal state will not change during training and weights will not be updated when fitting
base_model.trainable=False
 
model = tf.keras.Sequential([
        base_model,  
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax') # for the 4 classes present
    ])


# In[14]:


# categorical cross entropy = used as a loss function for multi-class classification problems
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[15]:


# get summary of the model
model.summary()


# In[16]:


# set early stopping call back
early = tf.keras.callbacks.EarlyStopping( patience=10,
                                          min_delta=0.001,
                                          restore_best_weights=True)


# In[17]:


batch_size=32

# hyperparameters
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
 
# fit model
history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=15,
                    callbacks=[early])
