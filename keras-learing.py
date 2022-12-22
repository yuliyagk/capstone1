#!/usr/bin/env python
# coding: utf-8

# 
# I have dowloaded the data from the [Kitchenware classification](https://www.kaggle.com/competitions/kitchenware-classification) competition on Kaggle.
# 
# To get started you need to download the kitchenware-classification.zip and extract it in the data subdirectory.

# Now let's train a baseline model

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras


# First, we will load the training dataframe and split it into train and validation

# In[2]:


df_train_full = pd.read_csv('data/train.csv', dtype={'Id': str})
df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'
df_train_full.head()


# In[3]:


val_cutoff = int(len(df_train_full) * 0.8)
df_train = df_train_full[:val_cutoff]
df_val = df_train_full[val_cutoff:]


# Now let's create image generators

# In[4]:


from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)


# In[6]:


base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(6)(vectors)

model = keras.Model(inputs, outputs)


# In[7]:


learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# In[10]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[11]:


history = model.fit(
    train_generator,
    epochs=4,
    validation_data=val_generator,
    callbacks=[checkpoint]
)


# Now let's use this model to predict the labels for test data

# In[16]:


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xticks(np.arange(4))
plt.legend()


# I am using the model from the last epoch because it has the highest accuracy.

# # Using the model
# * Loading the model
# * Evaluating the model
# * Getting predictions

# In[17]:


import tensorflow as tf
from tensorflow import keras


# In[18]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.xception import preprocess_input


# For quick testing we are using a cup

# In[20]:


path = 'data/images/4138.jpg'


# In[19]:


model = keras.models.load_model('xception_v1_04_0.868.h5')


# In[21]:


img = load_img(path, target_size=(150, 150))


# In[22]:


import numpy as np


# In[23]:


x = np.array(img)
X = np.array([x])
X.shape


# In[24]:


X = preprocess_input(X)


# In[25]:


pred = model.predict(X)


# In[26]:


classes = [
    'cups',
    'glasses',
    'plates',
    'spoons',
    'forks',
    'knifes'
]


# In[27]:


dict(zip(classes, pred[0]))


# We successfully detected the cup

# In[ ]:





