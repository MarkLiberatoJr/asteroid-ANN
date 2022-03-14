#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# In[2]:


asteroids_df = pd.read_csv('nasa.csv')


# In[3]:


asteroids_df.head(5)


# In[4]:


asteroids_df.describe()


# In[5]:


asteroids_df.groupby(['Hazardous','Absolute Magnitude','Minimum Orbit Intersection']).count()


# In[6]:


labels=asteroids_df['Hazardous']
features=asteroids_df.drop(columns=['Hazardous'])


# In[7]:


labels[0:5]


# In[8]:


features[0:5]


# In[9]:


labels.replace('F', 0, inplace=True)
labels.replace('T', 1, inplace=True)
labels[0:5]


# In[10]:


features=pd.get_dummies(features)
features[0:5]


# In[11]:


features=features.values.astype('float64')
labels=labels.values.astype('float64')


# In[12]:


features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2)
features_train, features_validation, labels_train, labels_validation = train_test_split(features_train,labels_train,test_size=0.2)


# In[13]:


model = keras.Sequential([keras.layers.Dense(32, input_shape=(1271,)),
                          keras.layers.Dense(20, activation=tf.nn.relu),
                          keras.layers.Dense(2,activation='softmax')])


# In[14]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


# In[18]:


history = model.fit(features_train, labels_train, epochs=30, validation_data=(features_validation,labels_validation))


# In[ ]:





# In[ ]:




