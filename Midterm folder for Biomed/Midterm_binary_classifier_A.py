#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries

import tensorflow as tf

from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


# Load the pre-trained VGG16 model and freeze all the layers except for the last few

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in vgg.layers[:-4]:
    
    layer.trainable = False


# In[3]:


# Define a new output layer with one neuron and a sigmoid activation function

x = Flatten()(vgg.output)

output_layer = Dense(1, activation='sigmoid')(x)


# In[4]:


# Create the new model

model = Model(inputs=vgg.input, outputs=output_layer)


# In[13]:


# Compile the model with an appropriate loss function and optimizer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


# Load and preprocess your dataset of images, resizing them to 128x128 pixels

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_set = train_datagen.flow_from_directory(r'C:\Users\kjbua\Dropbox\PC\Downloads\Midterm folder for Biomed\Data1\train', target_size=(128, 128), batch_size=32, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)

val_set = val_datagen.flow_from_directory(r'C:\Users\kjbua\Dropbox\PC\Downloads\Midterm folder for Biomed\Data1\test', target_size=(128, 128), batch_size=32, class_mode='binary')


# In[17]:


# Train the model on your dataset, using the extracted features and the new output layer
Epoch 1/10

history = model.fit(train_set, epochs=10, validation_data=val_set)


# In[18]:


# Evaluate the performance of the model on a validation set

loss, accuracy = model.evaluate(val_set)

print(f"Validation loss: {loss:.4f}")

print(f"Validation accuracy: {accuracy:.4f}")


# In[ ]:




