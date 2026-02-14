import tensorflow as tf 
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt 
import os 
from config import Config 
from data_loader import get_data_loaders



''' Create a function to create a model using MobileNetV2 
    The function takes : 1- image size 
                         2- Num of Classes 
    and return the model
'''                                                              
def create_pretrained_model(input_shape=(244,244,3), num_classes=7):
  print ("Loading Pre-trained MobileNetV2 model......")

  # Load the MobileNetV2
  base_model = tf.keras.applications.MobileNetV2(
    input_shape = input_shape, 
    include_top = False,       # Drop the head classifier from the model 
    weights = 'imagenet',      # Use the pretrained weights of imagenet
    pooling = 'avg'
  )

  base_model.trainable = False        # freeze the model don't train it or change the weights 

  model = models.Sequential ([
    layers.Input(shape = input_shape),              # first layer (input layer)
    layers.Lambda(lambda x: (x - 0.5) * 2),         # A layer Convert our image values from [0, 1] to [-1, 1] using lambda to fit the MobileNetV2 input
    base_model,                                      # the third layer 
    layers.BatchNormilzation(),                     # Fourth layer
    layers.Dropout(0.3),                            # Drop 30% of cells randomly during the training to prevent overfitting 
    layers.Dense(256, activation = 'relu'),         # a layer of 256 neurons 
    layers.Dropout(0.3),                            # Dropout layer again 
    layers.Dense(128, activation = 'relu'),         # a layer of 128 neurons 
    layers.Dropout(0.2),                            # Dropout layer with 20%
    layers.Dense(num_classes, activation = 'softmax') # last layer with 7 classes output
  ])

  return model, base_model
