import tensorflow as tf 
from tensorflow.keras import layers, Model


# A basic Residual block with 2 convolutional layers and a skip connection (The core building block of ResNet)
class ResidualBlock(layers.layer):
  def __init__(self, filters, strides = 1):
    super(ResidualBlock, self).__init__()
    self.filters = filters
    self.strides = strides

    #first convolutional layer in the block 
    self.conv1 = layers.conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
    self.bn1 = layers.BatchNormalization()

    #second conolutional layer in the block 
    self.conv2 = layers.conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()

    # Skip connection layer (only needed if dimensions change)
    self.skip_conv = None
    if strides != 1:
        self.skip_conv = layers.Conv2D(filters, kernel_size=1, strides=strides, use_bias=False)
        self.skip_bn = layers.BatchNormalization()
  #-------------------------------------------------------------
  def call(self, inputs):
    # The main path : Conv -> BN -> ReLU -> Conv -> BN
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = tf.nn.relu(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    # Skip connection path 
    identity = inputs 
    if self.skip_conv is not None:
      identity = self.skip_conv(identity)
      identity = self.skip_bn(identity)

    # Add the skip connection to the main path 
    output = x + identity 
    output = tf.nn.relu(output)

    return output

#----------------------------------------------------------------------------------------------------------------------------



