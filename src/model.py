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
# A custom ResNet model built from scratch for 7-class dental classification.
class ResNet(Model):
  def __init__(self, num_classes=7):
    super(ResNet, self).__init__()
        
    # Initial layers (not residual blocks)
    self.initial_conv = layers.Conv2D(64, kernel_size=7, strides=2,
                                          padding='same', use_bias=False)
    self.initial_bn = layers.BatchNormalization()
    self.initial_pool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        
    # Create 4 groups of residual blocks
    # Pattern for a small ResNet: [1, 1, 1, 1] blocks per group
    self.layer1 = self._make_layer(filters=64, num_blocks=1, strides=1)
    self.layer2 = self._make_layer(filters=128, num_blocks=1, strides=2)
    self.layer3 = self._make_layer(filters=256, num_blocks=1, strides=2)
    self.layer4 = self._make_layer(filters=512, num_blocks=1, strides=2)
        
    # Final classification layers
    self.global_pool = layers.GlobalAveragePooling2D()
    self.fc = layers.Dense(num_classes, activation='softmax')

  #------------------------------------------------------------
  
  # A functtion to create a layer of residual blocks
  def make_layer(self, filters, Num_blocks, strides):
    layers_list = []
    # First block in the layer 
    layers_list.append(ResidualBlock(filters, strides=strides))
        
    # Add the remaining blocks (all with stride=1)
    for _ in range(1, num_blocks):
        layers_list.append(ResidualBlock(filters, strides=1))
            
    return tf.keras.Sequential(layers_list)
    
  #-----------------------------------------------------
   def call(self, inputs):
     # Initial layers
     x = self.initial_conv(inputs)
     x = self.initial_bn(x)
     x = tf.nn.relu(x)
     x = self.initial_pool(x)
        
     # Go through all residual layers
     x = self.layer1(x)
     x = self.layer2(x)
     x = self.layer3(x)
     x = self.layer4(x)
        
     # Final classification
     x = self.global_pool(x)
     x = self.fc(x)
        
     return x

  #--------------------------------------------------------
  # A function to build and return the ResNet Model 
  def build_resnet(input_shape=(244,244,3), num_classes = 7):
    model = ResNet(num_classes = num_classes)

    model.build((None, *input_shape))

    return model


