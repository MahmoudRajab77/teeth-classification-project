import tensorflow as tf
from tensorflow.keras import layers, Model




# A basic Residual block with 2 convolutional layers and a skip connection (The core building block of ResNet)
class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.dropout = layers.Dropout(dropout_rate)
        
        self.skip_conv = None
        if strides != 1:
            self.skip_conv = layers.Conv2D(filters, kernel_size=1, strides=strides, use_bias=False)
            self.skip_bn = layers.BatchNormalization()
    #----------------------------------------------------------------
    def call(self, inputs, training=False):  
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)  
        x = self.dropout(x, training=training)  
        
        # Skip connection
        identity = inputs
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity, training=training)
            
        output = x + identity
        output = tf.nn.relu(output)
        
        return output
#----------------------------------------------------------------------------------------------------------------

# A custom ResNet model built from scratch for 7-class dental classification.
class ResNet(Model):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(ResNet, self).__init__()
        
        # Initial processing
        self.initial_conv = layers.Conv2D(32, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.initial_bn = layers.BatchNormalization()
        self.initial_pool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # ADD THIS: Second conv for better feature extraction
        self.initial_conv2 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.initial_bn2 = layers.BatchNormalization()
        
        # Attention mechanism
        self.attention = layers.Conv2D(64, 1, activation='sigmoid')
        
        # Enhanced residual layers
        self.layer1 = self._make_layer(filters=64, num_blocks=2, strides=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(filters=128, num_blocks=2, strides=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(filters=256, num_blocks=2, strides=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(filters=512, num_blocks=2, strides=2, dropout_rate=dropout_rate)
        
        # Additional regularization
        self.global_pool = layers.GlobalAveragePooling2D()
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(num_classes, activation='softmax')

    #----------------------------------------------------------------

    # A functtion to create a layer of residual blocks
    def _make_layer(self, filters, num_blocks, strides, dropout_rate):
        layers_list = []
        # ✅ FIXED: Use 'ResidualBlock' (the actual class name), not 'ImprovedResidualBlock'
        layers_list.append(ResidualBlock(filters, strides=strides, dropout_rate=dropout_rate))
        for _ in range(1, num_blocks):
            layers_list.append(ResidualBlock(filters, strides=1, dropout_rate=dropout_rate))
        
        # Create Sequential and override its call method to pass 'training'
        seq = tf.keras.Sequential(layers_list)
        
        # Create a wrapper that passes 'training' to all layers
        class TrainingSequential(tf.keras.layers.Layer):
            def __init__(self, sequential_layer):
                super(TrainingSequential, self).__init__()
                self.sequential_layer = sequential_layer
            
            def call(self, inputs, training=False):
                x = inputs
                for layer in self.sequential_layer.layers:
                    x = layer(x, training=training)
                return x
        
        return TrainingSequential(seq)
    #-----------------------------------------------------------------
    
    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.initial_pool(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Residual layers
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        # Final classification
        x = self.global_pool(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        
        return x
#--------------------------------------------------------------------------------------------
# A function to build and return the ResNet Model
def build_resnet(input_shape=(224, 224, 3), num_classes=7, dropout_rate=0.4):
    """Build and return the enhanced ResNet model."""
    # Create input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Build enhanced model
    model = ResNet(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Connect the model by calling it on the input
    outputs = model(inputs)
    
    # Create a proper Keras Model
    full_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="enhanced_resnet_model")
    
    return full_model

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = build_resnet()
    print(f"Parameters: {model.count_params():,}")
    
    # Test with training=False and training=True
    test_input = tf.random.normal([1, 224, 224, 3])
    
    print("\nTesting inference mode (training=False):")
    output_inference = model(test_input, training=False)
    print(f"Inference output shape: {output_inference.shape}")
    
    print("\nTesting training mode (training=True):")
    output_training = model(test_input, training=True)
    print(f"Training output shape: {output_training.shape}")
    
    print("\n✅ Model built and tested successfully!")






