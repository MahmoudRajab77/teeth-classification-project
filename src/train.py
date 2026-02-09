import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
import os
from datetime import datetime
# Import the project modules
from config import Config
from data_loader import get_data_loaders
from model import build_resnet
from utils import plot_training_history  


# Configuration 
config = Config()

# Training Hyperparameters 
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
MODEL_SAVE_PATH = 'saved_models/resnet_model.h5'
LOG_DIR = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')


#------------------------------------------------------------

# The Training Function 
def train_model():
  print ("Start Training.....")

  # load data 
  print ("Loading Dataset......")
  train_ds, val_ds, test_ds, class_names, class_weight_dict = get_data_loaders()

  # DEBUG: Check data shapes
  print(f"\nDEBUG: Number of classes: {len(class_names)}")
  print(f"DEBUG: config.NUM_CLASSES: {config.NUM_CLASSES}")

  for images, labels in train_ds.take(1):
        print(f"DEBUG: Batch image shape: {images.shape}")
        print(f"DEBUG: Batch label shape: {labels.shape}")
        print(f"DEBUG: Label values sample: {labels[0].numpy()}")
        break

  # build model 
  model = build_resnet(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), num_classes=config.NUM_CLASSES)
  model.summary()

  # DEBUG: Test model output shape
  test_input = tf.random.normal([1, config.IMG_HEIGHT, config.IMG_WIDTH, 3])
  test_output = model(test_input, training=False)
  print(f"DEBUG: Model output shape: {test_output.shape}")
  print(f"DEBUG: Model output sample: {test_output.numpy()}")

  # Compile the Model 
  print ("Compiling Model.....")

  optimizer = Adam (learning_rate = LEARNING_RATE, clipnorm=1.0)
  loss_fn = CategoricalCrossentropy()
  metrics = [CategoricalAccuracy(name='accuracy')]

  model.compile(optimizer = optimizer, loss = loss_fn, metrics = metrics)


  # Setup callbacks
  callbacks = [
    # Save best model (based on validation accuracy)
    tf.keras.callbacks.ModelCheckpoint(
        'saved_models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Early stopping (be patient with medical data)
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor loss, more stable than accuracy
        patience=25,  # Increased for dental images (can be noisy)
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # Small improvement threshold
    ),
    
    # TensorBoard for visualization
    tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        profile_batch=0  # Disable profiling to avoid overhead
    ),
    
    # REDUCE LR ON PLATEAU (RECOMMENDED - Use this ONLY)
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Watch validation loss
        factor=0.5,  # Reduce LR by half
        patience=10,  # Wait 10 epochs with no improvement
        min_lr=1e-6,  # Minimum learning rate
        verbose=1,
        mode='min'
    )
  ]
  # Train the model
  print ("Training model....")
  history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=NUM_EPOCHS, 
    callbacks=callbacks, 
    class_weight=class_weight_dict,  # ADD THIS LINE
    verbose=1
  )

  # Evaluate on test set
  print("Evaluating on test dataset.....")
  test_results = model.evaluate(test_ds, verbose=0)
  print(f"\nTest Results:")
  print(f"   Loss: {test_results[0]:.4f}")
  print(f"   Accuracy: {test_results[1]:.4f}")
    
  # Plot training history
  print("Plotting training history.....")
  plot_training_history(history)
    
  # Save final model
  model.save('saved_models/final_model.h5')
  print(f"💾 Model saved to 'saved_models/final_model.h5'")

  # enf of train_model funcrion 
  return model, history, test_results



# Main function (Execution)
if __name__ == '__main__':
  # Create necessary directories
  os.makedirs('saved_models', exist_ok=True)
  os.makedirs('logs', exist_ok=True)
    
  # Train the model
  model, history, test_results = train_model()
    
  print("\nTraining completed successfully")
  print(f"   Final Test Accuracy: {test_results[1]:.4f}")
  print(f"   Model saved in: saved_models/")
  print(f"   Logs saved in: {LOG_DIR}")














