import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
import os
from datetime import datetime
# Import the project modules
from src.config import Config
from src.data_loader import get_data_loaders
from src.model import build_resnet
from src.utils import plot_training_history  


# Configuration 
config = Config()

# Training Hyperparameters 
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MODEL_SAVE_PATH = 'saved_models/resnet_model.h5'
LOG_DIR = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')


# The Training Function 
def train_model():
  print ("Start Training.....")

  # load data 
  print ("Loading Dataset......")
  train_ds, val_ds, test_ds, class_names = get_data_loaders()


  # build model 
  model = build_resnet (input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), num_classes = config.NUM_CLASSES)

  model.summary()

  # Compile the Model 
  print ("Compiling Model.....")

  optimmizer = Adam (learning_rate = LEARNING_RATE)
  loss_fn = CategoricalCrossentropy()
  metrics = [CategoricalAccuracy(name='accuracy')]

  model.compile(optimizer = optimizer, loss = loss_fn, metrics = metrics)


  # Setup callbacks
  callbacks = [
      # Save best model
      tf.keras.callbacks.ModelCheckpoint(
          MODEL_SAVE_PATH,
          monitor='val_accuracy',
          save_best_only=True,
          mode='max',
          verbose=1
      ),
      # Early stopping
      tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=10,
          restore_best_weights=True,
          verbose=1
      ),
      # TensorBoard logging
      tf.keras.callbacks.TensorBoard(
          log_dir=LOG_DIR,
          histogram_freq=1
      ),
      # Reduce learning rate on plateau
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss',
          factor=0.5,
          patience=5,
          min_lr=1e-6,
          verbose=1
      )
  ]

  # Train the model
  print ("Training model....")
  history = model.fit (
    train_ds, 
    validation_data = val_ds, 
    epochs = NUM_EPOCHS, 
    callbacks = callbacks, 
    verbose = 1
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
  return model, history, tets_results



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




