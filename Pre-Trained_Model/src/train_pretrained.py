import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from datetime import datetime
from config import Config
from data_loader import get_data_loaders
from pretrained_model import create_pretrained_model




def train_pretrained():
  print("Starting training the model......")
  print("Loading Data.....")
  train_ds, valid_ds, test_ds, class_names, class_weight_dict = get_data_loaders()

  print("Data loaded Successfully!")

  # printing the classes names 
  for i, name in enumerate(class_names):
    print (f"    -{name}")

  # Call the function to build the model 
  model, base_model = create_pretrained_model(
    input_shape = (Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), 
    num_classes = Config.NUM_CLASSES
  )

  # printing the summary of the model 
  print("Model Summary:")
  model.summary()

  # preparing for the training 
  print ("Getting everythin ready for training process....")
  model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), 
    loss = 'categorical_crossentropy',  # the loss function 
    metrics = ['accuracy']            # to measure the accuracy 
  )

  # create folder for saved models 
  os.makedirs('saved_models', exist_ok=True)

  # some callbacks
  callbacks = [
    # saving the model with the best validation accuracy 
    tf.keras.callbacks.ModelCheckpoint(
      'saved_models/Pretrained_BestModel.h5',   # the path and name of saved model 
      monitor = 'val_accuracy', 
      save_best_only = True, 
      mode = 'max',  
      verbose = 1
    ), 

    # early stopping if no progress based on validation accuracy 
    tf.keras.callbacks.EarlyStopping(
      monitor = 'val_accuracy', 
      patience = 10,       # wait for 10 epochs before stopping 
      restore_best_weights = True,   # restoring the weights that give the best accuracy 
      verbose = 1
    ), 

    # reducing the learning rate if no progress 
    tf.keras.callbacks.ReduceLROnPlateau(
      monitor = 'val_loss', 
      factor = 0.5, 
      patience = 5, 
      min_lr = 1e-6, 
      verbose=1
    )  
  ]

  # Start the training 
  print("Training Started!")
  print(f"    Num of epochs : 30")
  print(f"    Batch Size : {Config.BATCH_SIZE}")
  
  history = model.fit(
    train_ds, 
    validation_data = valid_ds, 
    epochs = 30, 
    callbacks = callbacks, 
    verbose = 1
  )

  # Evaluating the model over test data 
  print("\nEvaluating the model over test data.....")
  test_loss, test_accuracy = model.evaluate(test_ds, verbose = 0)
  print(f"\n    Test Loss: {test_loss:.4f}")
  print(f"    Test Accuracy: {test_accuracy:.4f}")


  # plotting the training Curves
  print("Plotting the training Curves.....")

  flg, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

  # the accuracy 
  ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
  ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
  ax1.set_title('Model Accuracy', fontsize=14)
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')
  ax1.legend()
  ax1.grid(True)

  # the loss 
  ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
  ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
  ax2.set_title('Model Loss', fontsize=14)
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Loss')
  ax2.legend()
  ax2.grid(True)

  plt.tight_layout()
  plt.savefig('pretrained_training_history.png', dpi=150)
  plt.show()

  print("Training history diagram saved! 'pretrained_training_history.png'")

  # saving the model 
  print("\n Saving the final model.....")
  model.save('saved_models/pretrained_final.h5')
  print("Final Model Saved Successfully!")

  # Summary Results 
  print("\n" + "="*50)
  print("Training Completed Successfully!")
  print(f"    Best validation Accuracy: {max(history.history['val_accuracy']):.4f}")
  print(f"    Best test Accuarcy: {test_accuracy:.4f}")
  print("="*50)

  # return point of the function 
  return model, history, test_accuracy



#----------------------------------------------------------------


if __name__ == '__main__':
   os.makedirs('saved_models', exist_ok=True)
   model, history, test_accuracy = train_pretrained()






  
