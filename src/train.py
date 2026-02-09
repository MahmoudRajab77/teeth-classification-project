import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np
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
LEARNING_RATE = 0.0005  # Reduced for better convergence
WARMUP_EPOCHS = 5
NUM_EPOCHS = 150
MODEL_SAVE_PATH = 'saved_models/resnet_model.h5'
LOG_DIR = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def call(self, y_true, y_pred):
        # Apply label smoothing
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / y_true.shape[-1]
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred + 1e-7)
        
        # Focal loss weighting
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce
        
        return tf.reduce_sum(focal_loss, axis=-1)

#-------------------------------------------------------------------------------------------------
# Warmup Callback Class
class WarmUpCallback(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, initial_lr=0.00003, target_lr=0.0003):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            # Set learning rate - FIXED: using learning_rate not lr
            self.model.optimizer.learning_rate.assign(lr)
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}: Warmup LR = {lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Log when warmup completes
        if epoch == self.warmup_epochs - 1:
            print(f" Warmup complete. LR now fixed at {self.target_lr:.6f}")

#------------------------------------------------------------------------------------------------------
# The Training Function 
def train_model():
    print("Start Training.....")

    # load data 
    print("Loading Dataset......")
    train_ds, val_ds, test_ds, class_names, class_weight_dict = get_data_loaders()

    class_weight_dict = None  # temporarily to check 
    
    # DATA VERIFICATION
    print("\n" + "="*50)
    print("DATA VERIFICATION")
    print("="*50)
    for images, labels in train_ds.take(1):
        print(f"Batch size: {images.shape[0]}")
        print(f"Image shape: {images.shape}")
        print(f"Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"Labels shape: {labels.shape}")
        
        # Check one-hot encoding
        label_sums = np.sum(labels.numpy(), axis=1)
        print(f"Label sums (first 3): {label_sums[:3]} (should all be 1.0)")
        break

    # build model 
    model = build_resnet(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), 
                         num_classes=config.NUM_CLASSES)
    model.summary(expand_nested=True)  # Show detailed structure

    # Compile the Model 
    print("Compiling Model.....")
    
    # Optimizer with gradient clipping
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # Gradient clipping to prevent explosions
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Loss with label smoothing for 7-class problem
    #loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    
    metrics = [CategoricalAccuracy(name='accuracy')]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Setup callbacks
    callbacks = [
        
        WarmUpCallback(warmup_epochs=WARMUP_EPOCHS, initial_lr=0.00003, target_lr=0.0003),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            'saved_models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping (more patient with GPU)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.002
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=5,  # Less frequent to save memory
            write_graph=True,
            profile_batch=0  # Disable profiling for stability
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,  # Wait 8 epochs before reducing LR
            min_lr=1e-6,
            verbose=1,
            cooldown=2  # Wait 2 epochs after LR reduction
        )
    ]

    # Train the model
    print("Training model....")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=NUM_EPOCHS, 
        callbacks=callbacks, 
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate on test set (Manual evaluation to avoid errors)
    print("Evaluating on test dataset.....")
    total_correct = 0
    total_samples = 0
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        total_correct += np.sum(pred_classes == true_classes)
        total_samples += len(images)
    
    test_accuracy = total_correct / total_samples
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_accuracy:.4f}")
    
    # Get final validation accuracy from history
    if history.history['val_accuracy']:
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"   Final Validation Accuracy: {final_val_accuracy:.4f}")
    
    # Plot training history
    print("Plotting training history.....")
    plot_training_history(history)
    
    # Save final model
    model.save('saved_models/final_model.h5')
    print(f" Model saved to 'saved_models/final_model.h5'")

    # Return results
    test_results = [0.0, test_accuracy]  # Dummy loss, real accuracy
    return model, history, test_results

#-------------------------------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train the model
    model, history, test_results = train_model()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print(f"   Final Test Accuracy: {test_results[1]:.4f}")
    print(f"   Model saved in: saved_models/")
    print(f"   Logs saved in: {LOG_DIR}")
    print("\nTo view training logs:")
    print(f"   tensorboard --logdir {LOG_DIR}")

