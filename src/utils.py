import matplotlib.pyplot as plt
import numpy as np 
from data_loader import get_data_loaders
from config import Config



print ("loading dataset....")
train_ds, val_ds, test_ds, class_names = get_data_loaders()


def plot_calss_distribution(dataset, class_names, title):
    print (f"plotting {title}.....")    
    
    #count samples per class in the dataset 
    class_counts = np.zeros(len(class_names))

    # Count samples per class in the dataset
    class_counts = np.zeros(len(class_names))
    
    # Iterate through the dataset to count labels
    for _, labels in dataset:
        # Convert one-hot labels back to class indices
        class_indices = np.argmax(labels.numpy(), axis=1)
        for idx in class_indices:
            class_counts[idx] += 1
    
    # Create the bar plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, class_counts)
    plt.title(f'Class Distribution - {title}')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'class_distribution_{title.lower().replace(" ", "_")}.png')
    plt.show()


#--------------------------------------------------------------------------------


def display_sample_images(dataset, class_names, title, num_samples=8):
    print(f"Displaying {title}...")
    
    plt.figure(figsize=(12, 6))
    batch = next(iter(dataset.take(1)))
    images, labels = batch
    
    for i in range(min(num_samples, len(images))):
        # Get the class name from one-hot encoded label
        class_idx = np.argmax(labels[i].numpy())
        class_name = class_names[class_idx]
        
        # Display image
        plt.subplot(2, 4, i+1)
        # Convert from normalized [0,1] back to [0,255] for display
        img_display = (images[i].numpy() * 255).astype(np.uint8)
        plt.imshow(img_display)
        plt.title(f'{class_name}')
        plt.axis('off')
    
    plt.suptitle(f'{title} - Sample Images', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'sample_images_{title.lower().replace(" ", "_")}.png')
    plt.show()


#call the function to generate visualizations
# Plot class distribution for training set
plot_calss_distribution(train_ds, class_names, "Training Set")

# Display sample images from validation set (no augmentation)
display_sample_images(val_ds, class_names, "Validation Set (No Augmentation)")

# Display sample images from training set (with augmentation)
display_sample_images(train_ds, class_names, "Training Set (With Augmentation)")

print("\n✅ Visualization complete! Check the generated PNG files in your project folder.")



def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


