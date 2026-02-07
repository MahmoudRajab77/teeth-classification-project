import tensorflow as tf
from tensorflow import keras
from keras import layers
import os 
from config import Config 




# Creates and returns dataset objects for training, validation and testing 
def get_data_loaders():
   
    # create an object from class Config 
    config = Config()
    
    # a tuple to be used for image size 
    img_size = (config.IMG_HEIGHT, config.IMG_WIDTH)
    
    # Data augmentation for the training set 
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"), 
        layers.RandomRotation(0.2), 
        layers.RandomZoom(0.2),
    ])
    
    
    # a function to process each image 
    def preprocess_image(image_tensor, label_tensor, is_training=False):
         
        img = tf.cast(image_tensor, tf.float32) / 255.0
        
        # Resize to the target size
        img = tf.image.resize(img, img_size)
        
        # Apply augmentation only to training images
        if is_training:
            img = data_augmentation(img)
        return img, label_tensor
    
    
    # load datasets from directories 
    train_dataset = keras.utils.image_dataset_from_directory(
        Config.TRAIN_DIR, 
        image_size = img_size,
        batch_size = config.BATCH_SIZE,
        label_mode = 'categorical'      # this creates one-hot encoded labels for the 7 calsses 
    )
    
    val_dataset = keras.utils.image_dataset_from_directory(
        config.VAL_DIR, 
        image_size = img_size,
        batch_size = config.BATCH_SIZE, 
        label_mode = 'categorical'
    )
    
    test_dataset = keras.utils.image_dataset_from_directory(
        config.TEST_DIR, 
        image_size = img_size, 
        batch_size = config.BATCH_SIZE, 
        label_mode = 'categorical'
    )
    
    # Apply Preprocessing 
    train_dataset = train_dataset.map(lambda x, y: (preprocess_image(x, y, is_training=True)), num_parallel_calls = tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: (preprocess_image(x, y, is_training=False)), num_parallel_calls = tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (preprocess_image(x, y, is_training=False)), num_parallel_calls = tf.data.AUTOTUNE)
    
    
    # optimize datasets performance configuring caching and prefetching 
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    
    
    # Get the class names from the dataset's structure
    # The class names are stored in the 'class_names' attribute of the dataset's inner structure
    class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else None

    if class_names is None:
        # Alternative method: Get class names from the directory structure
        import os
        class_names = sorted(os.listdir(config.TRAIN_DIR))
        # Optional: Filter out only directories if needed
        class_names = [c for c in class_names if os.path.isdir(os.path.join(config.TRAIN_DIR, c))]

    print(f"Class names: {class_names}")
    
    
    return train_dataset, val_dataset, test_dataset, class_names



if __name__ == '__main__':
    train_ds, val_ds, test_ds, class_names = get_data_loaders()      # call the function and reccive the returned values 
    print ('Data was loaded successfully.')
    
    # check 1 batch as a sample 
    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
         