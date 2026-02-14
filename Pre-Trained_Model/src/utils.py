import tensorflow as tf
import numpy as np
from PIL import Image

# Loading the pretrained model 
@tf.keras.utils.register_keras_serializable()
def load_model(model_path='saved_models/Pretrained_BestModel.h5'):
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f" Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f" Error loading model: {e}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    
    # change the size
    image = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(image)
    
    # Convert to RGB if it is gray scale
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # if image is RGBA, only take the first three channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
        
    img_array = img_array / 255.0
    
    # Adding batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Classes Names
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Description for each class
CLASS_DESCRIPTIONS = {
    'CaS': 'Caries',
    'CoS': 'Caries - Another type',
    'Gum': 'Gum disease',
    'MC': 'Medical Condition',
    'OC': 'Oral Condition',
    'OLP': 'Oral Lichen Planus',
    'OT': 'Other'
}

def get_class_name(class_idx):
    """getting class name from number"""
    return CLASS_NAMES[class_idx]

def get_class_description(class_name):
    """getting class description"""
    return CLASS_DESCRIPTIONS.get(class_name, 'No description available')
