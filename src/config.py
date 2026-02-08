import os


class Config:
    
    #data paths
    DATA_DIR  = '/content/drive/MyDrive/Teeth_Dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    TEST_DIR  = os.path.join(DATA_DIR, 'Testing')
    VAL_DIR   = os.path.join(DATA_DIR, 'Validation')
    
    
    # Image Processing
    IMG_HEIGHT = 224
    IMG_WIDTH  = 224
    BATCH_SIZE = 64
    
    # Model
    NUM_CLASSES = 7
