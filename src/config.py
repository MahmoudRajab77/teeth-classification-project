import os


class Config:
    
    #data paths
    DATA_DIR  = '/home/mahmoud/Desktop/Teeth_DataSet/Teeth_Dataset'
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    TEST_DIR  = os.path.join(DATA_DIR, 'Testing')
    VAL_DIR   = os.path.join(DATA_DIR, 'Validation')
    
    
    # Image Processing
    IMG_HEIGHT = 224
    IMG_WIDTH  = 224
    BATCH_SIZE = 32
    
    # Model
    NUM_CLASSES = 7