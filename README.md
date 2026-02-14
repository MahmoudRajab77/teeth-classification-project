# Dental Disease Classification using ResNet from Scratch

## Project Overview
This project implements a **ResNet architecture from scratch** using TensorFlow to classify dental images into **7 categories of oral diseases**.

## Project Structure
teeth-classification-project/

├── Pre-Trained_Model/

│ ├── app.py

│ ├── requirements.txt

│ ├── src/

  │ ├── config.py
  
  │ ├── data_loader.py
  
  │ ├── pretrained_model.py
  
  │ ├── train_pretrained.py
  
  │ ├── pretrained_training_history.png
  
  │ ├── utils.py
  
  │ ├── saved_models/
  
  │ ├── Pretrained_BestModel.h5
    
  │ ├── pretrained_final.h5

├── src/ # Model from scratch

│ ├── config.py # Configuration settings

│ ├── data_loader.py # Data loading and augmentation

│ ├── model.py # model architecture script

│ ├── train.py # Training scripts

│ ├── utils.py # Visualization scripts

│ └── requirements.txt # Python dependencies

├── .gitignore # Files to ignore (dataset, etc.)

└── README.md # This file



text

## Current Progress (Week 1)
- ✅ **Data Preprocessing**: Implemented image loading, resizing, and augmentation.
- ✅ **Data Visualization**: Created class distribution plots and sample displays.
- 🚧 **Model Architecture**: Built ResNet from scratch.
- ⏳ **Training & Evaluation**: completed.

- The model is inspired by the paper found in this link: https://drive.google.com/file/d/1AQH_tkjcMzxrpddNNH_eyxpBtA6kwgZW/view?usp=sharing

## How to Run
1. Clone the repository: `git clone https://github.com/MahmoudRajab77/dental-disease-classification-resnet.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the train file: `python src/train.py`

## Notes
- The dataset (`Teeth_Dataset/`) is not included in this repository due to size and privacy.
