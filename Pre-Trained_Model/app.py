import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Adding the Path 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importing utils functions
from utils import load_model, preprocess_image, CLASS_NAMES, get_class_description

# Setting the page 
st.set_page_config(
    page_title="Teeth Classification App",
    page_icon="🦷",
    layout="wide"
)

# App title
st.title("Teeth Classification System")
st.markdown("---")

# loading the model 
@st.cache_resource
def get_model():
   
    model_path = '/content/teeth-classification-project/Pre-Trained_Model/src/saved_models/Pretrained_BestModel.h5'
    
    print(f"Looking for model at: {model_path}")
    
    if os.path.exists(model_path):
        print(" Model found!")
        return load_model(model_path)
    else:
        print(" Model not found!")
        return None

with st.spinner("Loading model..."):
    model = get_model()

if model is None:
    st.error("The Loading of the model failed!")
    st.stop()

st.success("Model loaded successfully!")

# 2 for images and results 
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose your teeth images to classify:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'],
        help="Images must be clear!"
    )
    
    if uploaded_file is not None:
        try:
            # Displaying uploaded images 
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.session_state['uploaded_image'] = image
            
        except Exception as e:
            st.error(f"Error opening image: {e}")
with col2:
    st.header("Classification Results")
    
    if uploaded_file is not None and 'uploaded_image' in st.session_state:
        if st.button("Classify", type="primary"):
            image = st.session_state['uploaded_image']
            with st.spinner("Image is loading..."):
                # preparing the image
                processed_image = preprocess_image(image)
                
                # predict the result
                predictions = model.predict(processed_image, verbose=0)
                
                # get the predicted class
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx]
                
                # showing result
                st.success(f"### Result: {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2%}")
                st.info(f"**Description:** {get_class_description(predicted_class)}")
                
                # Displaying all probabilities
                st.markdown("---")
                st.subheader("All Probabilities")
                
                # order the results descending
                results = [(CLASS_NAMES[i], predictions[0][i]) for i in range(len(CLASS_NAMES))]
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Progress bar
                for class_name, prob in results:
                    st.text(f"{class_name}: {prob:.2%}")
                    st.progress(float(prob))
    else:
        st.info("Please, Upload image first")

# Additional info 
st.markdown("---")
with st.expander("Info about the system:"):
    st.write("""
    **Teeth Classification System** is an AI model to classify teeth diseases.
    
    **Available Classes:**
    -  **CaS**: Caries 
    -  **CoS**: Caries - Another Type
    -  **Gum**: Gum disease 
    -  **MC**: Medical Condition 
    -  **OC**: Oral Condition 
    -  **OLP**: Oral Lichen Planus 
    -  **OT**: Other 
    
    **Model Accuracy over test data: 97.18%
    """)

# Footer
st.markdown("---")
st.markdown("Developed using TensorFlow and Streamlit")
