import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import time

# Adding the Path 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importing utils functions
from utils import load_model, preprocess_image, CLASS_NAMES, get_class_description

# Setting the page 
st.set_page_config(
    page_title="Teeth Classification App",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .stButton > button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# App title with emoji
st.title("🦷 Teeth Classification System")
st.markdown("AI-powered dental condition classifier with **97.18% accuracy**")
st.markdown("---")

# loading the model with progress bar
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the pretrained model with path handling"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(" Locating model file...")
    progress_bar.progress(10)
    time.sleep(0.5)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(current_dir, 'src', 'saved_models', 'Pretrained_BestModel.h5'),
        os.path.join(current_dir, 'Pretrained_BestModel.h5'),
        os.path.join(current_dir, 'saved_models', 'Pretrained_BestModel.h5'),
        os.path.join('/mount/src/teeth-classification-project/Pre-Trained_Model', 
                     'src', 'saved_models', 'Pretrained_BestModel.h5')
    ]
    
    status_text.text(" Searching in multiple locations...")
    progress_bar.progress(30)
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            status_text.text(f"Found model at: {os.path.basename(os.path.dirname(path))}")
            progress_bar.progress(50)
            break
    
    if model_path is None:
        status_text.text("Model not found in any location")
        progress_bar.empty()
        return None
    
    status_text.text(" Loading model (this may take 30-60 seconds)...")
    progress_bar.progress(70)
    
    try:
        model = load_model(model_path)
        progress_bar.progress(100)
        status_text.text(" Model loaded successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        return model
    except Exception as e:
        status_text.text(f" Error: {str(e)[:50]}...")
        progress_bar.empty()
        return None

# Load model with spinner
model_placeholder = st.empty()
with model_placeholder.container():
    with st.spinner("Initializing model..."):
        model = get_model()

if model is None:
    st.error("**Failed to load model!** Please check:")
    st.markdown("""
    - Ensure model file exists in `src/saved_models/`
    - Check file permissions
    - Verify TensorFlow installation
    """)
    st.stop()

st.success("**Model loaded successfully!** Ready for classification.")
st.markdown("---")

# 2 columns for images and results 
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("Upload Image")
    st.markdown("Upload a clear dental image for classification")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Supported formats: JPG, JPEG, PNG, BMP, WEBP"
    )
    
    if uploaded_file is not None:
        # Displaying uploaded images 
        image = Image.open(uploaded_file)
        
        # Show image with caption
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Image info
        st.markdown(f"**Image info:** {image.size[0]}x{image.size[1]} pixels")

with col2:
    st.markdown("###  Classification Results")
    
    if uploaded_file is not None:
        if st.button(" **Classify Image**", type="primary", use_container_width=True):
            with st.spinner(" Processing image..."):
                # preparing the image
                processed_image = preprocess_image(image)
                
                # predict the result
                predictions = model.predict(processed_image, verbose=0)[0]
                
                # get the predicted class
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[predicted_class_idx]
                
                # Create metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("Predicted Class", predicted_class)
                with col_metric2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Description
                st.info(f"**Description:** {get_class_description(predicted_class)}")
                
                # Displaying all probabilities
                st.markdown("---")
                st.markdown(" Class Probabilities")
                
                # order the results descending
                results = [(CLASS_NAMES[i], predictions[i]) for i in range(len(CLASS_NAMES))]
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Progress bars with colors based on confidence
                for class_name, prob in results:
                    # Color coding
                    if class_name == predicted_class:
                        st.markdown(f"** {class_name}**")
                    else:
                        st.markdown(f"**{class_name}**")
                    
                    # Progress bar
                    st.progress(float(prob), text=f"{prob:.2%}")
                    
                # Add to history (optional)
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)
                
                st.session_state.history.append({
                    'class': predicted_class,
                    'confidence': confidence,
                    'time': time.strftime("%H:%M:%S")
                })
    else:
        st.info("**Please upload an image first**")
        # Show placeholder
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 50px; text-align: center; border-radius: 10px;">
            <p style="color: #666; font-size: 18px;"> Upload an image to see results</p>
        </div>
        """, unsafe_allow_html=True)

# Additional info 
st.markdown("---")
with st.expander(" **About the System**"):
    
    tab1, tab2, tab3 = st.tabs([" Classes", " Performance", " How to Use"])
    
    with tab1:
        st.markdown("""
        | Class | Description |
        |-------|-------------|
        | **CaS** | Caries - Tooth decay |
        | **CoS** | Caries - Another type |
        | **Gum** | Gum disease / Periodontitis |
        | **MC** | Medical Condition |
        | **OC** | Oral Condition |
        | **OLP** | Oral Lichen Planus |
        | **OT** | Other dental conditions |
        """)
    
    with tab2:
        st.markdown(f"""
        **Model Performance:**
        -  **Test Accuracy:** 97.18%
        -  **Model Type:** MobileNetV2 (Pretrained)
        -  **Input Size:** 224x224 pixels
        -  **Classes:** 7 dental conditions
        """)
        
        # Simple gauge for accuracy
        st.markdown("**Model Accuracy:**")
        st.progress(0.9718, text="97.18%")
    
    with tab3:
        st.markdown("""
        1. **Upload** a clear dental image
        2. Click **"Classify Image"**
        3. View results with confidence scores
        4. Check class probabilities
        """)

# History section (optional)
if 'history' in st.session_state and len(st.session_state.history) > 0:
    with st.expander(" **Recent Classifications**"):
        for item in reversed(st.session_state.history):
            col_time, col_class, col_conf = st.columns([2, 2, 1])
            with col_time:
                st.markdown(f" {item['time']}")
            with col_class:
                st.markdown(f"**{item['class']}**")
            with col_conf:
                st.markdown(f"_{item['confidence']:.2%}_")
            st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Developed with using TensorFlow & Streamlit | Version 2.0</p>
    <p style="font-size: 12px;">© 2026 Teeth Classification System</p>
</div>
""", unsafe_allow_html=True)
