import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# --- Custom CSS for the Exact UI ---
st.markdown("""
    <style>
    /* Main Dark Theme background */
    .main { background-color: #0e1117; color: white; }
    
    /* Top Header Section */
    .header-text { text-align: center; font-family: 'sans-serif'; margin-bottom: 20px; }
    
    /* Diagnosis Box (Green Header) */
    .diag-box {
        background-color: #006400;
        color: white;
        padding: 24px;
        border-radius: 4px;
        border-left: 12px solid #a2d9a2;
        margin-bottom: 30px;
    }

    /* Buttons Styling (Green & Dark Rounded) */
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 24px;
        width: 100%;
        font-weight: bold;
        font-size: 16px;
    }
    /* Dark Camera Button */
    [data-testid="stVerticalBlock"] > div:nth-child(1) div.stButton > button {
        background-color: #262730;
    }

    /* Table Styling */
    .stTable { background-color: #1a1c24; border-radius: 8px; color: #e0e0e0; }

    /* Remove default padding */
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Configuration & Model Loader ---
CLASS_NAMES = [
    'Apple (Apple Scab)', 'Apple (Black Rot)', 'Apple (Cedar Rust)', 'Apple (Healthy)',
    'Blueberry (Healthy)', 'Cherry (Powdery Mildew)', 'Cherry (Healthy)',
    'Corn (Cercospora)', 'Corn (Common Rust)', 'Corn (Northern Blight)', 'Corn (Healthy)', 
    'Grape (Black Rot)', 'Grape (Esca)', 'Grape (Leaf Blight)', 'Grape (Healthy)',
    'Orange (Haunglongbing)', 'Peach (Bacterial Spot)', 'Peach (Healthy)',
    'Pepper Bell (Bacterial Spot)', 'Pepper Bell (Healthy)', 'Potato (Early Blight)',
    'Potato (Late Blight)', 'Potato (Healthy)', 'Raspberry (Healthy)', 'Soybean (Healthy)',
    'Squash (Powdery Mildew)', 'Strawberry (Leaf Scorch)', 'Strawberry (Healthy)',
    'Tomato (Bacterial Spot)', 'Tomato (Early Blight)', 'Tomato (Late Blight)', 'Tomato (Leaf Mold)',
    'Tomato (Septoria Spot)', 'Tomato (Spider Mites)', 'Tomato (Target Spot)', 
    'Tomato (Yellow Leaf Curl)', 'Tomato (Mosaic Virus)', 'Tomato (Healthy)'
]

@st.cache_resource
def load_trained_model():
    model_path = 'best_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

def is_leaf_check(image):
    """Filter to prevent faces from being diagnosed."""
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0].astype(int), img_np[:,:,1].astype(int), img_np[:,:,2].astype(int)
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 12.0

# --- User Interface Header ---
st.markdown("<div class='header-text'><h1>🌿 Plant Disease Detector</h1></div>", unsafe_allow_html=True)

model = load_trained_model()

# --- Action Buttons Layout ---
col1, col2 = st.columns(2)
with col1:
    st.button("📷 Open Camera")
with col2:
    st.button("📤 Upload Image")

st.markdown("<p style='text-align: center; color: #bbb; text-decoration: underline;'>View Gallery</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)
    
    if not is_leaf_check(image):
        st.error("⚠️ Validation Failed: Please scan a plant leaf.")
    else:
        with st.spinner("Analyzing..."):
            # --- Fix for InvalidArgumentError ---
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('float32') # Fix data type
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            
            # Predict
            preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            top_5_idx = np.argsort(preds[0])[-5:][::-1]
            
            # Diagnosis Banner
            st.markdown(f"""
                <div class="diag-box">
                    <h2 style='color: white; margin: 0;'>🌿 Diagnosis Confirmed!</h2>
                    <p style='font-size: 22px; margin: 10px 0;'>Primary Diagnosis: {CLASS_NAMES[top_5_idx[0]]}</p>
                    <p style='font-size: 18px; margin: 0;'>Confidence: {preds[0][top_5_idx[0]]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Detailed Predictions Table
            st.markdown("### 🏆 Detailed Top Predictions:")
            df = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Disease/Condition": [CLASS_NAMES[i] for i in top_5_idx],
                "Confidence (%)": [f"{preds[0][i]*100:.2f}" for i in top_5_idx]
            })
            st.table(df.set_index('Rank'))
            
            # Visualization
            st.markdown("### 💡 Confidence Distribution Visuallization (Top 5)")
            chart_data = pd.DataFrame({
                "Condition": [CLASS_NAMES[i] for i in top_5_idx],
                "Confidence (%)": [preds[0][i]*100 for i in top_5_idx]
            })
            fig = px.bar(chart_data, x="Confidence (%)", y="Condition", orientation='h',
                         color="Confidence (%)", color_continuous_scale='Greens')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig)
            
            # Final Button
            st.button("Get Recommendations")
