import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# --- Custom CSS for the Exact UI in your Screenshots ---
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #0e1117; color: white; }
    
    /* Header Container */
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

    /* Professional Buttons (Green rounded) */
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 24px;
        width: 100%;
        font-weight: bold;
        font-size: 16px;
        transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #1b5e20; border: none; color: white; }

    /* Table Styling */
    .stTable { background-color: #1a1c24; border-radius: 8px; color: #e0e0e0; }

    /* Remove Streamlit default padding */
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
        return tf.keras.models.load_model(model_path)
    return None

def is_leaf_check(image):
    """Green pixel filter to prevent misidentifying non-plant objects."""
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 12.0

# --- User Interface Header ---
st.markdown("<div class='header-text'><h1>🔬 International Plant Disease Diagnostic Dashboard 🌍</h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #bbb;'>Upload a <b>clear, focused photo</b> of the affected plant leaf for immediate AI analysis.</p>", unsafe_allow_html=True)
st.markdown("---")

# Model Status Alert
model = load_trained_model()
if model:
    st.success("✅ Model loaded successfully and ready for prediction!")
else:
    st.error("❌ Model 'best_model.keras' not found in the project directory.")
    st.stop()

# --- Image Input Source ---
st.subheader("🖼️ Image Input Source")
col_cam, col_up = st.columns(2)
with col_cam:
    cam_click = st.button("📷 Open Camera")
with col_up:
    up_click = st.button("📤 Upload Image")

# Handle input types
uploaded_file = st.file_uploader("Upload a Plant Leaf Image (JPG/PNG)", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    # 1. Validation (Green Filter)
    if not is_leaf_check(image):
        st.markdown("""
            <div style='background-color: #3a1a1a; padding: 20px; border-radius: 5px; border-left: 10px solid #dc3545;'>
                <h3 style='color: white; margin: 0;'>⚠️ Validation Failed</h3>
                <p style='color: #f8d7da;'>System detected a <b>non-plant object</b>. Please scan a plant leaf.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing Leaf..."):
            # 2. Preprocessing
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            
            # 3. Prediction
            preds = model.predict(np.expand_dims(img_array, axis=0))
            top_5_idx = np.argsort(preds[0])[-5:][::-1]
            
            main_label = CLASS_NAMES[top_5_idx[0]]
            main_conf = preds[0][top_5_idx[0]] * 100
            
            # 4. Results Display (Diagnosis Box)
            st.markdown(f"""
                <div class="diag-box">
                    <h2 style='color: white; margin: 0;'>🌿 Diagnosis Confirmed!</h2>
                    <p style='font-size: 22px; margin: 10px 0;'>Primary Diagnosis: <b>{main_label}</b></p>
                    <p style='font-size: 18px; margin: 0;'>Confidence: {main_conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # 5. Detailed Table
            st.markdown("### 🏆 Detailed Top Predictions:")
            results_df = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Disease/Condition": [CLASS_NAMES[i] for i in top_5_idx],
                "Confidence (%)": [f"{preds[0][i]*100:.2f}" for i in top_5_idx]
            })
            st.table(results_df.set_index('Rank'))
            
            # 6. Bar Chart Visualization
            st.markdown("### 💡 Confidence Distribution Visualization (Top 5)")
            chart_data = pd.DataFrame({
                "Condition": [CLASS_NAMES[i] for i in top_5_idx],
                "Confidence (%)": [preds[0][i]*100 for i in top_5_idx]
            })
            fig = px.bar(chart_data, x="Confidence (%)", y="Condition", orientation='h',
                         color="Confidence (%)", color_continuous_scale='Greens')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig)
            
            # 7. Action Button
            if st.button("Get Recommendations"):
                st.info(f"Generating organic and chemical treatment steps for {main_label}...")
