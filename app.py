import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(page_title="Leaf Doctor AI", page_icon="🌿", layout="centered")

# --- Advanced Professional UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    
    /* Title and Header */
    .app-header { text-align: center; padding: 10px; margin-bottom: 20px; }
    .app-header h1 { font-size: 3rem; color: #4CAF50; margin-bottom: 0; }
    
    /* Professional Diagnosis Card */
    .diag-card {
        background: linear-gradient(135deg, #1b5e20 0%, #002300 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        border-left: 15px solid #81c784;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        margin-bottom: 35px;
    }

    /* Modern Glassmorphism Buttons */
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 50px;
        border: 2px solid #4CAF50;
        padding: 15px 30px;
        width: 100%;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }

    /* Table & Chart Container */
    .results-container {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    
    .stTable { background-color: transparent; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic & Model ---
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
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0].astype(int), img_np[:,:,1].astype(int), img_np[:,:,2].astype(int)
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 12.0

# --- App Structure ---
st.markdown("<div class='app-header'><h1>🌿 Leaf Doctor AI</h1><p>International Plant Health Diagnostic Dashboard</p></div>", unsafe_allow_html=True)

model = load_trained_model()

# Camera Toggle Logic
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Use Camera"):
        st.session_state.show_camera = not st.session_state.show_camera
with col2:
    # This acts as a trigger to clear camera and show uploader
    if st.button("📤 Upload File"):
        st.session_state.show_camera = False

# Input Section
uploaded_file = None
if st.session_state.show_camera:
    uploaded_file = st.camera_input("Scan your plant leaf")
else:
    uploaded_file = st.file_uploader("Select image from gallery", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Current Scan", use_container_width=True)
    
    if not is_leaf_check(image):
        st.error("⚠️ Scan Rejected: No plant leaf detected. Please center the leaf in the frame.")
    else:
        with st.spinner("🧬 Deep Learning Analysis in Progress..."):
            # Fix for InvalidArgumentError
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('float32')
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            
            preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            top_5_idx = np.argsort(preds[0])[-5:][::-1]
            
            # --- Results Card ---
            st.markdown(f"""
                <div class="diag-card">
                    <h2 style='margin: 0;'>✅ Diagnosis Confirmed!</h2>
                    <p style='font-size: 26px; margin: 15px 0;'>Primary: <b>{CLASS_NAMES[top_5_idx[0]]}</b></p>
                    <p style='font-size: 20px; opacity: 0.9;'>AI Confidence Level: {preds[0][top_5_idx[0]]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # --- Detailed Table & Chart ---
            st.markdown("### 🏆 Competitor Analysis (Top 5)")
            df = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Condition": [CLASS_NAMES[i] for i in top_5_idx],
                "Confidence (%)": [preds[0][i]*100 for i in top_5_idx]
            })
            
            # Formatted Table
            st.table(df.set_index('Rank').style.format({"Confidence (%)": "{:.2f}%"}))
            
            # Visual Chart
            fig = px.bar(df, x="Confidence (%)", y="Condition", orientation='h',
                         color="Confidence (%)", color_continuous_scale='Greens',
                         title="Confidence Distribution Mapping")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig)
            
            st.button("🏥 View Full Treatment Plan")
