import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- Expert Page Configuration ---
st.set_page_config(
    page_title="Leaf Doctor AI | Professional Pathology",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Grade Custom CSS ---
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* Sidebar styling for a pro look */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Project ID Card View in Sidebar */
    .id-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(46, 160, 67, 0.3);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .id-label { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .id-value { color: #e4e6eb; font-size: 0.95rem; font-weight: 600; margin-bottom: 12px; }
    .id-title { color: #4ade80; font-size: 1.1rem; font-weight: 700; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 15px; }

    /* Card Glassmorphism Effect */
    .element-container div.stMarkdown div.diag-card {
        background: rgba(35, 134, 54, 0.1);
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        margin-top: 20px;
    }

    /* Professional Diagnostic Banner */
    .banner {
        background: linear-gradient(90deg, #1f6feb 0%, #238636 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }

    /* Buttons: Industrial Style */
    div.stButton > button {
        background-color: #238636;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid rgba(240,246,252,0.1);
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover { background-color: #2ea043; border-color: #8b949e; transform: translateY(-1px); }
    
    /* Rejection Alert Styling */
    .rejection-box { background-color: #3a1a1a; padding: 25px; border-radius: 12px; border-left: 10px solid #dc3545; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- Global Configurations ---
CLASS_NAMES = [
    'Apple (Apple Scab)', 'Apple (Black Rot)', 'Apple (Cedar Rust)', 'Apple (Healthy)',
    'Blueberry (Healthy)', 'Cherry (Powdery Mildew)', 'Cherry (Healthy)',
    'Corn (Cercospora)', 'Corn (Common Rust)', 'Corn (Northern Blight)', 'Corn (Healthy)', 
    'Grape (Black Rot)', 'Grape (Esca)', 'Grape (Leaf Blight)', 'Grape (Healthy)',
    'Orange (Citrus Greening)', 'Peach (Bacterial Spot)', 'Peach (Healthy)',
    'Pepper Bell (Bacterial Spot)', 'Pepper Bell (Healthy)', 'Potato (Early Blight)',
    'Potato (Late Blight)', 'Potato (Healthy)', 'Raspberry (Healthy)', 'Soybean (Healthy)',
    'Squash (Powdery Mildew)', 'Strawberry (Leaf Scorch)', 'Strawberry (Healthy)',
    'Tomato (Bacterial Spot)', 'Tomato (Early Blight)', 'Tomato (Late Blight)', 'Tomato (Leaf Mold)',
    'Tomato (Septoria Spot)', 'Tomato (Spider Mites)', 'Tomato (Target Spot)', 
    'Tomato (Yellow Leaf Curl)', 'Tomato (Mosaic Virus)', 'Tomato (Healthy)'
]

@st.cache_resource
def load_expert_model():
    if os.path.exists('best_model.keras'):
        return tf.keras.models.load_model('best_model.keras', compile=False)
    return None

def botanical_validation(image):
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0].astype(np.int32), img_np[:,:,1].astype(np.int32), img_np[:,:,2].astype(np.int32)
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 15.0

# --- Sidebar Controls with Research Identity Card ---
model = load_expert_model()
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4ade80;'>Leaf Doctor AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
        <div class="id-card">
            <div class="id-label">Project Title</div>
            <div class="id-title">Plant Disease Detection</div>
            <div class="id-label">Researcher Name</div>
            <div class="id-value">Anesh Meghwar</div>
            <div class="id-label">Roll Number</div>
            <div class="id-value">2K22/CSE/21</div>
            <div class="id-label">Under Supervision of</div>
            <div class="id-value">Dr. Ayaz Keerio</div>
            <div style="text-align: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid #30363d;">
                <span style="font-size: 0.75rem; color: #4ade80; font-weight: bold;">IMCS | University of Sindh</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📷 Input Interface")
    input_source = st.selectbox("Select Capture Source", ["Direct Image Upload", "Live Camera Stream"])

# --- Main Interface ---
st.markdown("""<div class="banner"><h1>🌿 LEAF DOCTOR AI</h1><p>Advanced Neural Network for Plant Pathology Detection</p></div>""", unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1.5], gap="large")

with col_input:
    st.markdown("### 📥 Diagnostic Sample")
    captured_file = st.camera_input("Scanner Interface") if input_source == "Live Camera Stream" else st.file_uploader("Upload Leaf Sample (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if captured_file:
        image = Image.open(captured_file)
        st.image(image, caption="Current Diagnostic Sample", use_container_width=True)
        if not botanical_validation(image):
            st.markdown("""<div class="rejection-box"><h3 style="color: #ff8080; margin: 0;">🚫 Integrity Check Failed</h3><p style="color: #f8d7da; margin-top: 10px;">The <b>Pathology Engine</b> detected non-botanical artifacts. Inference disabled to prevent false diagnostics.</p></div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Botanical Sample validated. Ready for deep inference.")

with col_results:
    if captured_file and botanical_validation(image):
        with st.spinner("🚀 Running Multi-Layer Neural Inference..."):
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('float32')
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            top_idx = np.argsort(preds[0])[-5:][::-1]
            
            st.markdown(f"""<div class="diag-card"><h2 style="color: #4ade80; margin: 0;">🔬 Diagnosis Confirmed</h2><p style="font-size: 1.5rem; color: white;">Pathology: <b>{CLASS_NAMES[top_idx[0]]}</b></p><p style="opacity: 0.8;">Inference Confidence: {preds[0][top_idx[0]]*100:.2f}%</p></div>""", unsafe_allow_html=True)

            df = pd.DataFrame({"Pathology": [CLASS_NAMES[i] for i in top_idx], "Confidence": [preds[0][i]*100 for i in top_idx]})
            fig = px.bar(df, x="Confidence", y="Pathology", orientation='h', color="Confidence", color_continuous_scale='Greens', template="plotly_dark")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.table(df.set_index('Pathology').style.format("{:.2f}%"))
    else:
        st.info("📤 Please provide a botanical leaf sample to initiate diagnostic inference.")
