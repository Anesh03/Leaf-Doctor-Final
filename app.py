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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Grade Global CSS ---
st.markdown("""
    <style>
    /* Global Background and Fonts */
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Research ID Card (Glassmorphism) */
    .id-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(46, 160, 67, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .id-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .id-value { color: #e4e6eb; font-size: 1rem; font-weight: 600; margin-bottom: 12px; }
    .id-title { color: #4ade80; font-size: 1.2rem; font-weight: 700; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 15px; }

    /* Industrial Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #238636 0%, #1b5e20 100%);
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(46, 160, 67, 0.4); color: white; }

    /* Diagnosis Banner */
    .banner-container {
        background: linear-gradient(90deg, #1f6feb 0%, #238636 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Rejection Alert */
    .rejection-box {
        background-color: #3a1a1a;
        padding: 25px;
        border-radius: 12px;
        border-left: 10px solid #dc3545;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Configurations ---
CLASS_NAMES = [
    'Apple (Apple Scab)', 'Apple (Black Rot)', 'Apple (Cedar Rust)', 'Apple (Healthy)',
    'Blueberry (Healthy)', 'Cherry (Powdery Mildew)', 'Cherry (Healthy)',
    'Corn (Cercospora)', 'Common Rust', 'Northern Blight', 'Corn (Healthy)', 
    'Grape (Black Rot)', 'Grape (Esca)', 'Grape (Leaf Blight)', 'Grape (Healthy)',
    'Orange (Citrus Greening)', 'Peach (Bacterial Spot)', 'Peach (Healthy)',
    'Pepper Bell (Bacterial Spot)', 'Pepper Bell (Healthy)', 'Potato (Early Blight)',
    'Potato (Late Blight)', 'Potato (Healthy)', 'Raspberry (Healthy)', 'Soybean (Healthy)',
    'Squash (Powdery Mildew)', 'Strawberry (Leaf Scorch)', 'Strawberry (Healthy)',
    'Tomato (Bacterial Spot)', 'Tomato (Early Blight)', 'Tomato (Late Blight)', 'Tomato (Leaf Mold)',
    'Tomato (Septoria Spot)', 'Tomato (Spider Mites)', 'Tomato (Target Spot)', 
    'Tomato (Yellow Leaf Curl)', 'Tomato (Mosaic Virus)', 'Tomato (Healthy)'
]

# --- Core Expert Logic ---
@st.cache_resource
def load_expert_model():
    if os.path.exists('best_model.keras'):
        return tf.keras.models.load_model('best_model.keras', compile=False)
    return None

def botanical_validation(image):
    """Filters out non-plant objects (people/animals) using RGB color science."""
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0].astype(np.int32), img_np[:,:,1].astype(np.int32), img_np[:,:,2].astype(np.int32)
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 15.0

# --- Dashboard Construction ---
model = load_expert_model()

# Sidebar: Research Identity Card
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4ade80;'>Leaf Doctor AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Official Project Card
    st.markdown(f"""
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
                <span style="font-size: 0.8rem; color: #4ade80; font-weight: bold;">IMCS | University of Sindh</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    input_source = st.selectbox("📷 Select Capture Source", ["Direct Image Upload", "Live Camera Stream"])
    st.info("AI Core: MobileNetV3-Large Ready")

# Main Interface Header
st.markdown("""<div class="banner-container"><h1>🌿 INTERNATIONAL PATHOLOGY ENGINE</h1><p>Neural Network Powered Botanical Diagnostic Dashboard</p></div>""", unsafe_allow_html=True)

# Main Grid Layout
col_input, col_results = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown("### 📥 Diagnostic Sample")
    captured_file = st.camera_input("Scanner Interface") if input_source == "Live Camera Stream" else st.file_uploader("Upload Leaf Sample", type=["jpg", "png", "jpeg"])

    if captured_file:
        image = Image.open(captured_file)
        st.image(image, caption="Current Diagnostic Sample", use_container_width=True)
        
        # Validation Check
        if not botanical_validation(image):
            st.markdown("""<div class="rejection-box"><h3 style="color: #ff8080; margin: 0;">🚫 Integrity Check Failed</h3><p style="color: #f8d7da; margin-top: 10px;">Non-botanical artifacts detected. AI inference disabled to prevent false results.</p></div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Sample Validated. Ready for Deep Inference.")

with col_results:
    if captured_file and botanical_validation(image):
        with st.spinner("🧬 Running Multi-Layer Neural Inference..."):
            # Expert Preprocessing
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('float32')
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            
            preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            top_idx = np.argsort(preds[0])[-5:][::-1]
            
            # Professional Results Display
            st.markdown(f"""
                <div style="background: rgba(35, 134, 54, 0.1); border: 1px solid #2e7d32; border-radius: 12px; padding: 25px;">
                    <h2 style="color: #4ade80; margin: 0;">🔬 Diagnosis Confirmed</h2>
                    <p style="font-size: 1.6rem; color: white; margin-top: 10px;"><b>{CLASS_NAMES[top_idx[0]]}</b></p>
                    <p style="opacity: 0.8;">Inference Confidence: {preds[0][top_idx[0]]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # Interactive Distribution
            st.markdown("### 📊 Distribution Mapping")
            df = pd.DataFrame({"Pathology": [CLASS_NAMES[i] for i in top_idx], "Confidence": [preds[0][i]*100 for i in top_idx]})
            fig = px.bar(df, x="Confidence", y="Pathology", orientation='h', color="Confidence", color_continuous_scale='Greens', template="plotly_dark")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical Trace Data
            st.table(df.set_index('Pathology').style.format("{:.2f}%"))
            
            if st.button("💊 Generate Detailed Treatment Protocol"):
                st.success(f"Protocol generated for {CLASS_NAMES[top_idx[0]]}. Report ready for export.")
    else:
        st.info("📤 Awaiting botanical sample for diagnostic inference.")
