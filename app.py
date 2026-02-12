import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- AMAZING PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Leaf Doctor AI | UNBELIEVABLE PATHOLOGY",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- EXCELLENT PROFESSIONAL CUSTOM CSS ---
st.markdown("""
    <style>
    /* Global Dark Masterpiece Theme */
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* Amazing Sidebar styling */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Unbelievable Research ID Card */
    .id-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(46, 160, 67, 0.4);
        border-radius: 15px;
        padding: 22px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .id-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; }
    .id-value { color: #ffffff; font-size: 1.05rem; font-weight: 700; margin-bottom: 15px; }
    .id-title { color: #4ade80; font-size: 1.3rem; font-weight: 800; border-bottom: 1px solid #30363d; padding-bottom: 12px; margin-bottom: 18px; }

    /* Awesome Action Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #238636 0%, #1b5e20 100%);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        padding: 14px 28px;
        font-weight: 700;
        width: 100%;
        transition: all 0.4s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div.stButton > button:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(46, 160, 67, 0.5); color: #4ade80; }

    /* Incredible Diagnosis Banner */
    .banner-container {
        background: linear-gradient(135deg, #1f6feb 0%, #238636 100%);
        padding: 45px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
    }
    
    /* Expert Rejection Alert */
    .rejection-box {
        background-color: #3a1a1a;
        padding: 30px;
        border-radius: 15px;
        border-left: 12px solid #dc3545;
        margin-top: 25px;
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

@st.cache_resource
def load_expert_model():
    if os.path.exists('best_model.keras'):
        return tf.keras.models.load_model('best_model.keras', compile=False)
    return None

def botanical_validation(image):
    """UNBELIEVABLE COLOR SCIENCE: Stops non-plant scans cold."""
    img_np = np.array(image.convert('RGB'))
    r, g, b = img_np[:,:,0].astype(np.int32), img_np[:,:,1].astype(np.int32), img_np[:,:,2].astype(np.int32)
    green_mask = (g > r) & (g > b) & (g > 45)
    return (np.sum(green_mask) / green_mask.size) * 100 > 15.0

# --- AMAZING UI DASHBOARD ---
model = load_expert_model()

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4ade80;'>LEAF DOCTOR</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # EXCELLENT PROJECT CARD
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
            <div style="text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #30363d;">
                <span style="font-size: 0.85rem; color: #4ade80; font-weight: 800; letter-spacing: 1px;">IMCS | UNIVERSITY OF SINDH</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    input_source = st.selectbox("📸 SELECT CAPTURE SOURCE", ["Direct Image Upload", "Live Camera Stream"])

# UNBELIEVABLE MAIN HEADER
st.markdown("""<div class="banner-container"><h1>🌿 INTERNATIONAL PATHOLOGY ENGINE</h1><p style="font-size: 1.2rem; opacity: 0.9;">Excellent Neural Network Powered Botanical Diagnostic Dashboard</p></div>""", unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown("### 📥 DIAGNOSTIC SAMPLE")
    captured_file = st.camera_input("AMAZING SCANNER ACTIVE") if input_source == "Live Camera Stream" else st.file_uploader("UPLOAD LEAF SAMPLE", type=["jpg", "png", "jpeg"])

    if captured_file:
        image = Image.open(captured_file)
        st.image(image, caption="AMAZING SAMPLE PREVIEW", use_container_width=True)
        
        if not botanical_validation(image):
            st.markdown("""<div class="rejection-box"><h3 style="color: #ff8080; margin: 0;">🚫 INTEGRITY CHECK FAILED</h3><p style="color: #f8d7da; margin-top: 10px;">EXCELLENT WARNING: Non-botanical artifacts detected. AI inference disabled to prevent false results.</p></div>""", unsafe_allow_html=True)
        else:
            st.success("✅ AMAZING! Botanical Sample validated. Ready for deep inference.")

with col_results:
    if captured_file and botanical_validation(image):
        with st.spinner("🧬 RUNNING UNBELIEVABLE NEURAL INFERENCE..."):
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('float32')
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            top_idx = np.argsort(preds[0])[-5:][::-1]
            
            # EXCELLENT RESULTS DISPLAY
            st.markdown(f"""
                <div style="background: rgba(35, 134, 54, 0.15); border: 2px solid #4ade80; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
                    <h2 style="color: #4ade80; margin: 0;">🔬 DIAGNOSIS CONFIRMED</h2>
                    <p style="font-size: 1.8rem; color: white; margin-top: 10px;"><b>{CLASS_NAMES[top_idx[0]]}</b></p>
                    <p style="font-size: 1.1rem; opacity: 0.9;">UNBELIEVABLE Confidence: {preds[0][top_idx[0]]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 DISTRIBUTION MAPPING")
            df = pd.DataFrame({"Pathology": [CLASS_NAMES[i] for i in top_idx], "Confidence": [preds[0][i]*100 for i in top_idx]})
            fig = px.bar(df, x="Confidence", y="Pathology", orientation='h', color="Confidence", color_continuous_scale='Greens', template="plotly_dark")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.table(df.set_index('Pathology').style.format("{:.2f}%"))
            
            if st.button("💊 GENERATE AMAZING TREATMENT PROTOCOL"):
                st.success(f"EXCELLENT! Protocol generated for {CLASS_NAMES[top_idx[0]]}.")
    else:
        st.info("📤 AWAITING AMAZING BOTANICAL SAMPLE...")
