import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

# Ensure local imports work cleanly
app_dir = Path(__file__).parent.resolve()
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from model_loader import (
    load_fruit_classifier,
    load_quality_classifier,
    FRUITS_13,
    QUALITY_CLASSES,
    BEST_MODEL_CONFIG
)
from utils import (
    preprocess_image,
    apply_clahe,
    get_random_test_sample,
    plot_fruit_probabilities,
    plot_quality_probabilities,
    FRUIT_METADATA,
    QUALITY_METADATA,
    IMG_SIZE
)

# Page configuration
st.set_page_config(
    page_title="Fruit Quality Inspection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for reduced top padding and minimalist layout
st.markdown("""
<style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.8rem !important;
    }
    .metric-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }
    .metric-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-badge {
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
    }
    .badge-match {
        color: #34D399;
        background: rgba(52, 211, 153, 0.15);
    }
    .badge-diff {
        color: #F87171;
        background: rgba(248, 113, 113, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Caching
@st.cache_resource(show_spinner="Loading Fruit Classifier...")
def get_cached_fruit_model(mode: str):
    return load_fruit_classifier(mode=mode, device=device)

@st.cache_resource(show_spinner="Loading Quality Classifier...")
def get_cached_quality_model(fruit_name: str, mode: str):
    return load_quality_classifier(fruit_name, mode=mode, device=device)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### Settings")

    mode_selection = st.radio(
        "Model Architecture:",
        options=["Best Models", "Custom CNN"],
        index=0
    )
    mode_key = "best" if mode_selection == "Best Models" else "custom_cnn"

    st.markdown("---")
    conf_threshold = st.slider(
        "Stage 1 Gate Threshold:",
        min_value=0.50,
        max_value=0.95,
        value=0.85,
        step=0.05
    )

    st.markdown("---")
    st.caption(f"Device: `{device.type.upper()}`")
    st.caption("13-Fruits ResNet-50: `99.93%` | Custom CNN: `96.37%`")
    st.caption("Quality Mean: `96.99%` (Transfer Champions)")

# --- Header ---
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.5rem;">
        <span style="font-size: 1.45rem; font-weight: 700;">Fruit Variety & Quality Inspection</span>
        <span style="font-size: 0.85rem; color: #94A3B8;">
            Mode: <b>{"Best Models" if mode_key == "best" else "Custom CNN"}</b> &nbsp;|&nbsp; 
            Gate: <b>{conf_threshold*100:.0f}%</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Top Action Bar: Upload & Random Side by Side ---
col_upload, col_random = st.columns([1, 1], gap="medium")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Image:",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key=f"file_uploader_{st.session_state.get('uploader_key', 0)}",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        st.session_state['active_image'] = Image.open(uploaded_file)
        st.session_state['active_source'] = 'upload'
        st.session_state['ground_truth'] = None

with col_random:
    r_c1, r_c2, r_c3 = st.columns([1.1, 1.1, 1.3], vertical_alignment="bottom")
    with r_c1:
        fruit_filter = st.selectbox("Fruit:", ["All Fruits"] + FRUITS_13, label_visibility="collapsed")
    with r_c2:
        quality_filter = st.selectbox("Quality:", ["All Qualities", "good", "medium", "bad"], label_visibility="collapsed")
    with r_c3:
        if st.button("Random Test Sample", use_container_width=True):
            sample = get_random_test_sample(fruit_filter, quality_filter)
            if sample:
                st.session_state['active_image'] = Image.open(sample['full_path'])
                st.session_state['active_source'] = 'test_set'
                st.session_state['ground_truth'] = sample
                st.session_state['uploader_key'] = st.session_state.get('uploader_key', 0) + 1

st.markdown("<hr style='margin: 0.5rem 0 0.8rem 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);' />", unsafe_allow_html=True)

# --- Main Display Area ---
active_image = st.session_state.get('active_image')

if active_image is not None:
    col_img, col_pred = st.columns([1.0, 1.8], gap="large")

    with col_img:
        st.image(active_image, use_container_width=True)

        if st.session_state.get('ground_truth'):
            gt = st.session_state['ground_truth']
            st.caption(f"Ground Truth: **{gt['fruit'].replace('_', ' ').title()}** &bull; **{gt['quality'].upper()}** (`{gt['file_name']}`)")

        with st.expander("Diagnostic CLAHE"):
            clahe_img = apply_clahe(active_image.resize((IMG_SIZE, IMG_SIZE)))
            st.image(clahe_img, use_container_width=True)

    with col_pred:
        # --- STAGE 1: Fruit Classification ---
        t0 = time.time()
        fruit_model = get_cached_fruit_model(mode=mode_key)
        input_tensor = preprocess_image(active_image).to(device)

        with torch.no_grad():
            logits_fruit = fruit_model(input_tensor)
            probs_fruit = F.softmax(logits_fruit, dim=1).cpu().numpy()[0]

        pred_fruit_idx = int(np.argmax(probs_fruit))
        pred_fruit_name = FRUITS_13[pred_fruit_idx]
        pred_fruit_conf = float(probs_fruit[pred_fruit_idx])
        time_stage1 = (time.time() - t0) * 1000

        f_meta = FRUIT_METADATA.get(pred_fruit_name, {'display': pred_fruit_name, 'local_name': ''})

        # Match indicator with Ground Truth
        gt_badge = ""
        if st.session_state.get('ground_truth'):
            gt_fruit = st.session_state['ground_truth']['fruit']
            if pred_fruit_name == gt_fruit:
                gt_badge = '<span class="metric-badge badge-match">MATCH</span>'
            else:
                gt_badge = f'<span class="metric-badge badge-diff">EXP: {gt_fruit.upper()}</span>'

        s1_model_name = "ResNet-50" if mode_key == "best" else "Custom CNN"

        st.markdown(
            f"""
            <div class="metric-row">
                <span class="metric-title">Stage 1: {f_meta['display']} ({f_meta['local_name']}) &mdash; {pred_fruit_conf*100:.1f}%</span>
                {gt_badge}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Model: **{s1_model_name}** | Latency: **{time_stage1:.1f} ms**")

        # Top-5 Horizontal Matplotlib Figure (White Background)
        top5_indices = np.argsort(probs_fruit)[::-1][:5]
        top5_labels = [FRUIT_METADATA[FRUITS_13[i]]['display'] for i in top5_indices]
        top5_values = [probs_fruit[i] * 100 for i in top5_indices]

        fig_fruit = plot_fruit_probabilities(
            labels=top5_labels,
            values=top5_values,
            title="Top-5 Fruit Species Confidence (%)"
        )
        st.pyplot(fig_fruit)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # --- STAGE 2: Quality Inspection ---
        if pred_fruit_conf >= conf_threshold:
            t1 = time.time()
            quality_model, q_model_name = get_cached_quality_model(pred_fruit_name, mode=mode_key)

            with torch.no_grad():
                logits_quality = quality_model(input_tensor)
                probs_quality = F.softmax(logits_quality, dim=1).cpu().numpy()[0]

            pred_q_idx = int(np.argmax(probs_quality))
            pred_q_name = QUALITY_CLASSES[pred_q_idx]
            pred_q_conf = float(probs_quality[pred_q_idx])
            time_stage2 = (time.time() - t1) * 1000

            gt_q_badge = ""
            if st.session_state.get('ground_truth'):
                gt_quality = st.session_state['ground_truth']['quality']
                if pred_q_name == gt_quality:
                    gt_q_badge = '<span class="metric-badge badge-match">MATCH</span>'
                else:
                    gt_q_badge = f'<span class="metric-badge badge-diff">EXP: {gt_quality.upper()}</span>'

            st.markdown(
                f"""
                <div class="metric-row">
                    <span class="metric-title">Stage 2: Quality &mdash; {pred_q_name.upper()} ({pred_q_conf*100:.1f}%)</span>
                    {gt_q_badge}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"Model: **{q_model_name}** | Latency: **{time_stage2:.1f} ms**")

            # Quality Horizontal Matplotlib Figure (White Background)
            q_labels = [q.capitalize() for q in QUALITY_CLASSES]
            q_values = [probs_quality[i] * 100 for i in range(len(QUALITY_CLASSES))]

            fig_quality = plot_quality_probabilities(
                labels=q_labels,
                values=q_values,
                title=f"{pred_fruit_name.replace('_', ' ').title()} Quality Distribution (%)"
            )
            st.pyplot(fig_quality)

        else:
            st.warning(f"Stage 2 Bypassed: Confidence ({pred_fruit_conf*100:.1f}%) is below the {conf_threshold*100:.0f}% threshold.")

else:
    st.info("Select a test image above or upload an image to view predictions.")
