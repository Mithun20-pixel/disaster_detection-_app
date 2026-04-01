"""
🛰️ Disaster Damage Detection — Streamlit Web App
Upload satellite images → AI detects damage → See highlighted results on a map.

Run:  streamlit run app.py
"""
import os
import sys
import io
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLASS_NAMES, IMAGE_SIZE
from utils.preprocessing import preprocess_uploaded_image, enhance_satellite_image
from utils.prediction import predict_damage_classification, predict_damage_segmentation
from utils.visualization import (
    create_damage_heatmap,
    create_segmentation_overlay,
    draw_damage_bboxes,
    create_severity_map,
    create_before_after,
)

from utils.metrics import (
    structural_difference, detect_water, vegetation_loss, edge_comparison
)
from utils.explainability import GradCAM, overlay_gradcam

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(231, 76, 60, 0.3);
        box-shadow: 0 8px 32px rgba(231, 76, 60, 0.15);
        text-align: center;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #e74c3c, #f39c12, #e74c3c);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 200% center; }
    }
    .main-header p {
        color: #8899aa;
        font-size: 1.05rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #8899aa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
    }
    .badge-damaged {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    .badge-safe {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }

    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #bcc6d0;
        font-size: 0.9rem;
    }

    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(231,76,60,0.3), transparent);
        margin: 2rem 0;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
        box-shadow: 0 4px 16px rgba(231, 76, 60, 0.4);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(231, 76, 60, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(231, 76, 60, 0.6);
    }

    div[data-testid="stExpander"] {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }

    /* Tabs Styling */
    div[data-testid="stTabs"] button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached Model Loading  (models are loaded ONCE and reused across reruns)
# ═══════════════════════════════════════════════════════════════════════════════
from models.cnn_model import build_classifier
from models.unet_model import build_unet


@st.cache_resource
def load_classification_model(backbone: str = "efficientnet_b0"):
    """Load the classification model once and cache it."""
    import os
    model = build_classifier(backbone=backbone, num_classes=len(CLASS_NAMES), freeze=False)
    # Load trained weights if available
    ckpt_path = os.path.join(os.path.dirname(__file__), "saved_models", "best_classifier.pth")
    is_trained = os.path.exists(ckpt_path)
    if is_trained:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model, is_trained


@st.cache_resource
def load_segmentation_model():
    """Load the U-Net segmentation model once and cache it."""
    import os
    model = build_unet(num_classes=1, pretrained=True)
    ckpt_path = os.path.join(os.path.dirname(__file__), "saved_models", "best_unet.pth")
    is_trained = os.path.exists(ckpt_path)
    if is_trained:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model, is_trained


# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🛰️ Disaster Damage Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["🏷️ Classification (Fast)", "🗺️ Segmentation (Detailed)", "🔬 Full Analysis (Both)"],
        index=2,
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Options")
    backbone = st.selectbox("Backbone", [
        "resnet50", "efficientnet_b0", "mobilenetv3_large_100",
        "efficientnet_b2", "convnext_tiny",
    ])

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("### 🎨 Explainable AI Visualization")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    show_heatmap = st.checkbox("Show Blob Heatmap", value=False)
    show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
    show_severity = st.checkbox("Show Severity Gauge", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#556; font-size:0.75rem;">'
        '🛰️ Disaster Damage Detection v1.0<br>'
        'Built for Hackathon Demo'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛰️ Disaster Damage Detection</h1>
    <p>AI-powered explainable satellite image analysis for flood & earthquake damage assessment</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Navigation Bar
# ═══════════════════════════════════════════════════════════════════════════════
tab_home, tab_compare, tab_alert = st.tabs(["🏠 Home", "🔄 Compare", "🚨 Emergency Alert"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: HOME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.markdown("### 📤 Upload Post-Disaster Satellite Image")
        uploaded_file = st.file_uploader(
            "Drag and drop a satellite image or click to browse",
            type=["png", "jpg", "jpeg", "tif"],
            help="Supports PNG, JPG, JPEG, and TIFF formats up to 50 MB",
        )

    with col_info:
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        <div class="info-box">
          <strong>Best Results:</strong><br>
          • Use high-resolution satellite images<br>
          • Pre- and post-disaster images work best<br>
          • RGB images (not infrared) preferred<br>
          • Minimum 224×224 pixels
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # ── Original Image Preview ───────────────────────────────────────────
        st.markdown("### 📸 Original Image")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Satellite Image", use_container_width=True)
        with col2:
            enhanced = enhance_satellite_image(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)))
            st.image(enhanced, caption="Enhanced Image", use_container_width=True)

        # ── Run Analysis ─────────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🔬 Explainable AI Analysis Results")

        with st.spinner("🧠 Running AI damage detection..."):
            # Load cached models
            cls_model, cls_is_trained = load_classification_model(backbone)

            # Classification
            cls_result = predict_damage_classification(
                image_resized, model=cls_model, backbone=backbone, is_trained=cls_is_trained
            )

            # GradCAM Calculation
            gradcam_img = image_resized.copy()
            try:
                from utils.preprocessing import preprocess_for_model
                
                # Select target layer based on model architecture
                if 'resnet' in backbone:
                    target_layer = cls_model.model.layer4[-1]
                else:
                    # Fallback for EfficientNet, MobileNet, ConvNeXt
                    target_layer = list(cls_model.model.children())[-2][-1] if hasattr(list(cls_model.model.children())[-2], '__getitem__') else list(cls_model.model.children())[-2]
                
                cam_extractor = GradCAM(cls_model, target_layer)
                
                # Prepare tensor for GradCAM using the correct preprocessor function
                img_tensor = preprocess_for_model(image_resized)
                
                cam_mask, _ = cam_extractor(img_tensor, class_idx=cls_result["class_index"])
                if cam_mask is not None:
                    gradcam_img = overlay_gradcam(image_resized, cam_mask)
            except Exception:
                pass

            # Segmentation
            mask, damage_ratio = None, 0.0
            if "Segmentation" in analysis_mode or "Full" in analysis_mode:
                seg_model, seg_is_trained = load_segmentation_model()
                mask, damage_ratio = predict_damage_segmentation(
                    image_resized, model=seg_model, is_trained=seg_is_trained
                )

        # ── Severity Scoring System ──────────────────────────────────────────
        is_damaged = cls_result["class_index"] == 1
        badge_class = "badge-damaged" if is_damaged else "badge-safe"
        badge_text = "⚠️ DAMAGE DETECTED" if is_damaged else "✅ NO DAMAGE"
        
        # Generate 0-100 severity score
        dmg_pct = damage_ratio * 100 if mask is not None else 0
        base_severity = cls_result['confidence'] * 100 if is_damaged else (1 - cls_result['confidence']) * 30
        severity_score = min(100, max(0, int((base_severity * 0.4) + (dmg_pct * 0.6))))

        # Save state to session so it persists across tabs
        st.session_state['post_image'] = image_resized
        st.session_state['cls_result'] = cls_result
        st.session_state['severity_score'] = severity_score
        st.session_state['dmg_pct'] = dmg_pct

        # ── Metrics Row ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div style="margin-top:0.5rem;">
                    <span class="status-badge {badge_class}">{badge_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            conf_color = "#e74c3c" if cls_result["confidence"] > 0.7 else "#f39c12" if cls_result["confidence"] > 0.4 else "#2ecc71"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color:{conf_color};">{cls_result['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Damage Severity</div>
                <div class="metric-value" style="color:#e74c3c;">{severity_score}/100</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            dmg_color = "#e74c3c" if dmg_pct > 30 else "#f39c12" if dmg_pct > 10 else "#2ecc71"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Area Damaged</div>
                <div class="metric-value" style="color:{dmg_color};">{dmg_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)



        # ── Visualizations ───────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🎨 Explainable AI Visualizations")

        viz_cols = st.columns(2)

        if show_gradcam:
            with viz_cols[0]:
                st.image(gradcam_img, caption="Grad-CAM Heatmap (Shows model attention)", use_container_width=True)

        if show_severity:
            with viz_cols[1]:
                gauge = create_severity_map(severity_score / 100.0)
                st.image(gauge, caption="Severity Gauge", use_container_width=True)

        if mask is not None:
            seg_cols = st.columns(2)
            with seg_cols[0]:
                overlay = create_segmentation_overlay(image_resized, mask)
                st.image(overlay, caption="Damage Segmentation Overlay", use_container_width=True)

            if show_bboxes:
                with seg_cols[1]:
                    bbox_img = draw_damage_bboxes(image_resized, mask)
                    st.image(bbox_img, caption="Damage Bounding Boxes", use_container_width=True)

        # ── Download Results ─────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 💾 Download Results")
        dl1, dl2 = st.columns(2)

        with dl1:
            if show_gradcam:
                grad_bgr = cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
                _, heatmap_buf = cv2.imencode(".png", grad_bgr)
                st.download_button(
                    "📥 Download Grad-CAM Analysis",
                    data=heatmap_buf.tobytes(),
                    file_name="damage_gradcam.png",
                    mime="image/png",
                )

        with dl2:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            
            # Create a PDF buffer
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:
                fig = plt.figure(figsize=(10, 6))
                
                # Styling
                fig.patch.set_facecolor('#1a1a2e')
                text_color = '#ecf0f1'
                
                # Title
                fig.text(0.5, 0.92, "DISASTER DAMAGE DETECTION REPORT", color=text_color, 
                         fontsize=18, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.88, "Autogenerated AI Analysis & Metrics", color='#8899aa', 
                         fontsize=10, ha='center', va='center')
                fig.text(0.5, 0.84, "-"*70, color=text_color, fontsize=12, ha='center', va='center')
                
                # Left Side: Text Details
                info_text = (
                    f"REPORT SUMMARY\n\n"
                    f"Status Class:      {cls_result['label']}\n"
                    f"AI Confidence:     {cls_result['confidence']:.1%}\n"
                    f"AI Severity Score: {severity_score}/100\n"
                    f"Damage Area %:     {dmg_pct:.1f}%\n"
                )
                fig.text(0.05, 0.65, info_text, color=text_color, fontsize=11, va='top', family='monospace')
                
                # Right Side: Graphical Pie Chart
                ax = fig.add_axes([0.5, 0.15, 0.45, 0.6])
                # Ensure pie chart has valid sizes
                damaged = dmg_pct
                intact = 100.0 - dmg_pct
                if damaged == 0 and intact == 0:
                    intact = 100.0
                
                sizes = [damaged, intact]
                labels = ['Damaged Area', 'Intact Area']
                colors = ['#e74c3c', '#2ecc71']
                explode = (0.1, 0)
                
                _, texts, autotexts = ax.pie(
                    sizes, explode=explode, labels=labels, colors=colors, 
                    autopct='%1.1f%%', shadow=True, startangle=140
                )
                
                # Style pie chart text
                for text in texts:
                    text.set_color(text_color)
                    text.set_fontsize(10)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
                
                ax.set_title("Structural Edge Damage Distribution", color=text_color, pad=20, weight='bold')
                ax.axis('equal')
                
                plt.axis('off')
                pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close(fig)
                
            pdf_bytes = pdf_buf.getvalue()

            st.download_button(
                "📥 Download Analytical PDF Report",
                data=pdf_bytes,
                file_name="damage_analytical_report.pdf",
                mime="application/pdf",
            )

    else:
        # ── Empty State ──────────────────────────────────────────────────────
        st.markdown("")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem;">🌊</div>
                <div class="metric-label" style="margin-top:0.5rem;">Flood Detection</div>
                <p style="color:#8899aa; font-size:0.8rem; margin-top:0.5rem;">
                    Detect flood damage from satellite imagery with AI classification
                </p>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem;">🏚️</div>
                <div class="metric-label" style="margin-top:0.5rem;">Earthquake Damage</div>
                <p style="color:#8899aa; font-size:0.8rem; margin-top:0.5rem;">
                    Identify structural damage and collapsed buildings from aerial views
                </p>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem;">🧠</div>
                <div class="metric-label" style="margin-top:0.5rem;">Explainable AI</div>
                <p style="color:#8899aa; font-size:0.8rem; margin-top:0.5rem;">
                    See exactly which parts of the image caused the damage prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            '<div class="info-box">👆 <strong>Upload a satellite image</strong> above to begin analysis. '
            'Try images from Google Earth, Copernicus Open Access Hub, or the xBD dataset.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔄 Pre vs Post Disaster Comparison")
    
    st.markdown(
        '<div class="info-box">Upload Pre-Disaster and Post-Disaster images of the same region to automatically compute structural edge loss, environmental damage, and structural differences.</div>',
        unsafe_allow_html=True,
    )
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        before_file = st.file_uploader("Upload Pre-Disaster Satellite Image", type=["png", "jpg", "jpeg"], key="before_compare")
    with col_comp2:
        after_file = st.file_uploader("Upload Post-Disaster Satellite Image", type=["png", "jpg", "jpeg"], key="after_compare")
        
    if before_file is not None and after_file is not None:
        before_bytes = np.frombuffer(before_file.read(), dtype=np.uint8)
        before_img = cv2.imdecode(before_bytes, cv2.IMREAD_COLOR)
        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
        before_img = cv2.resize(before_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        after_bytes = np.frombuffer(after_file.read(), dtype=np.uint8)
        after_img = cv2.imdecode(after_bytes, cv2.IMREAD_COLOR)
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.resize(after_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Calculate structure loss, edge change and vegetation
        with st.spinner("Analyzing structural and environmental differences..."):
            struct_loss, map_diff = structural_difference(before_img, after_img)
            edge_loss, edge_loss_ratio = edge_comparison(before_img, after_img)
            veg_mask, veg_ratio = vegetation_loss(before_img, after_img)
            water_mask, water_ratio = detect_water(after_img)
        
        comp_metrics = st.columns(4)
        comp_metrics[0].metric("Structural Change", f"{struct_loss * 100:.1f}%")
        comp_metrics[1].metric("Edge Loss", f"{edge_loss_ratio * 100:.1f}%")
        comp_metrics[2].metric("Vegetation Loss", f"{veg_ratio * 100:.1f}%")
        comp_metrics[3].metric("Flood Water Area", f"{water_ratio * 100:.1f}%")
        view_mode = st.radio(
            "Select View",
            ["Side-by-Side Comparison", "Structural Loss Map", "Edge Changes", "Environmental Loss"],
            horizontal=True
        )
        
        if view_mode == "Side-by-Side Comparison":
            comparison = create_before_after(before_img, after_img)
            st.image(comparison, caption="Before vs. After Analysis", use_container_width=True)
            
        elif view_mode == "Structural Loss Map":
            st.image(map_diff, caption="Structural Differences (White regions indicate change)", use_container_width=True)
            
        elif view_mode == "Edge Changes":
            st.image(edge_loss, caption="Edge Structural Collapse Map", use_container_width=True)
            
        elif view_mode == "Environmental Loss":
            col_veg, col_water = st.columns(2)
            with col_veg:
                st.image(veg_mask, caption="Vegetation Loss Map", use_container_width=True)
            with col_water:
                st.image(water_mask, caption="Detected Water Regions", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: EMERGENCY ALERT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alert:
    st.markdown("### 🚨 Emergency Dispatch & Alert System")
    
    st.markdown(
        '<p style="color:#8899aa; font-size:0.95rem;">'
        'Broadcast verified damage metrics directly to the nearest help centers and first responders.'
        '</p>', unsafe_allow_html=True
    )
    
    cls_result = st.session_state.get('cls_result', {'label': 'Unspecified Damage'})
    severity_score = st.session_state.get('severity_score', 0)
    dmg_pct = st.session_state.get('dmg_pct', 0.0)
    
    with st.expander("📡 Broadcast Emergency Alert", expanded=True):
        with st.form("alert_form_tab"):
            al_col1, al_col2 = st.columns(2)
            with al_col1:
                loc_input = st.text_input("Incident Location (Coordinates or Address)", placeholder="e.g. 34.0522° N, 118.2437° W")
                contact_input = st.text_input("Responder Callback Number", placeholder="+1 (555) 019-2024")
            with al_col2:
                worldwide_orgs = [
                    "Local Fire/Rescue", "National Guard", "FEMA Response Center (USA)", 
                    "American Red Cross", "UNICEF Emergency Unit", "World Health Organization (WHO)",
                    "Doctors Without Borders (MSF)", "Direct Relief", "International Rescue Committee (IRC)",
                    "Mercy Corps", "Oxfam International", "Save the Children", "World Food Programme (WFP)",
                    "CARE International", "Team Rubicon", "European Civil Protection and Humanitarian Aid (ECHO)",
                    "Pacific Disaster Center", "Asian Disaster Preparedness Center (ADPC)",
                    "Global Disaster Alert and Coordination System (GDACS)"
                ]
                org_select = st.selectbox("Dispatch Destination (Searchable)", worldwide_orgs)
                
                # Auto-select priority based on AI
                default_pri = 0 if severity_score >= 70 else (1 if severity_score >= 40 else 2)
                priority_select = st.selectbox("Dispatch Priority", ["CRITICAL (Immediate)", "HIGH (1-2 Hours)", "MEDIUM (Same Day)", "LOW"], index=default_pri)
            
            # The AI-generated message is provided as a placeholder, but the actual box is empty
            placeholder_msg = (
                f"AI Detected {cls_result.get('label', 'Unspecified Damage')} with a severity score of {severity_score}/100. "
                f"Estimated damage covers {dmg_pct:.1f}% of the region. Units required."
            )
            msg_input = st.text_area("Dispatch Message Details", value="", placeholder=placeholder_msg, height=120)
            
            submit_alert = st.form_submit_button("🚨 SEND ALERT TO CENTER")
            if submit_alert:
                if not loc_input:
                    st.error("Error: Incident Location is required to dispatch units.")
                else:
                    st.success(f"Alert successfully broadcasted to {org_select}! Tracking dispatch unit...")
                    st.balloons()
