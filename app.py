import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import pandas as pd
import cv2
from PIL import Image

st.set_page_config(
    page_title="RadXAI | Thoracic Triage Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIGH-CONTRAST MILD CLINICAL THEME (FIXES DARK-MODE TEXT VISIBILITY) ---
st.markdown("""
<style>
    /* Main background & top header */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F4F7F9 !important;
        color: #0F172A !important;
    }
    /* Sidebar background & all sidebar text */
    section[data-testid="stSidebar"] {
        background-color: #E6EEF4 !important;
        border-right: 1px solid #CBD5E1;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    /* Main area labels, headings, radio, file uploader label */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {
        color: #0F172A !important;
    }
    /* Keep selectbox dropdown text white on dark input box or make input light */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #94A3B8 !important;
    }
    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }
    /* File uploader box styling */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #0D9488 !important;
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    /* KPI Cards */
    .metric-card {
        background-color: #FFFFFF;
        padding: 16px 20px;
        border-radius: 12px;
        border-left: 5px solid #0D9488;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #475569 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-top: 4px;
    }
    /* Header Banner */
    .banner-header {
        background: linear-gradient(135deg, #D9F2EC 0%, #DBEAFE 100%);
        padding: 20px 24px;
        border-radius: 14px;
        border: 1px solid #93C5FD;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

CLINICAL_NOTES = {
    "Atelectasis": "Partial lung collapse; inspect lower lobes for volume loss or linear opacities.",
    "Cardiomegaly": "Enlarged cardiac silhouette; verify cardiothoracic ratio (>0.50 on PA view).",
    "Consolidation": "Alveolar airspace filling; correlate with infectious or inflammatory infiltrates.",
    "Edema": "Pulmonary venous congestion / Kerley B lines; check perihilar bat-wing distribution.",
    "Effusion": "Pleural fluid accumulation; inspect costophrenic angles for blunting or meniscus.",
    "Emphysema": "Hyperinflated lung fields with flattened diaphragms and reduced vascular markings.",
    "Fibrosis": "Reticular interstitial scarring; inspect peripheral and basilar lung zones.",
    "Hernia": "Retrocardiac / diaphragmatic hiatus shadow; verify against gastric air bubble overlap.",
    "Infiltration": "Diffuse or patchy parenchymal opacity; check multi-label co-occurrence.",
    "Mass": "Focal pulmonary lesion (>3 cm); correlate with cross-sectional CT imaging.",
    "Nodule": "Discrete rounded opacity (<=3 cm); inspect upper and mid lung zones carefully.",
    "Pleural_Thickening": "Localized or diffuse pleural scarring along lateral chest wall or apices.",
    "Pneumonia": "Focal or lobar parenchymal consolidation; correlate with clinical fever/WBC.",
    "Pneumothorax": "Visceral pleural line with absent peripheral lung markings; inspect apices."
}

DEFAULT_THRESHOLDS = {p: 0.50 for p in PATHOLOGIES}
DEFAULT_THRESHOLDS["Hernia"] = 0.20

@st.cache_resource
def load_model_and_thresholds():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier.in_features, 14)
    )
    model.load_state_dict(torch.load("best_densenet121.pth", map_location=device))
    model.to(device).eval()

    thresholds = DEFAULT_THRESHOLDS.copy()
    try:
        metrics_df = pd.read_csv("test_metrics.csv")
        for _, row in metrics_df.iterrows():
            thresholds[row["Pathology"]] = float(row["Threshold"])
    except Exception:
        pass
    return model, thresholds, device

class DenseNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        features = self.model.features(input_tensor)
        out = F.relu(features, inplace=False)
        self.activations = out.detach()
        out.register_hook(self._save_gradient)

        pooled = F.adaptive_avg_pool2d(out, (1, 1))
        logits = self.model.classifier(torch.flatten(pooled, 1))
        prob = torch.sigmoid(logits[0, class_idx]).item()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1).squeeze(0)).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cv2.resize(cam, (224, 224)), prob

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model, calibrated_thresholds, device = load_model_and_thresholds()
gradcam = DenseNetGradCAM(model)

# --- HEADER BANNER ---
st.markdown("""
<div class="banner-header">
    <h2 style="margin:0; color:#0F172A !important;">🩺 RadXAI — Thoracic Pathology Triage & Visual Audit</h2>
    <p style="margin:6px 0 0 0; color:#1E293B !important; font-size:0.98rem;">
        Multi-Label DenseNet121 Screening • Youden's J Threshold Calibration • Real-Time Grad-CAM Localization
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Diagnostic Controls")
threshold_mode = st.sidebar.radio(
    "Decision Threshold Mode",
    ["Youden's J Calibrated (Recommended)", "Standard 0.50 Cutoff", "Custom Global Cutoff"]
)

custom_cutoff = 0.50
if threshold_mode == "Custom Global Cutoff":
    custom_cutoff = st.sidebar.slider("Global Probability Cutoff", 0.05, 0.95, 0.40, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Heatmap Visualization")
colormap_choice = st.sidebar.selectbox("Colormap Palette", ["JET (Classic)", "TURBO (High Contrast)", "VIRIDIS (Colorblind Safe)"])
heatmap_opacity = st.sidebar.slider("Overlay Blend Opacity", 0.10, 0.80, 0.40, 0.05)

cmap_lookup = {
    "JET (Classic)": cv2.COLORMAP_JET,
    "TURBO (High Contrast)": cv2.COLORMAP_TURBO,
    "VIRIDIS (Colorblind Safe)": cv2.COLORMAP_VIRIDIS
}

# --- FILE UPLOAD ---
uploaded = st.file_uploader("Upload Frontal Chest Radiograph (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    raw_resized = np.array(pil_img.resize((224, 224)))
    input_tensor = eval_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]

    rows = []
    for i, p_name in enumerate(PATHOLOGIES):
        if threshold_mode == "Youden's J Calibrated (Recommended)":
            thresh = calibrated_thresholds[p_name]
        elif threshold_mode == "Standard 0.50 Cutoff":
            thresh = 0.50
        else:
            thresh = custom_cutoff

        rows.append({
            "Pathology": p_name,
            "Probability": round(float(probs[i]), 4),
            "Threshold": round(float(thresh), 4),
            "Margin": round(float(probs[i] - thresh), 4),
            "Status": "POSITIVE" if probs[i] >= thresh else "Negative"
        })

    res_df = pd.DataFrame(rows).sort_values(by="Probability", ascending=False).reset_index(drop=True)
    positives = res_df[res_df["Status"] == "POSITIVE"]
    top_pred = res_df.iloc[0]

    # --- KPI SUMMARY CARDS ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Flagged Findings</div><div class='metric-value'>{len(positives)} / 14</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Top Suspect Pathology</div><div class='metric-value'>{top_pred['Pathology']}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Top Confidence</div><div class='metric-value'>{top_pred['Probability']*100:.1f}%</div></div>", unsafe_allow_html=True)
    with k4:
        triage_priority = "HIGH PRIORITY" if len(positives) >= 2 else ("REVIEW" if len(positives) == 1 else "ROUTINE")
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Triage Status</div><div class='metric-value'>{triage_priority}</div></div>", unsafe_allow_html=True)

    # --- TABS FOR CLEAN WORKFLOW ---
    tab1, tab2, tab3 = st.tabs(["🩻 Grad-CAM Visual Inspection", "📊 Multi-Label Risk Chart", "📋 Full Triage Table & Export"])

    with tab1:
        sel_col, note_col = st.columns([1, 2])
        with sel_col:
            selected_pathology = st.selectbox(
                "Select Pathology to Localize",
                options=res_df["Pathology"].tolist(),
                index=0,
                format_func=lambda x: f"{x} ({res_df.loc[res_df['Pathology']==x, 'Probability'].values[0]*100:.1f}%)"
            )
        with note_col:
            st.info(f"**Clinical Guidance ({selected_pathology}):** {CLINICAL_NOTES[selected_pathology]}")

        target_idx = PATHOLOGIES.index(selected_pathology)
        heatmap, pred_prob = gradcam.generate(input_tensor, target_idx)
        heatmap_color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cmap_lookup[colormap_choice]), cv2.COLOR_BGR2RGB)
        overlay = np.uint8((1.0 - heatmap_opacity) * raw_resized + heatmap_opacity * heatmap_color)

        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(raw_resized, width="stretch", caption="Original Standardized Radiograph (224x224)")
        with v2:
            st.image(heatmap_color, width="stretch", caption=f"Raw Grad-CAM Attention Map ({selected_pathology})")
        with v3:
            st.image(overlay, width="stretch", caption=f"Anatomical Overlay | Score: {pred_prob:.3f}")

    with tab2:
        st.subheader("Predicted Probability Distribution Across All 14 Pathologies")
        chart_df = res_df.set_index("Pathology")[["Probability", "Threshold"]]
        st.bar_chart(chart_df, height=430)

    with tab3:
        st.subheader("Detailed Multi-Label Triage Report")
        st.dataframe(
            res_df.style.format({"Probability": "{:.4f}", "Threshold": "{:.4f}", "Margin": "{:+.4f}"}),
            width="stretch",
            height=450
        )
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Patient Triage Report (CSV)",
            data=csv_bytes,
            file_name=f"triage_report_{uploaded.name.split('.')[0]}.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Upload any frontal chest X-ray (PNG/JPG) above to launch interactive Grad-CAM localization and multi-label triage.")
