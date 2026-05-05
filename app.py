import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import io
from datetime import datetime

# ─── PDF REPORT GENERATOR ────────────────────────────────────────────────────
def generate_pdf_report(
    district, season, soil_ph, nitrogen, phosphorus, potassium,
    rainfall, temperature, best_crop, confidence,
    top3_crops, top3_probs, feature_names, importances,
    adv_title, adv_body
):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Crop Recommendation Report"
    )

    # ── Colors ──
    GREEN_DARK  = colors.HexColor("#1b5e20")
    GREEN_MID   = colors.HexColor("#2e7d32")
    GREEN_LIGHT = colors.HexColor("#e8f5e9")
    AMBER       = colors.HexColor("#f9a825")
    AMBER_PALE  = colors.HexColor("#fffde7")
    GRAY        = colors.HexColor("#546e7a")
    GRAY_LIGHT  = colors.HexColor("#eceff1")
    WHITE       = colors.white
    BLACK       = colors.HexColor("#1a1a1a")

    # ── Styles ──
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    s_title = S("title",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    s_subtitle = S("subtitle",
        fontSize=11, fontName="Helvetica",
        textColor=colors.HexColor("#2e7b34"), alignment=TA_CENTER, spaceAfter=4)
    s_section = S("section",
        fontSize=12, fontName="Helvetica-Bold",
        textColor=GREEN_DARK, spaceBefore=12, spaceAfter=6)
    s_adv_title = S("adv_title",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#7a5c00"), spaceAfter=4)
    s_adv_body = S("adv_body",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#5a4500"), leading=15)

    story = []

    # ══════════════════════════════════════════════════════════════
    # HEADER BANNER
    # ══════════════════════════════════════════════════════════════
    header_data = [[
        Paragraph("🌱 Smart Crop Advisor", s_title),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), GREEN_MID),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # Sub-header row: university info + date
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    sub_data = [[
        Paragraph("B.Tech Final Year Project · CSE Core · SDP Final Review", s_subtitle),
        Paragraph(f"Generated: {now}", S("dt",
            fontSize=8.5, fontName="Helvetica",
            textColor=GRAY, alignment=TA_RIGHT)),
    ]]
    sub_tbl = Table(sub_data, colWidths=[12*cm, 5*cm])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f5f7f2")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 14))

    # ══════════════════════════════════════════════════════════════
    # RECOMMENDATION RESULT CARD
    # ══════════════════════════════════════════════════════════════
    crop_icons = {"Rice":"🌾","Cotton":"🌿","Chilli":"🌶","Maize":"🌽","Groundnut":"🥜"}
    icon = crop_icons.get(best_crop, "🌱")

    conf_label = "High Confidence" if confidence >= 80 else ("Moderate Confidence" if confidence >= 60 else "Low Confidence")

    result_data = [
        [Paragraph(f"{icon}  Recommended Crop", S("rl",
            fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#a5d6a7"))),
         Paragraph("Confidence Score", S("rc",
            fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#a5d6a7"), alignment=TA_RIGHT))],
        [Paragraph(best_crop, S("rn",
            fontSize=26, fontName="Helvetica-Bold",
            textColor=WHITE)),
         Paragraph(f"{confidence:.1f}%", S("rpct",
            fontSize=26, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_RIGHT))],
        [Paragraph(f"{district} · {season} season", S("rs",
            fontSize=9, fontName="Helvetica",
            textColor=colors.HexColor("#c8e6c9"))),
         Paragraph(conf_label, S("rcl",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_RIGHT))],
    ]
    result_table = Table(result_data, colWidths=[10*cm, 7*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GREEN_DARK),
        ("ROUNDEDCORNERS",[10]),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 18),
        ("RIGHTPADDING",  (0,0), (-1,-1), 18),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # INPUT PARAMETERS TABLE
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Field Input Parameters", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    params = [
        ["Parameter", "Value", "Unit", "Optimal Range"],
        ["District",    district,           "—",      "Andhra Pradesh"],
        ["Season",      season,             "—",      "Kharif / Rabi / Zaid"],
        ["Soil pH",     f"{soil_ph:.1f}",   "pH",     "6.0 – 7.5"],
        ["Nitrogen",    str(nitrogen),      "kg/ha",  "40 – 80"],
        ["Phosphorus",  str(phosphorus),    "kg/ha",  "20 – 60"],
        ["Potassium",   str(potassium),     "kg/ha",  "30 – 70"],
        ["Rainfall",    str(rainfall),      "mm",     "500 – 900"],
        ["Temperature", f"{temperature:.1f}","°C",   "20 – 35"],
    ]
    param_table = Table(params, colWidths=[4.5*cm, 3.5*cm, 2.5*cm, 6.5*cm])
    param_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), GREEN_DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), BLACK),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # TOP 3 RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Top 3 Crop Recommendations", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    medals = ["🥇", "🥈", "🥉"]
    rec_rows = [["Rank", "Crop", "Probability", "Confidence Bar"]]
    for i in range(3):
        bar_pct = int(top3_probs[i] * 100)
        bar = "█" * (bar_pct // 5) + "░" * (20 - bar_pct // 5)
        rec_rows.append([
            f"{medals[i]} #{i+1}",
            top3_crops[i],
            f"{bar_pct}%",
            bar[:20]
        ])

    rec_table = Table(rec_rows, colWidths=[2.5*cm, 4.5*cm, 3*cm, 7*cm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), GREEN_MID),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("BACKGROUND",    (0,1), (-1,1), GREEN_LIGHT),
        ("TEXTCOLOR",     (0,1), (-1,1), GREEN_DARK),
        ("FONTNAME",      (0,1), (-1,1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,2), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("TEXTCOLOR",     (0,1), (-1,-1), BLACK),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("FONTNAME",      (3,1), (3,-1), "Courier"),
        ("FONTSIZE",      (3,1), (3,-1), 7.5),
        ("TEXTCOLOR",     (3,1), (3,1), GREEN_DARK),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE TABLE
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Model Feature Importance Analysis", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    label_map = {
        "Soil_pH":"Soil pH","Nitrogen":"Nitrogen","Phosphorus":"Phosphorus",
        "Potassium":"Potassium","Rainfall":"Rainfall","Temperature":"Temperature",
        "District":"District","Season":"Season"
    }
    feat_data = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    imp_rows = [["Feature", "Importance Score", "Visual", "Interpretation"]]
    interpretations = {
        "Rainfall":    "Primary water availability indicator",
        "Temperature": "Crop thermal requirement factor",
        "Soil_pH":     "Soil acidity/alkalinity suitability",
        "Nitrogen":    "Key macronutrient for leaf growth",
        "Phosphorus":  "Root development & energy transfer",
        "Potassium":   "Disease resistance & water regulation",
        "District":    "Region-specific agro-climatic zone",
        "Season":      "Seasonal sowing window suitability",
    }
    for feat, imp in feat_data:
        bar = "█" * int(imp * 100 // 5) + "░" * (20 - int(imp * 100 // 5))
        imp_rows.append([
            label_map.get(feat, feat),
            f"{imp*100:.2f}%",
            bar[:20],
            interpretations.get(feat, "—"),
        ])
    imp_table = Table(imp_rows, colWidths=[3.5*cm, 3*cm, 4*cm, 6.5*cm])
    imp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#37474f")),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), BLACK),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("FONTNAME",      (2,1), (2,-1), "Courier"),
        ("FONTSIZE",      (2,1), (2,-1), 7),
    ]))
    story.append(imp_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # AGRONOMIC ADVISORY
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Agronomic Advisory", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    adv_data = [[
        Paragraph(f"💡  {adv_title}", s_adv_title),
    ],[
        Paragraph(adv_body, s_adv_body),
    ]]
    adv_table = Table(adv_data, colWidths=[17*cm])
    adv_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), AMBER_PALE),
        ("ROUNDEDCORNERS",[8]),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ("TOPPADDING",    (0,0), (0,0),   10),
        ("BOTTOMPADDING", (0,-1),(-1,-1), 12),
        ("TOPPADDING",    (0,1), (-1,-1), 4),
        ("LINEABOVE",     (0,0), (-1,0),  2, AMBER),
    ]))
    story.append(adv_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # MODEL INFO BOX
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Model & System Information", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    model_params = [
        ["Algorithm",      "XGBoost Classifier (XGBClassifier)"],
        ["Estimators",     "300 trees"],
        ["Learning Rate",  "0.05 (slow, high accuracy)"],
        ["Max Depth",      "5 levels per tree"],
        ["Subsampling",    "80% row / 80% column per tree"],
        ["Imbalance Fix",  "SMOTE over-sampling (random_state=42)"],
        ["Eval Metric",    "mlogloss (multi-class log loss)"],
        ["Dataset",        "ap_crop_dataset_for_faculty.csv"],
        ["Features",       ", ".join([label_map.get(f,f) for f in feature_names])],
    ]
    info_table = Table(model_params, colWidths=[4.5*cm, 12.5*cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TEXTCOLOR",     (0,0), (0,-1), GREEN_DARK),
        ("TEXTCOLOR",     (1,0), (1,-1), BLACK),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))

    # ══════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════
    footer_data = [[
        Paragraph(
            "B.Tech Final Year Project — Smart Crop Recommendation System  |  "
            "CSE Core  |  SDP Final Review  |  Powered by XGBoost + SMOTE + Streamlit",
            S("ft", fontSize=8, fontName="Helvetica", textColor=WHITE, alignment=TA_CENTER)
        )
    ]]
    footer_table = Table(footer_data, colWidths=[17*cm])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GREEN_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
    ]))
    story.append(footer_table)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Advisor · AP",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Full-page background ── */
.stApp {
    background-image:
        linear-gradient(rgba(10,30,10,0.70), rgba(10,30,10,0.80)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1800&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ── Typography on dark bg ── */
h1, h2, h3, h4, h5 { color: #ffffff !important; }
label, p, .stMarkdown p { color: #d9efd2 !important; }
.stSlider label { color: #d9efd2 !important; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(6px);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem !important;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #81c784 !important;
    margin-bottom: 0.8rem;
}

/* ── Result banner ── */
.result-banner {
    background: linear-gradient(135deg, #1b5e20, #388e3c);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1rem 0;
    border: 1px solid #4caf50;
}
.crop-big { font-size: 2.2rem; font-weight: 800; color: #fff; }
.crop-sub { font-size: 0.78rem; color: #a5d6a7; text-transform: uppercase; letter-spacing: 0.08em; }
.conf-pct { font-size: 2.6rem; font-weight: 800; color: #fff; text-align: right; }
.conf-lbl { font-size: 0.78rem; color: #a5d6a7; text-align: right; }

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #d9efd2;
    margin: 3px 4px 3px 0;
}

/* ── Advisory ── */
.advisory {
    background: rgba(249,168,37,0.15);
    border: 1px solid rgba(249,168,37,0.4);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}
.advisory .adv-title { font-weight: 700; color: #ffe082 !important; font-size: 0.9rem; }
.advisory .adv-body  { color: #fff9c4 !important; font-size: 0.85rem; line-height: 1.6; }

/* ── Badges ── */
.badge-h { background:#1b5e20; color:#a5d6a7; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #4caf50; }
.badge-m { background:#0d3a75; color:#90caf9; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #42a5f5; }
.badge-l { background:#7a4a00; color:#ffe082; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #ffa000; }

/* ── Project badge ── */
.project-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.72rem;
    color: #a5d6a7 !important;
    display: inline-block;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Section divider ── */
.sdiv { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 8px !important; color: #fff !important; }
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #43a047) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover { background: linear-gradient(135deg, #1b5e20, #2e7d32) !important; }
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover { background: linear-gradient(135deg, #0d47a1, #1565c0) !important; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on AP crop dataset…")
def load_model():
    data = pd.read_csv("ap_crop_dataset_for_faculty.csv")
    le_d = LabelEncoder(); le_s = LabelEncoder(); le_c = LabelEncoder()
    data["District"] = le_d.fit_transform(data["District"])
    data["Season"]   = le_s.fit_transform(data["Season"])
    data["Crop"]     = le_c.fit_transform(data["Crop"])
    X, y = data.drop("Crop", axis=1), data["Crop"]
    X_r, y_r = SMOTE(random_state=42, k_neighbors=min(5, min(pd.Series(y).value_counts()) - 1)).fit_resample(X, y)
    mdl = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss", n_jobs=-1,
    )
    mdl.fit(X_r, y_r)
    return mdl, le_d, le_s, le_c, list(X.columns)

model, le_district, le_season, le_crop, feature_names = load_model()

# ─── CONTENT ─────────────────────────────────────────────────────────────────
ADVISORIES = {
    "Rice":      ("Water management is critical",
                  "Maintain 5–10 cm standing water during tillering. Apply split-dose nitrogen — 50% basal, 25% at tillering, 25% at panicle initiation. Target pH 5.5–7.0. Transplanting season should align with onset of monsoon."),
    "Cotton":    ("Balanced NPK with pest vigilance",
                  "Apply balanced NPK with micronutrients. Use drip irrigation for water efficiency. Scout regularly for bollworm and whitefly. Ideal soil pH 6.0–8.0. Avoid waterlogging during boll formation."),
    "Chilli":    ("High phosphorus + drip irrigation",
                  "Boost phosphorus at transplanting for strong root development. Drip irrigation reduces fungal risk from wet foliage. Apply mulch to retain moisture and suppress weeds. Target pH 6.0–7.0. Harvest at red-ripe stage for best market price."),
    "Maize":     ("Nitrogen timing determines yield",
                  "Apply one-third N at sowing, one-third at knee height, one-third at tasseling. Ensure good drainage — even brief waterlogging causes significant loss. Intercrop with legumes for soil health. pH 6.0–7.5."),
    "Groundnut": ("Calcium + light, frequent irrigation",
                  "Apply gypsum at pegging stage to supply calcium to developing pods. Use light irrigation every 8–10 days. Avoid waterlogging, which causes pod rot. Inoculate seeds with Rhizobium for nitrogen fixation. pH 6.0–6.5."),
}

CROP_IMAGES = {
    "Rice":      "https://images.unsplash.com/photo-1536304993881-ff86e0c9ef22?w=600&q=80",
    "Cotton":    "https://images.unsplash.com/photo-1598514982601-0e7e45c6e5a3?w=600&q=80",
    "Chilli":    "https://images.unsplash.com/photo-1604908176997-431f2a3b6c07?w=600&q=80",
    "Maize":     "https://images.unsplash.com/photo-1601593768799-76b08d3d0de5?w=600&q=80",
    "Groundnut": "https://images.unsplash.com/photo-1615485500704-8e990f9900f7?w=600&q=80",
}
CROP_ICONS = {"Rice":"🌾","Cotton":"🌿","Chilli":"🌶️","Maize":"🌽","Groundnut":"🥜"}


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 🌱 Smart Crop Recommendation System")
st.markdown(
    '<span class="project-badge">B.Tech Final Year</span>'
    '<span class="project-badge">CSE Core</span>'
    '<span class="project-badge">SDP Final Review</span>'
    '<span class="project-badge">XGBoost + SMOTE</span>',
    unsafe_allow_html=True
)

st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

# ─── INPUT FORM ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card-title">📍 Regional Details</div>', unsafe_allow_html=True)
    district = st.selectbox("District", le_district.classes_)
    season   = st.selectbox("Season",   le_season.classes_)

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Soil Composition</div>', unsafe_allow_html=True)
    soil_ph    = st.slider("Soil pH",            4.0, 9.0,   6.5, 0.1, format="%.1f")
    nitrogen   = st.slider("Nitrogen (kg/ha)",   0,   150,   50)
    phosphorus = st.slider("Phosphorus (kg/ha)", 0,   150,   50)
    potassium  = st.slider("Potassium (kg/ha)",  0,   150,   50)

with right:
    st.markdown('<div class="card-title">🌦 Environmental Factors</div>', unsafe_allow_html=True)
    rainfall    = st.slider("Rainfall (mm)",    300, 1200, 600, 10)
    temperature = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5, format="%.1f")

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌍 Additional Context</div>', unsafe_allow_html=True)
    soil_type = st.selectbox("Dominant Soil Type",
        ["Black cotton soil","Red loamy soil","Alluvial soil","Sandy loam","Clay loam"])
    organic_matter = st.slider("Organic Matter (%)", 0.0, 5.0, 1.5, 0.1, format="%.1f")
    irrigation = st.selectbox("Irrigation Source",
        ["Canal","Borewell","Rainfed","Tank","Drip/Sprinkler"])

    st.write("")
    predict_btn = st.button("🚀  Predict Best Crop", use_container_width=True)


# ─── PREDICTION ──────────────────────────────────────────────────────────────
if predict_btn:
    inp = pd.DataFrame([{
        "District":    le_district.transform([district])[0],
        "Season":      le_season.transform([season])[0],
        "Soil_pH":     soil_ph,
        "Nitrogen":    nitrogen,
        "Phosphorus":  phosphorus,
        "Potassium":   potassium,
        "Rainfall":    rainfall,
        "Temperature": temperature,
    }])

    probs       = model.predict_proba(inp)[0]
    top3_idx    = np.argsort(probs)[-3:][::-1]
    top3_crops  = le_crop.inverse_transform(top3_idx)
    top3_probs  = probs[top3_idx]
    best_crop   = top3_crops[0]
    confidence  = top3_probs[0] * 100
    icon        = CROP_ICONS.get(best_crop, "🌿")
    adv_title, adv_body = ADVISORIES.get(best_crop,
        ("Follow standard practices","Consult your local agricultural extension officer."))

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("## Recommendation Results")

    if confidence > 80:
        conf_badge = '<span class="badge-h">✓ High Confidence</span>'
    elif confidence > 60:
        conf_badge = '<span class="badge-m">~ Moderate Confidence</span>'
    else:
        conf_badge = '<span class="badge-l">⚠ Low Confidence</span>'

    # ── Result banner ──
    st.markdown(f"""
    <div class="result-banner">
      <div>
        <div class="crop-sub">Top Recommendation</div>
        <div class="crop-big">{icon} &nbsp;{best_crop}</div>
        <div style="margin-top:8px">{conf_badge}</div>
      </div>
      <div>
        <div class="conf-lbl">Confidence Score</div>
        <div class="conf-pct">{confidence:.1f}%</div>
        <div class="conf-lbl">{district} · {season}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary pills ──
    st.markdown(
        f'<span class="metric-pill">pH {soil_ph:.1f}</span>'
        f'<span class="metric-pill">N {nitrogen} kg/ha</span>'
        f'<span class="metric-pill">P {phosphorus} kg/ha</span>'
        f'<span class="metric-pill">K {potassium} kg/ha</span>'
        f'<span class="metric-pill">🌧 {rainfall} mm</span>'
        f'<span class="metric-pill">🌡 {temperature:.1f}°C</span>'
        f'<span class="metric-pill">💧 {irrigation}</span>',
        unsafe_allow_html=True
    )

    st.write("")

    # ── Three-column output ──
    c1, c2, c3 = st.columns([1.2, 1, 1], gap="medium")

    with c1:
        st.markdown("#### 🏆 Top 3 Crops")
        for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs)):
            medal = ["🥇","🥈","🥉"][i]
            pct = prob * 100
            st.markdown(f"**{medal} {crop}** &nbsp; `{pct:.1f}%`")
            st.progress(int(pct))
        st.write("")

        # Crop image
        if best_crop in CROP_IMAGES:
            st.markdown(f"**{icon} {best_crop} — Field Image**")
            st.image(CROP_IMAGES[best_crop], use_container_width=True,
                     caption=f"{best_crop} crop — Andhra Pradesh")

    with c2:
        st.markdown("#### 📊 Feature Importance")
        importance = model.feature_importances_
        label_map = {
            "Soil_pH":"Soil pH","Nitrogen":"Nitrogen","Phosphorus":"Phosphorus",
            "Potassium":"Potassium","Rainfall":"Rainfall","Temperature":"Temperature",
            "District":"District","Season":"Season"
        }
        feat_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importance})
            .sort_values("Importance", ascending=False)
        )
        feat_df["Feature"] = feat_df["Feature"].map(label_map).fillna(feat_df["Feature"])
        st.bar_chart(feat_df.set_index("Feature"), height=250)
        top3_feat = feat_df["Feature"].head(3).tolist()
        st.caption(f"Key drivers: **{', '.join(top3_feat)}**")

    with c3:
        st.markdown("#### 🌾 Why This Crop?")
        st.markdown(f"""
        <div class="advisory">
          <div class="adv-title">💡 {adv_title}</div>
          <div class="adv-body">{adv_body}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        # Full probability expander
        with st.expander("📋 All crops probability table"):
            all_crops = le_crop.classes_
            prob_df = (
                pd.DataFrame({"Crop": all_crops, "Probability (%)": (probs*100).round(2)})
                .sort_values("Probability (%)", ascending=False)
                .reset_index(drop=True)
            )
            prob_df.index += 1
            st.dataframe(prob_df, use_container_width=True)

    # ── PDF Download ──
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("#### 📥 Download Recommendation Report")

    pdf_bytes = generate_pdf_report(
        district, season, soil_ph, nitrogen, phosphorus, potassium,
        rainfall, temperature, best_crop, confidence,
        top3_crops, top3_probs, feature_names, model.feature_importances_,
        adv_title, adv_body
    )

    fname = f"CropAdvisor_{best_crop}_{district}_{season}_{datetime.now().strftime('%d%b%Y')}.pdf"
    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="📄  Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    with info_col:
        st.markdown(
            f'<p style="color:#a5d6a7;font-size:0.82rem;padding-top:12px">'
            f'Full report includes: input parameters, top 3 recommendations, '
            f'feature importance table, agronomic advisory, and model details.<br>'
            f'Filename: <code style="color:#81c784">{fname}</code></p>',
            unsafe_allow_html=True
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:rgba(255,255,255,0.1)'>"
    "<p style='text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.4)'>"
    "B.Tech Final Year Project · CSE Core · SDP Final Review &nbsp;|&nbsp; "
    "Powered by XGBoost · SMOTE · Streamlit · ReportLab &nbsp;|&nbsp; "
    "For advisory use only — verify with local agricultural extension officer"
    "</p>",
    unsafe_allow_html=True
)