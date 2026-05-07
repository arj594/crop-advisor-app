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
    adv_title, adv_body,
    # ── NEW: uncertainty / robustness params ──
    reliability_label="—", raw_entropy=0.0, norm_entropy=0.0,
    robustness_score=0.0, climate_sensitivity="—",
    risk_level="—", risk_flags=None
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
         Paragraph("Adaptive AI Confidence Score", S("rc",
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
    # UNCERTAINTY & ROBUSTNESS ANALYSIS TABLE  (NEW SECTION)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Uncertainty & Robustness Analysis", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    risk_flags = risk_flags or []
    risk_flag_text = "; ".join([f[2] for f in risk_flags]) if risk_flags else "None"

    # Helper: wrap every cell in Paragraph so text auto-wraps instead of overflowing
    def _P(text, bold=False, color=BLACK, size=8.5):
        return Paragraph(str(text), ParagraphStyle(
            "cell", fontSize=size, fontName="Helvetica-Bold" if bold else "Helvetica",
            textColor=color, leading=11, wordWrap="LTR"
        ))

    HDR = colors.HexColor("#1a237e")
    uncertainty_rows = [
        [_P("Metric",                     bold=True, color=WHITE, size=8.5),
         _P("Value",                      bold=True, color=WHITE, size=8.5),
         _P("Interpretation",             bold=True, color=WHITE, size=8.5)],
        [_P("Adaptive AI Confidence Score"),
         _P(f"{confidence:.1f}%",         bold=True),
         _P("High (>80%), Moderate (60–80%), Low (<60%)")],
        [_P("Prediction Reliability"),
         _P(reliability_label,            bold=True),
         _P("Derived from Shannon entropy of class probabilities")],
        [_P("Predictive Entropy (raw)"),
         _P(f"{raw_entropy:.4f} nats",    bold=True),
         _P("Low entropy = model is certain; High entropy = model is uncertain")],
        [_P("Normalised Entropy"),
         _P(f"{norm_entropy:.3f}",        bold=True),
         _P("0 = fully certain,  1 = maximally uncertain")],
        [_P("Robustness Score"),
         _P(f"{robustness_score*100:.0f}%", bold=True),
         _P("% of 20 trials where top crop stayed the same under climate perturbation")],
        [_P("Climate Sensitivity"),
         _P(climate_sensitivity,          bold=True),
         _P("Sensitivity of recommendation to rainfall and temperature variation")],
        [_P("Overall Risk Level"),
         _P(risk_level,                   bold=True),
         _P("Composite of confidence score + robustness flags")],
        [_P("Active Risk Flags"),
         _P(risk_flag_text,               bold=True),
         _P("Warnings triggered by threshold analysis")],
    ]

    unc_table = Table(uncertainty_rows, colWidths=[5*cm, 3.2*cm, 8.8*cm],
                      repeatRows=1)          # repeat header on page break
    unc_style = [
        ("BACKGROUND",    (0,0), (-1,0), HDR),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),   # TOP so wrapped lines align
    ]
    unc_table.setStyle(TableStyle(unc_style))
    story.append(unc_table)
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
    story.append(Paragraph("Explainable AI Analysis — Feature Importance", s_section))
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
    page_title="Adaptive Crop Recommendation · AP",
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

/* ── Project subtitle ── */
.project-subtitle {
    font-size: 0.85rem;
    color: #81c784 !important;
    letter-spacing: 0.04em;
    margin: 6px 0 10px 0;
    font-style: italic;
}

/* ── Metric cards row (uncertainty / robustness) ── */
.metric-cards-row { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px; padding: 0.9rem 1rem;
    backdrop-filter: blur(4px);
}
.metric-card .mc-icon  { font-size: 1.3rem; margin-bottom: 4px; }
.metric-card .mc-label { font-size: 0.68rem; color: #81c784 !important; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .mc-value { font-size: 1.15rem; font-weight: 700; color: #ffffff !important; margin-top: 2px; }
.metric-card .mc-sub   { font-size: 0.72rem; color: #a5d6a7 !important; margin-top: 2px; }

/* ── Entropy / reliability panel ── */
.entropy-panel {
    background: rgba(30,60,100,0.35);
    border: 1px solid rgba(100,160,255,0.3);
    border-radius: 12px; padding: 1rem 1.25rem; margin: 0.5rem 0;
}
.entropy-panel .ep-title { font-size: 0.72rem; color: #90caf9 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.entropy-panel .ep-score { font-size: 1.6rem; font-weight: 800; color: #fff !important; }
.entropy-panel .ep-label { font-size: 0.82rem; color: #90caf9 !important; margin-top: 2px; }

/* ── Risk warning banners ── */
.risk-banner { border-radius: 10px; padding: 0.75rem 1.1rem; margin: 6px 0; display: flex; align-items: flex-start; gap: 10px; }
.risk-banner.risk-low-conf  { background: rgba(183,28,28,0.25); border: 1px solid rgba(229,115,115,0.5); }
.risk-banner.risk-climate   { background: rgba(230,81,0,0.25);  border: 1px solid rgba(255,183,77,0.5); }
.risk-banner .rb-icon  { font-size: 1.1rem; margin-top: 1px; }
.risk-banner .rb-title { font-size: 0.82rem; font-weight: 700; color: #fff !important; }
.risk-banner .rb-body  { font-size: 0.77rem; color: rgba(255,255,255,0.75) !important; margin-top: 2px; line-height: 1.4; }

/* ── Robustness bar ── */
.robust-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 999px; height: 10px; margin: 6px 0 2px; overflow: hidden; }
.robust-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ef9a9a, #a5d6a7); transition: width 0.6s; }

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
st.markdown("# 🌱 An Adaptive, Uncertainty-Aware Decision Framework for Intelligent Crop Recommendation Under Dynamic Agricultural Conditions")
st.markdown(
    '<p class="project-subtitle">Adaptive AI &nbsp;•&nbsp; Uncertainty-Aware Prediction &nbsp;•&nbsp; Explainable Recommendation System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<span class="project-badge">B.Tech Final Year</span>'
    '<span class="project-badge">CSE Core</span>'
    '<span class="project-badge">SDP Final Review</span>'
    '<span class="project-badge">XGBoost + SMOTE</span>'
    '<span class="project-badge">Uncertainty-Aware</span>'
    '<span class="project-badge">Explainable AI</span>',
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


# ─── UNCERTAINTY / ENTROPY HELPER ───────────────────────────────────────────
def compute_entropy(probs):
    """Shannon entropy over class probabilities.
    Low entropy  → model is certain  → High reliability
    High entropy → model is spread   → Low reliability
    Max possible entropy for N classes = log(N)
    """
    eps = 1e-12                          # avoid log(0)
    raw = -np.sum(probs * np.log(probs + eps))
    n_classes = len(probs)
    max_entropy = np.log(n_classes)      # normalise to [0,1]
    normalised  = raw / max_entropy if max_entropy > 0 else 0.0
    return round(raw, 4), round(normalised, 4)

def entropy_to_reliability(normalised_entropy):
    """Convert normalised entropy → reliability label + colour."""
    if normalised_entropy < 0.35:
        return "High",   "#4caf50"    # green
    elif normalised_entropy < 0.65:
        return "Medium", "#ffa726"    # amber
    else:
        return "Low",    "#ef5350"    # red


# ─── ROBUSTNESS / CLIMATE SENSITIVITY HELPER ────────────────────────────────
def compute_robustness(model, base_input_df, le_district, le_season,
                       best_crop, le_crop, n_trials=20):
    """Lightweight counterfactual robustness test.
    Perturbs rainfall (±10 %) and temperature (±2 °C) n_trials times,
    re-runs prediction each time, and returns the fraction of trials
    where the top-recommended crop stays the same.
    """
    rng  = np.random.default_rng(seed=42)
    same = 0
    rain_val = float(base_input_df["Rainfall"].iloc[0])
    temp_val = float(base_input_df["Temperature"].iloc[0])

    for _ in range(n_trials):
        perturbed = base_input_df.copy()
        perturbed["Rainfall"]    = rain_val * (1 + rng.uniform(-0.10, 0.10))
        perturbed["Temperature"] = temp_val + rng.uniform(-2.0,  2.0)
        trial_probs = model.predict_proba(perturbed)[0]
        trial_best  = le_crop.inverse_transform([np.argmax(trial_probs)])[0]
        if trial_best == best_crop:
            same += 1

    robustness = same / n_trials           # 0.0 – 1.0
    # Climate sensitivity: inverse of robustness
    if robustness >= 0.85:
        climate_sens = "Low"
    elif robustness >= 0.65:
        climate_sens = "Moderate"
    else:
        climate_sens = "High"
    return round(robustness, 4), climate_sens


# ─── RISK FLAG HELPER ────────────────────────────────────────────────────────
def get_risk_flags(confidence_pct, robustness_score):
    """Return list of active risk flags as (css_class, icon, title, body)."""
    flags = []
    if confidence_pct < 40:
        flags.append((
            "risk-low-conf", "⚠️",
            "Low Confidence Recommendation",
            f"Model confidence is {confidence_pct:.1f}% — below the 40% reliability threshold. "
            "Consider verifying with local agricultural experts before acting."
        ))
    if robustness_score < 0.75:
        flags.append((
            "risk-climate", "🌡️",
            "Prediction Sensitive to Climate Changes",
            f"Robustness score is {robustness_score*100:.0f}% — recommendation may change "
            "with moderate rainfall or temperature variation. Plan for contingency crops."
        ))
    return flags


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

    # ── Uncertainty analysis ──
    raw_entropy, norm_entropy = compute_entropy(probs)
    reliability_label, reliability_color = entropy_to_reliability(norm_entropy)

    # ── Robustness testing (lightweight, 20 trials) ──
    robustness_score, climate_sensitivity = compute_robustness(
        model, inp, le_district, le_season, best_crop, le_crop, n_trials=20
    )
    robustness_pct = robustness_score * 100

    # ── Risk flags ──
    risk_flags = get_risk_flags(confidence, robustness_score)

    # ── Derived risk level label ──
    if len(risk_flags) == 0:
        risk_level, risk_color = "Low Risk",    "#4caf50"
    elif len(risk_flags) == 1:
        risk_level, risk_color = "Medium Risk", "#ffa726"
    else:
        risk_level, risk_color = "High Risk",   "#ef5350" 

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
        <div class="conf-lbl">Adaptive AI Confidence Score</div>
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

    # ── Uncertainty & Robustness Metric Cards ──────────────────────────────────
    st.markdown(f"""
    <div class="metric-cards-row">
      <div class="metric-card">
        <div class="mc-icon">🎯</div>
        <div class="mc-label">Prediction Reliability</div>
        <div class="mc-value" style="color:{reliability_color} !important">{reliability_label}</div>
        <div class="mc-sub">Entropy: {raw_entropy:.4f} nats</div>
      </div>
      <div class="metric-card">
        <div class="mc-icon">🌡️</div>
        <div class="mc-label">Climate Sensitivity</div>
        <div class="mc-value">{climate_sensitivity}</div>
        <div class="mc-sub">Rainfall &amp; Temp perturbation</div>
      </div>
      <div class="metric-card">
        <div class="mc-icon">⚡</div>
        <div class="mc-label">Risk Level</div>
        <div class="mc-value" style="color:{risk_color} !important">{risk_level}</div>
        <div class="mc-sub">{len(risk_flags)} active flag(s)</div>
      </div>
      <div class="metric-card">
        <div class="mc-icon">🛡️</div>
        <div class="mc-label">Robustness Score</div>
        <div class="mc-value">{robustness_pct:.0f}%</div>
        <div class="mc-sub">{int(robustness_pct*0.2)}/20 stable trials</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Warning Banners (shown only when triggered) ───────────────────────
    for css_cls, rb_icon, rb_title, rb_body in risk_flags:
        st.markdown(f"""
        <div class="risk-banner {css_cls}">
          <div class="rb-icon">{rb_icon}</div>
          <div>
            <div class="rb-title">{rb_title}</div>
            <div class="rb-body">{rb_body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("#### 🔬 Explainable AI Analysis")
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

        # Entropy detail under chart
        st.markdown(f"""
        <div class="entropy-panel">
          <div class="ep-title">&#x1F4CA; Predictive Entropy (Uncertainty Measure)</div>
          <div class="ep-score">{raw_entropy:.4f} <span style="font-size:0.9rem;font-weight:400">nats</span></div>
          <div class="ep-label">Normalised: {norm_entropy:.3f} &nbsp;|&nbsp; Reliability: <b>{reliability_label}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("#### 🌾 Why This Crop?")
        st.markdown(f"""
        <div class="advisory">
          <div class="adv-title">💡 {adv_title}</div>
          <div class="adv-body">{adv_body}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Robustness bar
        st.markdown(f"""
        <div style="margin-bottom:8px">
          <div style="font-size:0.72rem;color:#81c784;text-transform:uppercase;letter-spacing:0.08em">
            &#x1F6E1; Robustness Score
          </div>
          <div class="robust-bar-wrap">
            <div class="robust-bar-fill" style="width:{robustness_pct:.0f}%"></div>
          </div>
          <div style="font-size:0.78rem;color:#a5d6a7">{robustness_pct:.0f}% stable across climate perturbations
          &nbsp;|&nbsp; Climate sensitivity: <b>{climate_sensitivity}</b></div>
        </div>
        """, unsafe_allow_html=True)

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
        adv_title, adv_body,
        # uncertainty / robustness params
        reliability_label=reliability_label,
        raw_entropy=raw_entropy,
        norm_entropy=norm_entropy,
        robustness_score=robustness_score,
        climate_sensitivity=climate_sensitivity,
        risk_level=risk_level,
        risk_flags=risk_flags,
    )

    fname = f"AdaptiveCropReport_{best_crop}_{district}_{season}_{datetime.now().strftime('%d%b%Y')}.pdf"
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
            f'Full report includes: input parameters, uncertainty analysis, '
            f'robustness scores, explainable AI section, advisory, and model details.<br>'
            f'Filename: <code style="color:#81c784">{fname}</code></p>',
            unsafe_allow_html=True
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:rgba(255,255,255,0.1)'>"
    "<p style='text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.4)'>"
    "B.Tech Final Year Project · CSE Core · SDP Final Review &nbsp;|&nbsp; "
    "Adaptive AI · Uncertainty-Aware · Explainable Recommendation System &nbsp;|&nbsp; "
    "Recommendation generated using adaptive machine learning analysis."
    "</p>",
    unsafe_allow_html=True
)
