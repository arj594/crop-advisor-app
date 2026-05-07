
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import io
from datetime import datetime
"""
multi_agent.py  —  Yield Agent · Risk Agent · Market Agent · Fusion Rule
Four-module decision layer that wraps XGBoost predictions with
domain-specific intelligence and a weighted fusion rule.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASES  (static, no external calls needed — self-contained)
# ══════════════════════════════════════════════════════════════════════════════

# Expected yield range (quintal/acre) per crop per season
YIELD_KB = {
    "Rice":      {"Kharif": (18, 28), "Rabi": (20, 30), "Zaid": (15, 22)},
    "Cotton":    {"Kharif": ( 6, 10), "Rabi": ( 5,  8), "Zaid": ( 4,  7)},
    "Chilli":    {"Kharif": ( 8, 14), "Rabi": (10, 16), "Zaid": ( 6, 10)},
    "Maize":     {"Kharif": (16, 24), "Rabi": (18, 26), "Zaid": (14, 20)},
    "Groundnut": {"Kharif": ( 8, 13), "Rabi": ( 9, 14), "Zaid": ( 7, 11)},
}

# Soil-pH suitability window per crop
PH_WINDOW = {
    "Rice":      (5.5, 7.0),
    "Cotton":    (6.0, 8.0),
    "Chilli":    (5.5, 7.0),
    "Maize":     (5.8, 7.5),
    "Groundnut": (5.5, 6.5),
}

# Optimal NPK range (kg/ha) per crop
NPK_OPTIMA = {
    "Rice":      {"N": (80, 120), "P": (40, 60),  "K": (40, 60)},
    "Cotton":    {"N": (60, 100), "P": (30, 50),  "K": (30, 60)},
    "Chilli":    {"N": (60, 100), "P": (40, 60),  "K": (40, 60)},
    "Maize":     {"N": (80, 120), "P": (40, 60),  "K": (30, 50)},
    "Groundnut": {"N": (20,  40), "P": (40, 60),  "K": (40, 60)},
}

# Rainfall tolerance window (mm / season) per crop
RAIN_WINDOW = {
    "Rice":      (800, 1200),
    "Cotton":    (500,  900),
    "Chilli":    (500,  800),
    "Maize":     (500,  900),
    "Groundnut": (400,  700),
}

# Temperature tolerance (°C) per crop
TEMP_WINDOW = {
    "Rice":      (22, 35),
    "Cotton":    (21, 38),
    "Chilli":    (20, 35),
    "Maize":     (18, 33),
    "Groundnut": (20, 35),
}

# ── Market KB: MSP (₹/quintal) + demand trend + price volatility
MARKET_KB = {
    "Rice":      {"mspp": 2183, "demand": "High",   "volatility": "Low",      "export_potential": "Medium"},
    "Cotton":    {"mspp": 6620, "demand": "High",   "volatility": "High",     "export_potential": "High"},
    "Chilli":    {"mspp": 3000, "demand": "Medium", "volatility": "Very High","export_potential": "High"},
    "Maize":     {"mspp": 1962, "demand": "Medium", "volatility": "Low",      "export_potential": "Medium"},
    "Groundnut": {"mspp": 5850, "demand": "Medium", "volatility": "Medium",   "export_potential": "Medium"},
}

# ── Risk KB: common pest/disease + natural disaster sensitivity
RISK_KB = {
    "Rice":      {"primary_pest": "Stem Borer, Blast",          "flood_risk": "High",   "drought_risk": "Low",    "ipm_difficulty": "Medium"},
    "Cotton":    {"primary_pest": "Bollworm, Whitefly",         "flood_risk": "Low",    "drought_risk": "High",   "ipm_difficulty": "High"},
    "Chilli":    {"primary_pest": "Thrips, Anthracnose",        "flood_risk": "Medium", "drought_risk": "Medium", "ipm_difficulty": "High"},
    "Maize":     {"primary_pest": "Fall Armyworm, Stem Borer",  "flood_risk": "Medium", "drought_risk": "Medium", "ipm_difficulty": "Medium"},
    "Groundnut": {"primary_pest": "Leaf Miner, Collar Rot",     "flood_risk": "Low",    "drought_risk": "High",   "ipm_difficulty": "Low"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: score a numeric input vs an (low, high) optimal window  → 0‥1
# ══════════════════════════════════════════════════════════════════════════════
def _window_score(val, lo, hi):
    """Returns 1.0 if val is inside [lo, hi], tapers linearly outside."""
    if lo <= val <= hi:
        return 1.0
    margin = (hi - lo) * 0.5
    if val < lo:
        return max(0.0, 1.0 - (lo - val) / max(margin, 1))
    return max(0.0, 1.0 - (val - hi) / max(margin, 1))


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 1 — YIELD AGENT
#  Estimates expected yield, soil-suitability, & growth-factor scores
# ══════════════════════════════════════════════════════════════════════════════
def yield_agent(crop, season, soil_ph, nitrogen, phosphorus,
                potassium, rainfall, temperature):
    """
    Returns a dict with:
      yield_lo, yield_hi   — expected quintal/acre range
      yield_score          — 0‥1 overall yield suitability
      soil_ph_ok           — bool
      npk_score            — 0‥1
      climate_score        — 0‥1
      factors              — list of (label, score, emoji) for display
    """
    # Soil pH
    ph_lo, ph_hi = PH_WINDOW.get(crop, (5.5, 7.5))
    ph_score = _window_score(soil_ph, ph_lo, ph_hi)
    soil_ph_ok = (ph_lo <= soil_ph <= ph_hi)

    # NPK
    npk = NPK_OPTIMA.get(crop, {"N": (60, 100), "P": (30, 50), "K": (30, 50)})
    n_score = _window_score(nitrogen,   *npk["N"])
    p_score = _window_score(phosphorus, *npk["P"])
    k_score = _window_score(potassium,  *npk["K"])
    npk_score = (n_score + p_score + k_score) / 3

    # Climate
    r_lo, r_hi = RAIN_WINDOW.get(crop, (500, 900))
    t_lo, t_hi = TEMP_WINDOW.get(crop, (20, 35))
    rain_score = _window_score(rainfall,    r_lo, r_hi)
    temp_score = _window_score(temperature, t_lo, t_hi)
    climate_score = (rain_score + temp_score) / 2

    # Composite yield score
    yield_score = round(0.30 * ph_score + 0.35 * npk_score + 0.35 * climate_score, 3)

    # Yield range — scale by yield_score
    y_kb = YIELD_KB.get(crop, {})
    base_lo, base_hi = y_kb.get(season, y_kb.get("Kharif", (10, 20)))
    adj_lo = round(base_lo * (0.7 + 0.3 * yield_score), 1)
    adj_hi = round(base_hi * (0.7 + 0.3 * yield_score), 1)

    factors = [
        ("Soil pH",      round(ph_score * 100),      "🧪"),
        ("Nitrogen",     round(n_score  * 100),      "🌿"),
        ("Phosphorus",   round(p_score  * 100),      "⚗️"),
        ("Potassium",    round(k_score  * 100),      "🔋"),
        ("Rainfall",     round(rain_score * 100),    "🌧️"),
        ("Temperature",  round(temp_score * 100),    "🌡️"),
    ]

    return {
        "yield_lo":      adj_lo,
        "yield_hi":      adj_hi,
        "yield_score":   yield_score,
        "soil_ph_ok":    soil_ph_ok,
        "npk_score":     round(npk_score, 3),
        "climate_score": round(climate_score, 3),
        "factors":       factors,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 2 — RISK AGENT
#  Evaluates pest, flood, drought and input-cost risk
# ══════════════════════════════════════════════════════════════════════════════
def risk_agent(crop, rainfall, temperature, robustness_score):
    """
    Returns:
      risk_score     — 0‥1  (higher = riskier)
      risk_grade     — "Low" / "Moderate" / "High" / "Very High"
      pest_warnings  — list of str
      risk_factors   — list of (label, level, emoji)
    """
    kb = RISK_KB.get(crop, {})

    # Flood risk from actual rainfall
    r_hi = RAIN_WINDOW.get(crop, (500, 900))[1]
    flood_score = min(1.0, max(0.0, (rainfall - r_hi) / 300)) if rainfall > r_hi else 0.0

    # Drought risk
    r_lo = RAIN_WINDOW.get(crop, (500, 900))[0]
    drought_score = min(1.0, max(0.0, (r_lo - rainfall) / 200)) if rainfall < r_lo else 0.0

    # Pest risk — higher temp & humidity (rainfall proxy) → more pest pressure
    t_hi = TEMP_WINDOW.get(crop, (20, 35))[1]
    pest_score = min(1.0, max(0.0, (temperature - (t_hi - 4)) / 4))
    if rainfall > 700:
        pest_score = min(1.0, pest_score + 0.2)

    # Climate robustness risk (inverse)
    stability_risk = round(1.0 - robustness_score, 3)

    # Composite
    risk_score = round(
        0.30 * pest_score +
        0.25 * flood_score +
        0.20 * drought_score +
        0.25 * stability_risk, 3
    )

    if risk_score < 0.25:
        risk_grade = "Low"
    elif risk_score < 0.50:
        risk_grade = "Moderate"
    elif risk_score < 0.75:
        risk_grade = "High"
    else:
        risk_grade = "Very High"

    pest_warnings = []
    if pest_score > 0.4:
        pest_warnings.append(f"Elevated pest pressure — monitor for {kb.get('primary_pest','common pests')}")
    if flood_score > 0.3:
        pest_warnings.append("Above-optimal rainfall — waterlogging / fungal disease risk")
    if drought_score > 0.3:
        pest_warnings.append("Below-optimal rainfall — moisture stress / irrigation needed")
    if stability_risk > 0.35:
        pest_warnings.append("Prediction instability under climate variation — consider contingency crop")

    risk_factors = [
        ("Pest Pressure",      _level(pest_score),      "🐛"),
        ("Flood Risk",         kb.get("flood_risk",   "Medium"), "🌊"),
        ("Drought Risk",       kb.get("drought_risk", "Medium"), "☀️"),
        ("Climate Stability",  _level(stability_risk),  "🌪️"),
        ("IPM Difficulty",     kb.get("ipm_difficulty","Medium"),"🔬"),
    ]

    return {
        "risk_score":    risk_score,
        "risk_grade":    risk_grade,
        "pest_warnings": pest_warnings,
        "risk_factors":  risk_factors,
    }


def _level(score):
    if score < 0.25: return "Low"
    if score < 0.50: return "Moderate"
    if score < 0.75: return "High"
    return "Very High"


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT 3 — MARKET AGENT
#  Evaluates MSP, price volatility, demand outlook, revenue estimate
# ══════════════════════════════════════════════════════════════════════════════
def market_agent(crop, yield_lo, yield_hi):
    """
    Returns:
      mspp           — Minimum Support Price ₹/quintal
      demand         — str
      volatility     — str
      export_potential — str
      revenue_lo     — ₹ estimated (yield_lo × msp)
      revenue_hi     — ₹ estimated (yield_hi × msp)
      market_score   — 0‥1 (higher = better market opportunity)
      market_insight — str
    """
    kb = MARKET_KB.get(crop, {"mspp": 2000, "demand": "Medium",
                               "volatility": "Medium", "export_potential": "Medium"})
    mspp = kb["mspp"]
    demand = kb["demand"]
    vol = kb["volatility"]
    export = kb["export_potential"]

    # Revenue estimate per acre
    revenue_lo = round(yield_lo * mspp)
    revenue_hi = round(yield_hi * mspp)

    # Market score
    demand_s   = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(demand, 0.5)
    vol_s      = {"Low": 1.0, "Medium": 0.7, "High": 0.4, "Very High": 0.2}.get(vol, 0.5)
    export_s   = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(export, 0.5)
    market_score = round((demand_s + vol_s + export_s) / 3, 3)

    # Text insight
    if market_score >= 0.75:
        insight = "Strong market outlook — high demand, stable prices, good export potential."
    elif market_score >= 0.50:
        insight = "Moderate market outlook — demand is acceptable, monitor price trends before selling."
    else:
        insight = "Cautious market outlook — high price volatility; consider contract farming or staggered sale."

    return {
        "mspp":             mspp,
        "demand":           demand,
        "volatility":       vol,
        "export_potential": export,
        "revenue_lo":       revenue_lo,
        "revenue_hi":       revenue_hi,
        "market_score":     market_score,
        "market_insight":   insight,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION RULE
#  Weighted combination of ML confidence + Yield + Risk + Market scores
#  → Final Composite Decision Score + Natural-language verdict
# ══════════════════════════════════════════════════════════════════════════════
FUSION_WEIGHTS = {
    "ml_confidence": 0.35,   # XGBoost model confidence
    "yield_score":   0.30,   # Yield Agent
    "market_score":  0.20,   # Market Agent
    "risk_penalty":  0.15,   # Risk Agent (inverted — lower risk = higher score)
}

def fusion_rule(ml_confidence_pct, yield_result, risk_result, market_result):
    """
    Returns:
      composite_score   — 0‥100
      verdict           — "Strongly Recommended" / "Recommended" /
                          "Conditionally Recommended" / "Not Recommended"
      verdict_color     — hex color
      rationale         — list of str (bullet points for UI)
      agent_scores      — dict of individual normalized scores (0‥1)
    """
    ml_norm    = ml_confidence_pct / 100
    yield_norm = yield_result["yield_score"]          # already 0‥1
    risk_norm  = 1.0 - risk_result["risk_score"]      # invert — less risk = better
    market_norm = market_result["market_score"]       # already 0‥1

    composite = (
        FUSION_WEIGHTS["ml_confidence"] * ml_norm +
        FUSION_WEIGHTS["yield_score"]   * yield_norm +
        FUSION_WEIGHTS["market_score"]  * market_norm +
        FUSION_WEIGHTS["risk_penalty"]  * risk_norm
    )
    composite_score = round(composite * 100, 1)

    if composite_score >= 75:
        verdict       = "Strongly Recommended"
        verdict_color = "#4caf50"
    elif composite_score >= 55:
        verdict       = "Recommended"
        verdict_color = "#8bc34a"
    elif composite_score >= 38:
        verdict       = "Conditionally Recommended"
        verdict_color = "#ffa726"
    else:
        verdict       = "Not Recommended"
        verdict_color = "#ef5350"

    # Build rationale bullets
    rationale = []
    if ml_norm >= 0.8:
        rationale.append("ML model shows high confidence — strong agro-climatic pattern match.")
    elif ml_norm >= 0.6:
        rationale.append("ML model shows moderate confidence — reasonable pattern match.")
    else:
        rationale.append("ML confidence is low — consider alternative crops from Top-3 list.")

    if yield_norm >= 0.75:
        rationale.append("Soil and climate conditions are highly favourable for this crop.")
    elif yield_norm >= 0.50:
        rationale.append("Soil/climate conditions are adequate; some parameter optimisation advised.")
    else:
        rationale.append("Suboptimal soil or climate conditions detected — yield may be reduced.")

    if risk_result["risk_score"] < 0.25:
        rationale.append("Risk profile is low — minimal pest, flood, and drought concerns.")
    elif risk_result["risk_score"] < 0.50:
        rationale.append("Moderate risk detected — implement IPM and water management protocols.")
    else:
        rationale.append("High risk level — strong mitigation measures required before proceeding.")

    if market_result["market_score"] >= 0.70:
        rationale.append(f"Market outlook is strong — MSP ₹{market_result['mspp']:,}/qtl, high demand.")
    elif market_result["market_score"] >= 0.50:
        rationale.append(f"Market conditions are fair — MSP ₹{market_result['mspp']:,}/qtl, moderate volatility.")
    else:
        rationale.append(f"Market outlook is uncertain — price volatility is {market_result['volatility'].lower()}; hedge before sowing.")

    agent_scores = {
        "ML Confidence":  round(ml_norm * 100, 1),
        "Yield Score":    round(yield_norm * 100, 1),
        "Risk Score":     round(risk_result["risk_score"] * 100, 1),
        "Market Score":   round(market_norm * 100, 1),
    }

    return {
        "composite_score": composite_score,
        "verdict":         verdict,
        "verdict_color":   verdict_color,
        "rationale":       rationale,
        "agent_scores":    agent_scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: run all three agents + fusion in one call
# ══════════════════════════════════════════════════════════════════════════════
def run_all_agents(crop, season, soil_ph, nitrogen, phosphorus, potassium,
                   rainfall, temperature, ml_confidence_pct, robustness_score):
    ya = yield_agent(crop, season, soil_ph, nitrogen, phosphorus,
                     potassium, rainfall, temperature)
    ra = risk_agent(crop, rainfall, temperature, robustness_score)
    ma = market_agent(crop, ya["yield_lo"], ya["yield_hi"])
    fu = fusion_rule(ml_confidence_pct, ya, ra, ma)
    return ya, ra, ma, fu
# ─── CONSTANTS ───────────────────────────────────────────────────────────────
PROJECT_TITLE = "An Adaptive, Uncertainty-Aware Decision Framework for Intelligent Crop Recommendation Under Dynamic Agricultural Conditions"
PROJECT_SHORT = "Adaptive Crop Recommendation System"
AUTHOR_NAME   = "Arjun Mishra"
INSTITUTION   = "B.Tech Final Year Project · CSE Core · SDP Final Review"

# ─── PDF REPORT GENERATOR ────────────────────────────────────────────────────
def generate_pdf_report(
    district, season, soil_ph, nitrogen, phosphorus, potassium,
    rainfall, temperature, best_crop, confidence,
    top3_crops, top3_probs, feature_names, importances,
    adv_title, adv_body,
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
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=f"Crop Recommendation Report — {PROJECT_SHORT}"
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
    NAVY        = colors.HexColor("#1a237e")

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    s_title     = S("title",     fontSize=20, fontName="Helvetica-Bold",  textColor=WHITE,                     alignment=TA_CENTER, spaceAfter=2)
    s_subtitle  = S("subtitle",  fontSize=9,  fontName="Helvetica",       textColor=colors.HexColor("#2e7b34"), alignment=TA_CENTER, spaceAfter=2)
    s_author    = S("author",    fontSize=9,  fontName="Helvetica-Bold",  textColor=colors.HexColor("#1b5e20"), alignment=TA_CENTER, spaceAfter=4)
    s_section   = S("section",   fontSize=12, fontName="Helvetica-Bold",  textColor=GREEN_DARK,                spaceBefore=12, spaceAfter=6)
    s_adv_title = S("adv_title", fontSize=11, fontName="Helvetica-Bold",  textColor=colors.HexColor("#7a5c00"), spaceAfter=4)
    s_adv_body  = S("adv_body",  fontSize=10, fontName="Helvetica",       textColor=colors.HexColor("#5a4500"), leading=15)

    story = []

    # ══════════════════════════════════════════════════════════════
    # HEADER BANNER
    # ══════════════════════════════════════════════════════════════
    # Two-line header: icon+short line, then full project title below
    s_banner_top = S("banner_top",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#a5d6a7"),
        alignment=TA_CENTER, spaceAfter=4, leading=14)
    s_banner_title = S("banner_title",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=0, leading=17)

    header_data = [[
        Paragraph("🌱  Smart Crop Advisor  —  Adaptive AI Recommendation System", s_banner_top),
    ],[
        Paragraph(PROJECT_TITLE, s_banner_title),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GREEN_MID),
        ("TOPPADDING",    (0,0), (0,0),   14),
        ("BOTTOMPADDING", (0,0), (0,0),   2),
        ("TOPPADDING",    (0,1), (-1,-1), 2),
        ("BOTTOMPADDING", (0,1), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))

    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    sub_data = [[
        Paragraph(f"Created by: {AUTHOR_NAME}  ·  {INSTITUTION}", s_author),
        Paragraph(f"Generated: {now}", S("dt",
            fontSize=8.5, fontName="Helvetica", textColor=GRAY, alignment=TA_RIGHT)),
    ]]
    sub_tbl = Table(sub_data, colWidths=[12*cm, 5*cm])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f5f7f2")),
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
    crop_icons  = {"Rice":"🌾","Cotton":"🌿","Chilli":"🌶","Maize":"🌽","Groundnut":"🥜"}
    icon        = crop_icons.get(best_crop, "🌱")
    conf_label  = ("High Confidence" if confidence >= 80 else
                   ("Moderate Confidence" if confidence >= 60 else "Low Confidence"))

    result_data = [
        [Paragraph(f"{icon}  Recommended Crop", S("rl",
             fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#a5d6a7"))),
         Paragraph("Adaptive AI Confidence Score", S("rc",
             fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#a5d6a7"), alignment=TA_RIGHT))],
        [Paragraph(best_crop, S("rn",
             fontSize=26, fontName="Helvetica-Bold", textColor=WHITE)),
         Paragraph(f"{confidence:.1f}%", S("rpct",
             fontSize=26, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_RIGHT))],
        [Paragraph(f"{district} · {season} season", S("rs",
             fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#c8e6c9"))),
         Paragraph(conf_label, S("rcl",
             fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_RIGHT))],
    ]
    result_table = Table(result_data, colWidths=[10*cm, 7*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GREEN_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 18),
        ("RIGHTPADDING",  (0,0), (-1,-1), 18),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # UNCERTAINTY & ROBUSTNESS ANALYSIS TABLE
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Uncertainty & Robustness Analysis", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    risk_flags      = risk_flags or []
    risk_flag_text  = "; ".join([f[2] for f in risk_flags]) if risk_flags else "None"

    def _P(text, bold=False, color=BLACK, size=8.5):
        return Paragraph(str(text), ParagraphStyle(
            "cell", fontSize=size,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            textColor=color, leading=11, wordWrap="LTR"
        ))

    uncertainty_rows = [
        [_P("Metric",                       bold=True, color=WHITE),
         _P("Value",                        bold=True, color=WHITE),
         _P("Interpretation",               bold=True, color=WHITE)],
        [_P("Adaptive AI Confidence Score"),
         _P(f"{confidence:.1f}%",           bold=True),
         _P("High (>80%), Moderate (60-80%), Low (<60%)")],
        [_P("Prediction Reliability"),
         _P(reliability_label,              bold=True),
         _P("Derived from Shannon entropy of class probabilities")],
        [_P("Predictive Entropy (raw)"),
         _P(f"{raw_entropy:.4f} nats",      bold=True),
         _P("Low entropy = model is certain; High entropy = model is uncertain")],
        [_P("Normalised Entropy"),
         _P(f"{norm_entropy:.3f}",          bold=True),
         _P("0 = fully certain,  1 = maximally uncertain")],
        [_P("Robustness Score"),
         _P(f"{robustness_score*100:.0f}%", bold=True),
         _P("% of 20 trials where top crop stayed same under climate perturbation")],
        [_P("Climate Sensitivity"),
         _P(climate_sensitivity,            bold=True),
         _P("Sensitivity of recommendation to rainfall and temperature variation")],
        [_P("Overall Risk Level"),
         _P(risk_level,                     bold=True),
         _P("Composite of confidence score + robustness flags")],
        [_P("Active Risk Flags"),
         _P(risk_flag_text,                 bold=True),
         _P("Warnings triggered by threshold analysis")],
    ]

    unc_table = Table(uncertainty_rows, colWidths=[5*cm, 3.2*cm, 8.8*cm], repeatRows=1)
    unc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.append(unc_table)
    story.append(Spacer(1, 16))

    # ══════════════════════════════════════════════════════════════
    # INPUT PARAMETERS TABLE
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Field Input Parameters", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN_LIGHT, spaceAfter=6))

    params = [
        ["Parameter",   "Value",              "Unit",  "Optimal Range"],
        ["District",    district,             "—",     "Andhra Pradesh"],
        ["Season",      season,               "—",     "Kharif / Rabi / Zaid"],
        ["Soil pH",     f"{soil_ph:.1f}",     "pH",    "6.0 – 7.5"],
        ["Nitrogen",    str(nitrogen),        "kg/ha", "40 – 80"],
        ["Phosphorus",  str(phosphorus),      "kg/ha", "20 – 60"],
        ["Potassium",   str(potassium),       "kg/ha", "30 – 70"],
        ["Rainfall",    str(rainfall),        "mm",    "500 – 900"],
        ["Temperature", f"{temperature:.1f}", "°C",    "20 – 35"],
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

    medals   = ["#1", "#2", "#3"]
    rec_rows = [["Rank", "Crop", "Probability", "Confidence Bar"]]
    for i in range(3):
        bar_pct = int(top3_probs[i] * 100)
        bar = "█" * (bar_pct // 5) + "░" * (20 - bar_pct // 5)
        rec_rows.append([f"{medals[i]}", top3_crops[i], f"{bar_pct}%", bar[:20]])

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
    feat_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
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
    imp_rows = [["Feature", "Importance Score", "Visual", "Interpretation"]]
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

    adv_data = [[Paragraph(f"  {adv_title}", s_adv_title)],
                [Paragraph(adv_body, s_adv_body)]]
    adv_table = Table(adv_data, colWidths=[17*cm])
    adv_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), AMBER_PALE),
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
        ["Features",       ", ".join([label_map.get(f, f) for f in feature_names])],
        ["Developed by",   AUTHOR_NAME],
        ["Project",        PROJECT_SHORT],
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
    footer_text = (
        f"{PROJECT_SHORT}  |  {INSTITUTION}  |  "
        f"Created by {AUTHOR_NAME}  |  Powered by XGBoost + SMOTE + Streamlit"
    )
    footer_data = [[
        Paragraph(footer_text, S("ft",
            fontSize=8, fontName="Helvetica", textColor=WHITE, alignment=TA_CENTER))
    ]]
    footer_table = Table(footer_data, colWidths=[17*cm])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GREEN_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
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
.stApp {
    background-image:
        linear-gradient(rgba(10,30,10,0.70), rgba(10,30,10,0.80)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1800&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
h1, h2, h3, h4, h5 { color: #ffffff !important; }
label, p, .stMarkdown p { color: #d9efd2 !important; }
.stSlider label { color: #d9efd2 !important; }

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
.advisory {
    background: rgba(249,168,37,0.15);
    border: 1px solid rgba(249,168,37,0.4);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}
.advisory .adv-title { font-weight: 700; color: #ffe082 !important; font-size: 0.9rem; }
.advisory .adv-body  { color: #fff9c4 !important; font-size: 0.85rem; line-height: 1.6; }
.badge-h { background:#1b5e20; color:#a5d6a7; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #4caf50; }
.badge-m { background:#0d3a75; color:#90caf9; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #42a5f5; }
.badge-l { background:#7a4a00; color:#ffe082; border-radius:999px; padding:3px 12px; font-size:0.75rem; font-weight:700; display:inline-block; border:1px solid #ffa000; }
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
.sdiv { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }
.project-subtitle {
    font-size: 0.85rem;
    color: #81c784 !important;
    letter-spacing: 0.04em;
    margin: 6px 0 10px 0;
    font-style: italic;
}
/* Author credit strip */
.author-strip {
    background: rgba(27,94,32,0.35);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #a5d6a7 !important;
    display: inline-block;
    margin-bottom: 8px;
}
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
.entropy-panel {
    background: rgba(30,60,100,0.35);
    border: 1px solid rgba(100,160,255,0.3);
    border-radius: 12px; padding: 1rem 1.25rem; margin: 0.5rem 0;
}
.entropy-panel .ep-title { font-size: 0.72rem; color: #90caf9 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.entropy-panel .ep-score { font-size: 1.6rem; font-weight: 800; color: #fff !important; }
.entropy-panel .ep-label { font-size: 0.82rem; color: #90caf9 !important; margin-top: 2px; }
.risk-banner { border-radius: 10px; padding: 0.75rem 1.1rem; margin: 6px 0; display: flex; align-items: flex-start; gap: 10px; }
.risk-banner.risk-low-conf  { background: rgba(183,28,28,0.25); border: 1px solid rgba(229,115,115,0.5); }
.risk-banner.risk-climate   { background: rgba(230,81,0,0.25);  border: 1px solid rgba(255,183,77,0.5); }
.risk-banner .rb-icon  { font-size: 1.1rem; margin-top: 1px; }
.risk-banner .rb-title { font-size: 0.82rem; font-weight: 700; color: #fff !important; }
.risk-banner .rb-body  { font-size: 0.77rem; color: rgba(255,255,255,0.75) !important; margin-top: 2px; line-height: 1.4; }
.robust-bar-wrap { background: rgba(255,255,255,0.1); border-radius: 999px; height: 10px; margin: 6px 0 2px; overflow: hidden; }
.robust-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ef9a9a, #a5d6a7); transition: width 0.6s; }
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
st.markdown(f"# 🌱 {PROJECT_TITLE}")
st.markdown(
    '<p class="project-subtitle">Adaptive AI &nbsp;•&nbsp; Uncertainty-Aware Prediction &nbsp;•&nbsp; Explainable Recommendation System</p>',
    unsafe_allow_html=True
)
# Author credit visible in the UI
st.markdown(
    f'<span class="author-strip">👤 Created by {AUTHOR_NAME}</span>',
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
    eps = 1e-12
    raw = -np.sum(probs * np.log(probs + eps))
    n_classes = len(probs)
    max_entropy = np.log(n_classes)
    normalised  = raw / max_entropy if max_entropy > 0 else 0.0
    return round(raw, 4), round(normalised, 4)

def entropy_to_reliability(normalised_entropy):
    if normalised_entropy < 0.35:
        return "High",   "#4caf50"
    elif normalised_entropy < 0.65:
        return "Medium", "#ffa726"
    else:
        return "Low",    "#ef5350"


# ─── ROBUSTNESS / CLIMATE SENSITIVITY HELPER ────────────────────────────────
def compute_robustness(model, base_input_df, le_district, le_season,
                       best_crop, le_crop, n_trials=20):
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

    robustness = same / n_trials
    if robustness >= 0.85:
        climate_sens = "Low"
    elif robustness >= 0.65:
        climate_sens = "Moderate"
    else:
        climate_sens = "High"
    return round(robustness, 4), climate_sens


# ─── RISK FLAG HELPER ────────────────────────────────────────────────────────
def get_risk_flags(confidence_pct, robustness_score):
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

    raw_entropy, norm_entropy = compute_entropy(probs)
    reliability_label, reliability_color = entropy_to_reliability(norm_entropy)

    robustness_score, climate_sensitivity = compute_robustness(
        model, inp, le_district, le_season, best_crop, le_crop, n_trials=20
    )
    robustness_pct = robustness_score * 100
        # ─── MULTI AGENT SYSTEM ─────────────────────────────
    yield_result, risk_result, market_result, fusion_result = run_all_agents(
        crop=best_crop,
        season=season,
        soil_ph=soil_ph,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        rainfall=rainfall,
        temperature=temperature,
        ml_confidence_pct=confidence,
        robustness_score=robustness_score
    )

    risk_flags = get_risk_flags(confidence, robustness_score)

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

    c1, c2, c3 = st.columns([1.2, 1, 1], gap="medium")

    with c1:
        st.markdown("#### 🏆 Top 3 Crops")
        for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs)):
            medal = ["🥇","🥈","🥉"][i]
            pct = prob * 100
            st.markdown(f"**{medal} {crop}** &nbsp; `{pct:.1f}%`")
            st.progress(int(pct))
        st.write("")
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

        with st.expander("📋 All crops probability table"):
            all_crops = le_crop.classes_
            prob_df = (
                pd.DataFrame({"Crop": all_crops, "Probability (%)": (probs*100).round(2)})
                .sort_values("Probability (%)", ascending=False)
                .reset_index(drop=True)
            )
            prob_df.index += 1
            st.dataframe(prob_df, use_container_width=True)
                # ─── MULTI AGENT DECISION FRAMEWORK ─────────────────────────────

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("## 🤖 Multi-Agent Decision Framework")

    st.markdown(f"""
    <div class="result-banner">
      <div>
        <div class="crop-sub">Composite Decision Score</div>
        <div class="crop-big">🧠 {fusion_result['composite_score']}%</div>
        <div style="margin-top:8px">
            <span class="badge-h">{fusion_result['verdict']}</span>
        </div>
      </div>
      <div>
        <div class="conf-lbl">Fusion-Based Agricultural Intelligence</div>
        <div class="conf-pct">{fusion_result['composite_score']}%</div>
        <div class="conf-lbl">Yield + Risk + Market + ML</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ag1, ag2, ag3 = st.columns(3)

    with ag1:
        st.markdown("### 🌾 Yield Agent")
        st.metric(
            "Expected Yield",
            f"{yield_result['yield_lo']} - {yield_result['yield_hi']} q/acre"
        )
        st.write(f"Suitability Score: {yield_result['yield_score']*100:.1f}%")

    with ag2:
        st.markdown("### ⚠️ Risk Agent")
        st.metric(
            "Risk Grade",
            risk_result['risk_grade']
        )
        st.write(f"Risk Score: {risk_result['risk_score']*100:.1f}%")

    with ag3:
        st.markdown("### 💰 Market Agent")
        st.metric(
            "MSP",
            f"₹{market_result['mspp']}/qtl"
        )
        st.write(f"Demand: {market_result['demand']}")
        st.write(f"Export Potential: {market_result['export_potential']}")

    st.markdown("### 📌 Fusion Rationale")

    for point in fusion_result["rationale"]:
        st.write(f"• {point}")

    # ── PDF Download ──
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("#### 📥 Download Recommendation Report")

    pdf_bytes = generate_pdf_report(
        district, season, soil_ph, nitrogen, phosphorus, potassium,
        rainfall, temperature, best_crop, confidence,
        top3_crops, top3_probs, feature_names, model.feature_importances_,
        adv_title, adv_body,
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
    f"<p style='text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.4)'>"
    f"{INSTITUTION} &nbsp;|&nbsp; "
    f"Adaptive AI · Uncertainty-Aware · Explainable Recommendation System &nbsp;|&nbsp; "
    f"Created by <strong style='color:rgba(255,255,255,0.6)'>{AUTHOR_NAME}</strong>"
    f"</p>",
    unsafe_allow_html=True
)
