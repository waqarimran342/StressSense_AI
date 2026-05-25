"""
StressSense AI — Streamlit Web Application
Full-featured expert system interface for stress & mental wellbeing assessment
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup so sibling packages resolve correctly ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from knowledge_base.facts import (
    Fact,FactType,StressLevel,ConfidenceLevel,FactBase
)
from knowledge_base.rules import RuleBase
from inference.engine import InferenceEngine
from inference.fuzzy_logic import ( FuzzyRuleEvaluator, FuzzyMembership)
from inference.certainty_factors import CertaintyFactorCalculator
from explanation.explainer import StressExplainer

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #6C63FF;
    --primary-light: #8B85FF;
    --secondary: #FF6584;
    --accent: #43E97B;
    --warning: #F7B731;
    --danger: #FC5C65;
    --bg-dark: #0F1117;
    --bg-card: #1A1D2E;
    --bg-card-hover: #1F2235;
    --text-primary: #E8EAF6;
    --text-secondary: #9FA8DA;
    --text-muted: #5C6BC0;
    --border: rgba(108, 99, 255, 0.2);
    --border-hover: rgba(108, 99, 255, 0.5);
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    --radius: 14px;
    --radius-sm: 8px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-dark) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12152b 0%, #0F1117 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary-light) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Headers ── */
h1, h2, h3, h4 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* ── Cards ── */
.ss-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: var(--shadow);
}
.ss-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 12px 40px rgba(108, 99, 255, 0.15);
}

/* ── Hero Banner ── */
.ss-hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #12152b 50%, #1a1d2e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.ss-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.ss-hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #FF6584, #43E97B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.ss-hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* ── Section Labels ── */
.ss-section-label {
    display: inline-block;
    background: rgba(108, 99, 255, 0.15);
    color: var(--primary-light);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ── Metric Pills ── */
.ss-metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(108, 99, 255, 0.1);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0.2rem;
}

/* ── Stress Level Badge ── */
.ss-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.ss-badge-minimal { background: rgba(67,233,123,0.15); color: #43E97B; border: 1px solid rgba(67,233,123,0.4); }
.ss-badge-mild    { background: rgba(247,183,49,0.15); color: #F7B731; border: 1px solid rgba(247,183,49,0.4); }
.ss-badge-moderate{ background: rgba(255,101,52,0.15); color: #FF6534; border: 1px solid rgba(255,101,52,0.4); }
.ss-badge-high    { background: rgba(252,92,101,0.15); color: #FC5C65; border: 1px solid rgba(252,92,101,0.4); }
.ss-badge-severe  { background: rgba(152,0,0,0.2);     color: #FF4444; border: 1px solid rgba(152,0,0,0.5); }

/* ── Progress bars ── */
.ss-progress-wrap { margin: 0.3rem 0; }
.ss-progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.2rem;
}
.ss-progress-bar {
    height: 6px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}
.ss-progress-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.4s ease;
}

/* ── Symptom Chips ── */
.ss-chip-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.ss-chip {
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid;
}
.ss-chip-present { background: rgba(252,92,101,0.1); color: #FC5C65; border-color: rgba(252,92,101,0.35); }
.ss-chip-absent  { background: rgba(67,233,123,0.1); color: #43E97B; border-color: rgba(67,233,123,0.35); }
.ss-chip-stressor{ background: rgba(247,183,49,0.1); color: #F7B731; border-color: rgba(247,183,49,0.35); }
.ss-chip-protect { background: rgba(67,233,123,0.1); color: #43E97B; border-color: rgba(67,233,123,0.35); }

/* ── Reasoning step ── */
.ss-step {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    padding: 0.9rem 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--primary);
}
.ss-step-num {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--primary);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ss-step-body { flex: 1; }
.ss-step-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 0.2rem; }
.ss-step-desc  { font-size: 0.8rem; color: var(--text-secondary); }

/* ── Rule card ── */
.ss-rule-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid var(--primary);
}
.ss-rule-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    background: rgba(108,99,255,0.1);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    margin-right: 0.5rem;
}
.ss-rule-name { font-weight: 600; font-size: 0.88rem; }
.ss-rule-meta { font-size: 0.76rem; color: var(--text-secondary); margin-top: 0.5rem; }
.ss-rule-rationale {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    font-style: italic;
    border-top: 1px solid var(--border);
    padding-top: 0.4rem;
}

/* ── Recommendation card ── */
.ss-rec-card {
    background: rgba(67,233,123,0.05);
    border: 1px solid rgba(67,233,123,0.15);
    border-radius: var(--radius-sm);
    padding: 1rem;
    margin-bottom: 0.6rem;
}
.ss-rec-priority {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.ss-rec-p1 { color: #FC5C65; }
.ss-rec-p2 { color: #F7B731; }
.ss-rec-p3 { color: #43E97B; }

/* ── Divider ── */
.ss-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Streamlit widget overrides ── */
.stSlider > div > div > div { background: var(--primary) !important; }
.stCheckbox label { font-size: 0.88rem !important; color: var(--text-secondary) !important; }
.stCheckbox label:hover { color: var(--text-primary) !important; }
.stSelectbox label, .stMultiSelect label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }
div[data-baseweb="select"] { background: var(--bg-card) !important; border-color: var(--border) !important; }
div[data-baseweb="input"] { background: var(--bg-card) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.25rem !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary-light) !important;
    background: rgba(108,99,255,0.1) !important;
    border-bottom: 2px solid var(--primary) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1.5rem !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(108,99,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* Alerts */
.stAlert { border-radius: var(--radius-sm) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* Code / mono */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(108,99,255,0.15) !important;
    color: var(--primary-light) !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.82rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_COLOURS = {
    "minimal":  "#43E97B",
    "mild":     "#F7B731",
    "moderate": "#FF6534",
    "high":     "#FC5C65",
    "severe":   "#FF2244",
}
LEVEL_ICONS = {
    "minimal":  "🟢",
    "mild":     "🟡",
    "moderate": "🟠",
    "high":     "🔴",
    "severe":   "🚨",
}
DOMAIN_COLOURS = {
    "physical":       "#6C63FF",
    "psychological":  "#FF6584",
    "behavioral":     "#43E97B",
    "stressors":      "#F7B731",
    "protective":     "#3DD6F5",
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "assessment_done": False,
        "assessment_result": None,
        "user_inputs": {},
        "why_explanation": None,
        "how_explanation": None,
        "rec_explanation": None,
        "active_page": "assessment",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Engine initialisation (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    rule_base    = RuleBase()
    fuzzy        = FuzzyRuleEvaluator()
    cf_engine    = CertaintyFactorCalculator()
    engine       = InferenceEngine()
    explainer    = StressExplainer()
    return engine, explainer

engine, explainer = load_engine()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div style="font-size:2.8rem; margin-bottom:0.3rem;">🧠</div>
            <div style="font-size:1.15rem; font-weight:700; color:#E8EAF6;">StressSense AI</div>
            <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;
                        letter-spacing:0.12em; margin-top:0.2rem;">Expert System v1.0</div>
        </div>
        <hr style="border-color:rgba(108,99,255,0.2); margin-bottom:1.2rem;">
        """, unsafe_allow_html=True)

        pages = {
            "assessment":    ("📋", "Assessment"),
            "results":       ("📊", "Results"),
            "explanations":  ("🔍", "Explanations"),
            "recommendations":("💡", "Recommendations"),
            "knowledge_base":("📚", "Knowledge Base"),
            "about":         ("ℹ️",  "About"),
        }

        st.markdown("### Navigation")
        for key, (icon, label) in pages.items():
            disabled = (key in ("results", "explanations", "recommendations")
                        and not st.session_state.assessment_done)
            style = (
                "color:#6C63FF; font-weight:600; background:rgba(108,99,255,0.12);"
                if st.session_state.active_page == key
                else "color:#9FA8DA;"
            )
            if not disabled:
                btn = st.button(
                    f"{icon}  {label}",
                    key=f"nav_{key}",
                    use_container_width=True,
                )
                if btn:
                    st.session_state.active_page = key
                    st.rerun()
            else:
                st.markdown(
                    f'<div style="padding:0.5rem 0.8rem; font-size:0.85rem; '
                    f'color:#3a3f5c; cursor:not-allowed;">{icon}  {label} 🔒</div>',
                    unsafe_allow_html=True
                )

        if st.session_state.assessment_done:
            st.markdown("---")
            result = st.session_state.assessment_result or {}
            level  = result.get("stress_level", "unknown")
            colour = LEVEL_COLOURS.get(level, "#9FA8DA")
            icon   = LEVEL_ICONS.get(level, "❓")
            cf     = result.get("overall_cf", 0)
            st.markdown(f"""
            <div style="padding:1rem; background:rgba(108,99,255,0.06);
                        border:1px solid rgba(108,99,255,0.2); border-radius:10px;">
                <div style="font-size:0.7rem; color:#5C6BC0; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:0.4rem;">Last Assessment</div>
                <div style="font-size:1.1rem; font-weight:700; color:{colour};">
                    {icon} {level.title()} Stress
                </div>
                <div style="font-size:0.78rem; color:#9FA8DA; margin-top:0.2rem;">
                    Confidence: {cf:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="position:fixed; bottom:1.5rem; left:0; width:260px;
                    text-align:center; font-size:0.68rem; color:#3a3f5c;">
            ⚠️ Not a substitute for professional advice
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Assessment
# ─────────────────────────────────────────────────────────────────────────────
def page_assessment():
    st.markdown("""
    <div class="ss-hero">
        <div class="ss-hero-title">Stress & Wellbeing Assessment</div>
        <div class="ss-hero-subtitle">
            Answer honestly for the most accurate analysis. All responses are processed locally.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demographics
    with st.expander("👤  Personal Context  (optional but improves accuracy)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age_group = st.selectbox(
                "Age Group",
                ["Prefer not to say", "18–24", "25–34", "35–44", "45–54", "55+"],
                key="q_age"
            )
        with c2:
            occupation = st.selectbox(
                "Occupation Type",
                ["Prefer not to say", "Student", "Professional", "Self-employed",
                 "Caregiver", "Unemployed / Seeking work", "Retired"],
                key="q_occ"
            )
        with c3:
            stress_duration = st.selectbox(
                "How long have symptoms lasted?",
                ["Just started (<1 week)", "1–4 weeks", "1–3 months",
                 "3–6 months", "6+ months"],
                key="q_dur"
            )

    st.markdown("---")

    # ── Physical Symptoms ──────────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">🫀 Physical Symptoms</div>', unsafe_allow_html=True)
    st.markdown("**Which of the following physical symptoms are you currently experiencing?**")

    phys_cols = st.columns(2)
    physical_symptoms = {
        "headaches":           "Frequent Headaches",
        "muscle_tension":      "Muscle Tension / Tightness",
        "fatigue":             "Persistent Fatigue",
        "sleep_problems":      "Sleep Disturbances",
        "digestive_issues":    "Digestive Problems",
        "heart_racing":        "Racing Heart / Palpitations",
        "shortness_of_breath": "Shortness of Breath",
        "sweating":            "Excessive Sweating",
        "trembling":           "Trembling or Shaking",
        "dizziness":           "Dizziness / Lightheadedness",
    }
    phys_vals = {}
    items = list(physical_symptoms.items())
    for i, (key, label) in enumerate(items):
        col = phys_cols[i % 2]
        with col:
            phys_vals[key] = st.checkbox(label, key=f"phys_{key}")

    st.markdown("---")

    # ── Psychological Symptoms ─────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">🧠 Psychological Symptoms</div>', unsafe_allow_html=True)
    st.markdown("**Which emotional or cognitive experiences resonate with you?**")

    psych_cols = st.columns(2)
    psychological_symptoms = {
        "anxiety_worry":        "Persistent Anxiety / Worry",
        "mood_swings":          "Frequent Mood Swings",
        "irritability":         "Increased Irritability",
        "depression_feelings":  "Feelings of Depression",
        "overwhelmed":          "Feeling Overwhelmed",
        "cognitive_difficulties":"Difficulty Concentrating / Memory Issues",
        "negative_thinking":    "Negative / Catastrophic Thinking",
        "hopelessness":         "Sense of Hopelessness",
        "panic_attacks":        "Panic Attacks",
        "social_withdrawal":    "Withdrawing from Social Situations",
    }
    psych_vals = {}
    items = list(psychological_symptoms.items())
    for i, (key, label) in enumerate(items):
        col = psych_cols[i % 2]
        with col:
            psych_vals[key] = st.checkbox(label, key=f"psych_{key}")

    st.markdown("---")

    # ── Behavioral Changes ─────────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">🔄 Behavioral Changes</div>', unsafe_allow_html=True)
    st.markdown("**Have you noticed these changes in your behavior?**")

    beh_cols = st.columns(2)
    behavioral_symptoms = {
        "appetite_changes":          "Changes in Appetite",
        "procrastination":           "Increased Procrastination",
        "substance_use":             "More Alcohol / Caffeine Use",
        "poor_time_management":      "Poor Time Management",
        "neglecting_responsibilities":"Neglecting Responsibilities",
        "increased_errors":          "Making More Mistakes",
        "reduced_productivity":      "Reduced Productivity",
        "avoidance":                 "Avoiding People / Situations",
    }
    beh_vals = {}
    items = list(behavioral_symptoms.items())
    for i, (key, label) in enumerate(items):
        col = beh_cols[i % 2]
        with col:
            beh_vals[key] = st.checkbox(label, key=f"beh_{key}")

    st.markdown("---")

    # ── Stressors ──────────────────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">⚡ Active Stressors</div>', unsafe_allow_html=True)
    st.markdown("**Which of the following are currently causing you significant stress?**")

    str_cols = st.columns(2)
    stressors = {
        "work_overload":        "Work / Job Overload",
        "relationship_conflict":"Relationship Conflicts",
        "financial_stress":     "Financial Stress",
        "health_concerns":      "Health Concerns (self or others)",
        "major_life_changes":   "Major Life Changes",
        "academic_pressure":    "Academic Pressure",
        "social_isolation":     "Social Isolation",
        "trauma_history":       "History of Trauma",
        "chronic_illness":      "Chronic Illness",
        "caregiving_burden":    "Caregiving Responsibilities",
    }
    str_vals = {}
    items = list(stressors.items())
    for i, (key, label) in enumerate(items):
        col = str_cols[i % 2]
        with col:
            str_vals[key] = st.checkbox(label, key=f"str_{key}")

    st.markdown("---")

    # ── Protective Factors ─────────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">🛡️ Protective Factors</div>', unsafe_allow_html=True)
    st.markdown("**Which coping resources do you currently have access to?**")

    prot_cols = st.columns(2)
    protective_factors = {
        "social_support":       "Strong Social Support Network",
        "regular_exercise":     "Regular Exercise (3×/week+)",
        "healthy_diet":         "Generally Healthy Diet",
        "adequate_sleep":       "Adequate Sleep (7–9 hrs)",
        "mindfulness_practice": "Mindfulness / Meditation Practice",
        "hobbies_interests":    "Active Hobbies & Interests",
        "professional_help":    "Access to Mental Health Professional",
        "work_life_balance":    "Good Work-Life Balance",
        "relaxation_techniques":"Relaxation Techniques",
        "positive_outlook":     "Generally Positive Outlook",
    }
    prot_vals = {}
    items = list(protective_factors.items())
    for i, (key, label) in enumerate(items):
        col = prot_cols[i % 2]
        with col:
            prot_vals[key] = st.checkbox(label, key=f"prot_{key}")

    st.markdown("---")

    # ── Severity Sliders ───────────────────────────────────────────────────
    st.markdown('<div class="ss-section-label">📊 Overall Severity</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        overall_severity = st.slider(
            "Overall stress intensity (0 = none, 10 = unbearable)",
            0, 10, 5, key="q_overall_sev"
        )
    with s2:
        functional_impact = st.slider(
            "Impact on daily functioning (0 = no impact, 10 = cannot function)",
            0, 10, 4, key="q_func_impact"
        )

    st.markdown("---")

    # ── Submit ─────────────────────────────────────────────────────────────
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        submit = st.button("🔍  Run Assessment", use_container_width=True)
    with col_note:
        st.markdown(
            '<div style="padding:0.6rem 0; font-size:0.8rem; color:#5C6BC0;">'
            '⚠️ This tool is for educational purposes only. It does not replace '
            'professional medical or psychological evaluation.</div>',
            unsafe_allow_html=True
        )

    if submit:
        user_inputs = {
            **phys_vals, **psych_vals, **beh_vals,
            **str_vals, **prot_vals,
            "overall_severity":   overall_severity,
            "functional_impact":  functional_impact,
            "stress_duration":    stress_duration,
            "age_group":          age_group,
            "occupation":         occupation,
        }
        run_assessment(user_inputs)

# ─────────────────────────────────────────────────────────────────────────────
# Run assessment logic
# ─────────────────────────────────────────────────────────────────────────────
def run_assessment(user_inputs: Dict):
    with st.spinner("🧠  Analysing your responses…"):
        try:
            result = engine.run(user_inputs)
            explainer.clear_logs()
            why = explainer.generate_why_explanation(result, user_inputs)
            how = explainer.generate_how_explanation(result, user_inputs)
            recs = result.get("recommendations", [])
            rec_exp = explainer.generate_recommendation_explanation(
                result.get("stress_level", "moderate"),
                recs,
                user_inputs
            )
            st.session_state.assessment_result  = result
            st.session_state.user_inputs        = user_inputs
            st.session_state.why_explanation    = why
            st.session_state.how_explanation    = how
            st.session_state.rec_explanation    = rec_exp
            st.session_state.assessment_done    = True
            st.session_state.active_page        = "results"
            st.success("✅  Assessment complete!")
            st.rerun()
        except Exception as e:
            st.error(f"Assessment error: {e}")
            with st.expander("Debug info"):
                import traceback
                st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# Page: Results
# ─────────────────────────────────────────────────────────────────────────────
def page_results():
    result = st.session_state.assessment_result
    inputs = st.session_state.user_inputs
    if not result:
        st.warning("No assessment results yet. Please complete the assessment first.")
        return

    level      = result.get("stress_level", "unknown")
    overall_cf = result.get("overall_cf", 0.0)
    fuzzy      = result.get("fuzzy_score", 0.0)
    domain_sc  = result.get("domain_scores", {})
    fired      = result.get("fired_rules", [])
    colour     = LEVEL_COLOURS.get(level, "#9FA8DA")
    icon       = LEVEL_ICONS.get(level, "❓")

    # ── Hero result card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ss-card" style="border-color:{colour}33; background:linear-gradient(135deg,
         rgba(108,99,255,0.04) 0%, rgba(0,0,0,0) 100%);">
        <div style="display:flex; align-items:center; gap:2rem; flex-wrap:wrap;">
            <div style="flex:0 0 auto; text-align:center;">
                <div style="font-size:4rem; line-height:1;">{icon}</div>
                <div class="ss-badge ss-badge-{level}" style="margin-top:0.5rem;">{level}</div>
            </div>
            <div style="flex:1;">
                <div style="font-size:0.75rem; color:#5C6BC0; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:0.3rem;">Assessment Result</div>
                <div style="font-size:2rem; font-weight:700; color:{colour}; line-height:1.2;">
                    {level.title()} Stress Level
                </div>
                <div style="font-size:0.9rem; color:#9FA8DA; margin-top:0.5rem;">
                    {_level_description(level)}
                </div>
                <div style="display:flex; gap:1rem; margin-top:1rem; flex-wrap:wrap;">
                    <span class="ss-metric-pill">🎯 CF: {overall_cf:.0%}</span>
                    <span class="ss-metric-pill">🌊 Fuzzy: {fuzzy:.2f}</span>
                    <span class="ss-metric-pill">⚡ {len(fired)} rules fired</span>
                    <span class="ss-metric-pill">📊 {len(domain_sc)} domains</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    st.markdown("### 📊 Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Domain Scores", "Radar Chart", "Score Breakdown"])

    with tab1:
        render_domain_bar_chart(domain_sc, colour)

    with tab2:
        render_radar_chart(domain_sc)

    with tab3:
        render_score_breakdown(result, inputs)

    # ── Symptom evidence ──────────────────────────────────────────────────
    st.markdown("### 🔬 Symptom Evidence")
    ec1, ec2, ec3 = st.columns(3)

    phys_present = [k for k, v in inputs.items()
                    if v and k in {"headaches","muscle_tension","fatigue","sleep_problems",
                                   "digestive_issues","heart_racing","shortness_of_breath",
                                   "sweating","trembling","dizziness"}]
    psyc_present = [k for k, v in inputs.items()
                    if v and k in {"anxiety_worry","mood_swings","irritability",
                                   "depression_feelings","overwhelmed","cognitive_difficulties",
                                   "negative_thinking","hopelessness","panic_attacks","social_withdrawal"}]
    beha_present = [k for k, v in inputs.items()
                    if v and k in {"appetite_changes","procrastination","substance_use",
                                   "poor_time_management","neglecting_responsibilities",
                                   "increased_errors","reduced_productivity","avoidance"}]

    label_map = {
        "headaches":"Headaches","muscle_tension":"Muscle Tension","fatigue":"Fatigue",
        "sleep_problems":"Sleep Issues","digestive_issues":"Digestive","heart_racing":"Heart Racing",
        "shortness_of_breath":"Breathlessness","sweating":"Sweating","trembling":"Trembling",
        "dizziness":"Dizziness","anxiety_worry":"Anxiety","mood_swings":"Mood Swings",
        "irritability":"Irritability","depression_feelings":"Depression","overwhelmed":"Overwhelmed",
        "cognitive_difficulties":"Cognition","negative_thinking":"Neg. Thinking",
        "hopelessness":"Hopelessness","panic_attacks":"Panic","social_withdrawal":"Withdrawal",
        "appetite_changes":"Appetite","procrastination":"Procrastination","substance_use":"Substance Use",
        "poor_time_management":"Time Mgmt","neglecting_responsibilities":"Neglect",
        "increased_errors":"More Errors","reduced_productivity":"Low Productivity","avoidance":"Avoidance",
    }

    with ec1:
        st.markdown(f"**🫀 Physical** ({len(phys_present)})")
        chips = "".join(
            f'<span class="ss-chip ss-chip-present">{label_map.get(s, s)}</span>'
            for s in phys_present
        ) or '<span class="ss-chip ss-chip-absent">None reported</span>'
        st.markdown(f'<div class="ss-chip-grid">{chips}</div>', unsafe_allow_html=True)

    with ec2:
        st.markdown(f"**🧠 Psychological** ({len(psyc_present)})")
        chips = "".join(
            f'<span class="ss-chip ss-chip-present">{label_map.get(s, s)}</span>'
            for s in psyc_present
        ) or '<span class="ss-chip ss-chip-absent">None reported</span>'
        st.markdown(f'<div class="ss-chip-grid">{chips}</div>', unsafe_allow_html=True)

    with ec3:
        st.markdown(f"**🔄 Behavioral** ({len(beha_present)})")
        chips = "".join(
            f'<span class="ss-chip ss-chip-present">{label_map.get(s, s)}</span>'
            for s in beha_present
        ) or '<span class="ss-chip ss-chip-absent">None reported</span>'
        st.markdown(f'<div class="ss-chip-grid">{chips}</div>', unsafe_allow_html=True)

    # ── Stressors & Protective row ─────────────────────────────────────────
    sc1, sc2 = st.columns(2)

    stressor_keys = {"work_overload","relationship_conflict","financial_stress","health_concerns",
                     "major_life_changes","academic_pressure","social_isolation",
                     "trauma_history","chronic_illness","caregiving_burden"}
    protect_keys  = {"social_support","regular_exercise","healthy_diet","adequate_sleep",
                     "mindfulness_practice","hobbies_interests","professional_help",
                     "work_life_balance","relaxation_techniques","positive_outlook"}

    str_labels = {
        "work_overload":"Work Overload","relationship_conflict":"Relationship Conflict",
        "financial_stress":"Financial Stress","health_concerns":"Health Concerns",
        "major_life_changes":"Life Changes","academic_pressure":"Academic Pressure",
        "social_isolation":"Social Isolation","trauma_history":"Trauma History",
        "chronic_illness":"Chronic Illness","caregiving_burden":"Caregiving",
    }
    prot_labels = {
        "social_support":"Social Support","regular_exercise":"Exercise",
        "healthy_diet":"Healthy Diet","adequate_sleep":"Good Sleep",
        "mindfulness_practice":"Mindfulness","hobbies_interests":"Hobbies",
        "professional_help":"Professional Help","work_life_balance":"Work-Life Balance",
        "relaxation_techniques":"Relaxation","positive_outlook":"Positive Outlook",
    }

    active_stressors   = [k for k, v in inputs.items() if v and k in stressor_keys]
    active_protectives = [k for k, v in inputs.items() if v and k in protect_keys]

    with sc1:
        st.markdown(f"**⚡ Active Stressors** ({len(active_stressors)})")
        chips = "".join(
            f'<span class="ss-chip ss-chip-stressor">{str_labels.get(s, s)}</span>'
            for s in active_stressors
        ) or '<span class="ss-chip ss-chip-absent">None identified</span>'
        st.markdown(f'<div class="ss-chip-grid">{chips}</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown(f"**🛡️ Active Protectives** ({len(active_protectives)})")
        chips = "".join(
            f'<span class="ss-chip ss-chip-protect">{prot_labels.get(s, s)}</span>'
            for s in active_protectives
        ) or '<span class="ss-chip" style="color:#FC5C65;">None identified</span>'
        st.markdown(f'<div class="ss-chip-grid">{chips}</div>', unsafe_allow_html=True)


def _level_description(level: str) -> str:
    descs = {
        "minimal":  "Your stress indicators are within healthy ranges. Keep up your current self-care practices.",
        "mild":     "You're experiencing some stress. Early attention to coping strategies is beneficial.",
        "moderate": "Notable stress present across multiple domains. Active stress management is recommended.",
        "high":     "Significant stress that is likely affecting your daily functioning and health.",
        "severe":   "Severe stress indicators detected. Professional support is strongly recommended.",
    }
    return descs.get(level, "Stress assessment complete.")


def render_domain_bar_chart(domain_scores: Dict, accent_colour: str):
    if not domain_scores:
        st.info("No domain data available.")
        return

    domains = list(domain_scores.keys())
    scores  = [domain_scores[d] for d in domains]
    colours = [DOMAIN_COLOURS.get(d, "#6C63FF") for d in domains]
    labels  = [d.replace("_", " ").title() for d in domains]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(color=colours, opacity=0.85),
        text=[f"{s:.0%}" for s in scores],
        textposition="auto",
        textfont=dict(color="#E8EAF6", family="Sora", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9FA8DA", family="Sora"),
        margin=dict(l=10, r=20, t=10, b=10),
        height=300,
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.05)",
            title="Score",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(domain_scores: Dict):
    if not domain_scores:
        st.info("No domain data available.")
        return

    cats   = [d.replace("_", " ").title() for d in domain_scores]
    vals   = list(domain_scores.values())
    cats  += [cats[0]]
    vals  += [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(108,99,255,0.15)",
        line=dict(color="#6C63FF", width=2),
        marker=dict(color="#6C63FF", size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.08)",
                            linecolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#5C6BC0", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             linecolor="rgba(255,255,255,0.15)",
                             tickfont=dict(color="#9FA8DA", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9FA8DA", family="Sora"),
        margin=dict(l=30, r=30, t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_score_breakdown(result: Dict, inputs: Dict):
    cf    = result.get("overall_cf", 0)
    fuzzy = result.get("fuzzy_score", 0)
    sev   = inputs.get("overall_severity", 5) / 10
    func  = inputs.get("functional_impact", 4) / 10
    composite = (cf * 0.4 + fuzzy * 0.35 + sev * 0.15 + func * 0.10)

    rows = [
        ("Certainty Factor (CF)", cf, "#6C63FF", 0.40),
        ("Fuzzy Score",           fuzzy, "#FF6584", 0.35),
        ("Self-reported Severity",sev,  "#F7B731", 0.15),
        ("Functional Impact",     func,  "#43E97B", 0.10),
        ("Composite Score",       composite, "#3DD6F5", 1.0),
    ]

    for name, val, colour, weight in rows:
        pct = int(val * 100)
        st.markdown(f"""
        <div class="ss-progress-wrap">
            <div class="ss-progress-label">
                <span>{name}</span>
                <span style="color:{colour}; font-weight:600;">{pct}%</span>
            </div>
            <div class="ss-progress-bar">
                <div class="ss-progress-fill"
                     style="width:{pct}%; background:{colour};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Explanations
# ─────────────────────────────────────────────────────────────────────────────
def page_explanations():
    why = st.session_state.why_explanation
    how = st.session_state.how_explanation
    result = st.session_state.assessment_result

    if not why or not result:
        st.warning("Please complete the assessment first.")
        return

    st.markdown("""
    <div class="ss-hero">
        <div class="ss-hero-title">Explanation Centre</div>
        <div class="ss-hero-subtitle">
            Understand the reasoning behind your assessment in full transparency.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_why, tab_how, tab_cf, tab_fuzzy = st.tabs([
        "🤔 WHY — Justification",
        "⚙️ HOW — Derivation",
        "📐 Certainty Factors",
        "〰️ Fuzzy Logic"
    ])

    # ── WHY tab ──────────────────────────────────────────────────────────
    with tab_why:
        conclusion = why.get("conclusion", {})
        level   = conclusion.get("stress_level", "unknown")
        cf      = conclusion.get("certainty_factor", 0)
        cl_lbl  = conclusion.get("confidence_label", "")
        colour  = LEVEL_COLOURS.get(level, "#9FA8DA")

        st.markdown(f"""
        <div class="ss-card" style="border-color:{colour}44;">
            <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;
                        letter-spacing:0.1em;">Why this conclusion?</div>
            <div style="font-size:1.5rem; font-weight:700; color:{colour}; margin:0.3rem 0;">
                {level.title()} Stress — {cf:.0%} confidence
            </div>
            <div style="font-size:0.83rem; color:#9FA8DA;">{cl_lbl}</div>
        </div>
        """, unsafe_allow_html=True)

        # Narrative
        narrative = why.get("narrative", "")
        if narrative:
            st.markdown(f"""
            <div class="ss-card">
                <div class="ss-section-label">📖 Narrative Explanation</div>
                <div style="font-size:0.88rem; color:#C5CAE9; line-height:1.75;">
                    {narrative.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Reasoning chain
        chain = why.get("reasoning_chain", [])
        if chain:
            st.markdown("#### 🔗 Reasoning Chain")
            for step in chain:
                type_colour = {
                    "data_collection":  "#6C63FF",
                    "domain_analysis":  "#FF6584",
                    "rule_application": "#F7B731",
                    "cf_combination":   "#43E97B",
                    "fuzzy_aggregation":"#3DD6F5",
                    "conclusion":       "#FFFFFF",
                }.get(step.get("type", ""), "#9FA8DA")

                detail = ""
                if "score" in step:
                    detail = f' <span style="color:{type_colour}; font-weight:600;">{step["score"]:.0%}</span>'
                elif "cf" in step and step.get("type") in ("cf_combination", "conclusion"):
                    detail = f' <span style="color:{type_colour}; font-weight:600;">CF={step["cf"]:.0%}</span>'
                elif "fuzzy_score" in step:
                    detail = f' <span style="color:{type_colour}; font-weight:600;">{step["fuzzy_score"]:.2f}</span>'

                st.markdown(f"""
                <div class="ss-step">
                    <div class="ss-step-num">{step.get("step", "")}</div>
                    <div class="ss-step-body">
                        <div class="ss-step-title" style="color:{type_colour};">
                            {step.get("title","")} {detail}
                        </div>
                        <div class="ss-step-desc">{step.get("description","")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Key factors
        key_factors = why.get("key_contributing_factors", [])
        if key_factors:
            st.markdown("#### 🎯 Key Contributing Factors")
            for f in key_factors:
                impact = f.get("impact", "")
                impact_colour = {"highest":"#FC5C65", "high":"#F7B731",
                                 "amplifying":"#FF6584"}.get(impact, "#9FA8DA")
                st.markdown(f"""
                <div class="ss-card">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span class="ss-badge" style="background:rgba(255,255,255,0.04);
                              color:{impact_colour}; border-color:{impact_colour}44;
                              font-size:0.7rem;">{impact.upper()}</span>
                        <strong style="font-size:0.9rem;">{f.get("factor","")}</strong>
                    </div>
                    <div style="font-size:0.8rem; color:#9FA8DA; margin-top:0.5rem;">
                        {f.get("description","")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── HOW tab ───────────────────────────────────────────────────────────
    with tab_how:
        method = how.get("inference_method", {})
        steps  = how.get("derivation_steps", [])

        st.markdown(f"""
        <div class="ss-card">
            <div class="ss-section-label">⚙️ Method</div>
            <div style="font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                {method.get("name","")}
            </div>
            <div style="font-size:0.83rem; color:#9FA8DA; line-height:1.65;">
                {method.get("description","")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Steps summary
        summary = method.get("steps_summary", [])
        if summary:
            with st.expander("📋 Algorithm Overview"):
                for s in summary:
                    st.markdown(f"- {s}")

        # Rule-by-rule derivation
        st.markdown(f"#### ⚡ Rule Derivation ({len(steps) - 1} rules fired)")
        for step in steps:
            if step.get("type") == "initial_facts":
                facts = step.get("facts", [])
                total = step.get("facts_total", len(facts))
                extra = f" (+{total - len(facts)} more)" if total > len(facts) else ""
                st.markdown(f"""
                <div class="ss-card">
                    <div class="ss-section-label">🗂️ Working Memory</div>
                    <div style="font-size:0.83rem; color:#9FA8DA; margin-bottom:0.5rem;">
                        {total} facts established from user input{extra}:
                    </div>
                    <div class="ss-chip-grid">
                        {''.join(f'<span class="ss-chip ss-chip-present">{f}</span>' for f in facts)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif step.get("type") == "rule_firing":
                rule_id  = step.get("rule_id", "")
                cf_val   = step.get("cf", 0)
                ev_type  = step.get("evidence_type", "general")
                ev_colour= DOMAIN_COLOURS.get(ev_type, "#9FA8DA")
                rationale= step.get("rationale", "")
                cb       = step.get("confidence_basis", "")
                conditions = step.get("conditions", [])
                conclusion_txt = step.get("conclusion", "")

                st.markdown(f"""
                <div class="ss-rule-card">
                    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
                        <span class="ss-rule-id">{rule_id}</span>
                        <span class="ss-rule-name">{step.get("title","")}</span>
                        <span style="margin-left:auto; font-size:0.8rem; font-weight:700;
                               color:{ev_colour};">{cf_val:.0%}</span>
                    </div>
                    <div class="ss-rule-meta">
                        <span style="color:{ev_colour};">● {ev_type.title()}</span>
                        {"  |  IF " + " AND ".join(conditions[:3]) if conditions else ""}
                        {"  →  " + conclusion_txt if conclusion_txt else ""}
                    </div>
                    {f'<div class="ss-rule-rationale">{rationale}</div>' if rationale else ""}
                    {f'<div style="font-size:0.72rem; color:#3a4a5c; margin-top:0.3rem; font-style:italic;">📚 {cb}</div>' if cb else ""}
                </div>
                """, unsafe_allow_html=True)

    # ── Certainty Factors tab ─────────────────────────────────────────────
    with tab_cf:
        cf_data = how.get("certainty_factor_explanation", {})
        st.markdown(f"""
        <div class="ss-card">
            <div class="ss-section-label">📐 Certainty Factor Algebra</div>
            <div style="display:flex; gap:2rem; flex-wrap:wrap; margin-bottom:1rem;">
                <div>
                    <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;">Final CF</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#6C63FF;">
                        {cf_data.get("final_cf",0):.0%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;">Rules Contributing</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#FF6584;">
                        {cf_data.get("rules_contributing",0)}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;">Avg Rule CF</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#F7B731;">
                        {cf_data.get("average_rule_cf",0):.0%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.72rem; color:#5C6BC0; text-transform:uppercase;">Peak Rule CF</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#43E97B;">
                        {cf_data.get("highest_rule_cf",0):.0%}
                    </div>
                </div>
            </div>
            <div style="font-size:0.83rem; color:#9FA8DA;">{cf_data.get("interpretation","")}</div>
        </div>
        """, unsafe_allow_html=True)

        formula = cf_data.get("combination_formula", "")
        if formula:
            st.markdown("**CF Combination Formula:**")
            st.code(formula, language="text")

    # ── Fuzzy Logic tab ───────────────────────────────────────────────────
    with tab_fuzzy:
        fz = how.get("fuzzy_logic_explanation", {})
        memberships = fz.get("membership_degrees", {})

        st.markdown(f"""
        <div class="ss-card">
            <div class="ss-section-label">〰️ Fuzzy Membership Degrees</div>
            <div style="font-size:0.83rem; color:#9FA8DA; margin-bottom:1rem;">
                {fz.get("interpretation","")}
            </div>
        """, unsafe_allow_html=True)

        level_colours = {
            "minimal": "#43E97B", "mild": "#F7B731", "moderate": "#FF6534",
            "high": "#FC5C65", "severe": "#FF2244"
        }
        for lvl, deg in memberships.items():
            pct = int(deg * 100)
            col = level_colours.get(lvl, "#9FA8DA")
            st.markdown(f"""
            <div class="ss-progress-wrap">
                <div class="ss-progress-label">
                    <span style="text-transform:capitalize;">{lvl}</span>
                    <span style="color:{col}; font-weight:600;">{pct}%</span>
                </div>
                <div class="ss-progress-bar">
                    <div class="ss-progress-fill" style="width:{pct}%; background:{col};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        dom_contrib = fz.get("domain_contributions", {})
        if dom_contrib:
            st.markdown("**Domain Contributions to Fuzzy Score:**")
            df = pd.DataFrame(
                [(d.replace("_"," ").title(), v) for d, v in dom_contrib.items()],
                columns=["Domain", "Score"]
            )
            fig = px.bar(
                df, x="Score", y="Domain", orientation="h",
                color="Score", color_continuous_scale=["#1A1D2E","#6C63FF","#FF6584"],
                text="Score"
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="auto")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9FA8DA", family="Sora"),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=5, b=5), height=280,
                xaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Recommendations
# ─────────────────────────────────────────────────────────────────────────────
def page_recommendations():
    rec_exp = st.session_state.rec_explanation
    result  = st.session_state.assessment_result
    if not rec_exp or not result:
        st.warning("Please complete the assessment first.")
        return

    level   = rec_exp.get("stress_level", "unknown")
    recs    = rec_exp.get("recommendations", [])
    rationale= rec_exp.get("general_rationale", "")
    colour  = LEVEL_COLOURS.get(level, "#9FA8DA")

    st.markdown(f"""
    <div class="ss-hero">
        <div class="ss-hero-title">Personalised Recommendations</div>
        <div class="ss-hero-subtitle">
            Evidence-based interventions tailored to your <span style="color:{colour};">
            {level.title()} Stress</span> profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rationale:
        st.markdown(f"""
        <div class="ss-card">
            <div class="ss-section-label">💡 Why these recommendations?</div>
            <div style="font-size:0.88rem; color:#9FA8DA; line-height:1.7;">{rationale}</div>
        </div>
        """, unsafe_allow_html=True)

    # Priority groups
    p1 = [r for r in recs if r.get("priority") == 1]
    p2 = [r for r in recs if r.get("priority") == 2]
    p3 = [r for r in recs if r.get("priority") == 3]

    def render_rec_group(title: str, items: List, priority: int, colour: str, icon: str):
        if not items:
            return
        st.markdown(f"#### {icon} {title}")
        for rec in items:
            r_text    = rec.get("recommendation", "")
            r_rat     = rec.get("rationale", "")
            r_ev      = rec.get("evidence_basis", "")
            st.markdown(f"""
            <div class="ss-rec-card">
                <div style="display:flex; align-items:flex-start; gap:0.75rem;">
                    <span style="font-size:1.2rem; flex-shrink:0;">{icon}</span>
                    <div style="flex:1;">
                        <div style="font-weight:600; font-size:0.92rem; color:#E8EAF6;
                                    margin-bottom:0.3rem;">{r_text}</div>
                        <div style="font-size:0.8rem; color:#9FA8DA; margin-bottom:0.3rem;
                                    line-height:1.55;">{r_rat}</div>
                        {f'<div style="font-size:0.72rem; color:#5C6BC0; font-style:italic;">📚 {r_ev}</div>' if r_ev else ""}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    render_rec_group("Priority — Seek Professional Support",   p1, 1, "#FC5C65", "🚨")
    render_rec_group("Important — Active Coping Strategies",   p2, 2, "#F7B731", "⚡")
    render_rec_group("Supportive — Lifestyle & Resilience",    p3, 3, "#43E97B", "🌿")

    # Emergency resources
    if level in ("high", "severe"):
        st.markdown("---")
        st.markdown("""
        <div class="ss-card" style="border-color:rgba(252,92,101,0.4); background:rgba(252,92,101,0.05);">
            <div style="font-size:0.9rem; font-weight:700; color:#FC5C65; margin-bottom:0.75rem;">
                🆘 Crisis Resources
            </div>
            <div style="font-size:0.83rem; color:#C5CAE9; line-height:1.8;">
                • <strong>International Association for Suicide Prevention:</strong>
                  <a href="https://www.iasp.info/resources/Crisis_Centres/"
                     style="color:#6C63FF;">Crisis Centres Directory</a><br>
                • <strong>Crisis Text Line (US):</strong> Text HOME to 741741<br>
                • <strong>Samaritans (UK/Ireland):</strong> 116 123<br>
                • <strong>Befrienders Worldwide:</strong>
                  <a href="https://www.befrienders.org" style="color:#6C63FF;">befrienders.org</a><br>
                • <strong>Your local emergency services:</strong> Contact if you feel unsafe
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
def page_knowledge_base():
    st.markdown("""
    <div class="ss-hero">
        <div class="ss-hero-title">Knowledge Base</div>
        <div class="ss-hero-subtitle">
            Explore the expert rules, facts ontology, and inference mechanisms powering StressSense AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_rules, tab_facts, tab_inference = st.tabs([
        "📜 Inference Rules", "🗂️ Facts Ontology", "🔧 Inference Engine"
    ])

    with tab_rules:
        rule_explanations = explainer.rule_explanations
        domain_filter = st.selectbox(
            "Filter by evidence type",
            ["All", "physical", "psychological", "behavioral", "environmental",
             "protective", "synthesis"],
            key="kb_filter"
        )
        shown = 0
        for rule_id, info in rule_explanations.items():
            ev_type = info.get("evidence_type", "general")
            if domain_filter != "All" and ev_type != domain_filter:
                continue
            col = DOMAIN_COLOURS.get(ev_type, "#9FA8DA")
            st.markdown(f"""
            <div class="ss-rule-card" style="border-left-color:{col};">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
                    <span class="ss-rule-id">{rule_id}</span>
                    <span class="ss-rule-name">{info.get("name","")}</span>
                    <span style="margin-left:auto; font-size:0.75rem; color:{col};
                           background:rgba(255,255,255,0.03); border-radius:4px;
                           padding:0.1rem 0.4rem; border:1px solid {col}44;">
                        {ev_type.title()}
                    </span>
                </div>
                <div class="ss-rule-rationale" style="color:#9FA8DA;">
                    {info.get("rationale","")}
                </div>
                <div style="font-size:0.72rem; color:#3a4a5c; margin-top:0.35rem; font-style:italic;">
                    📚 {info.get("confidence_basis","")}
                </div>
            </div>
            """, unsafe_allow_html=True)
            shown += 1

        st.markdown(f"<div style='font-size:0.78rem; color:#5C6BC0; margin-top:0.5rem;'>"
                    f"Showing {shown} of {len(rule_explanations)} rules</div>",
                    unsafe_allow_html=True)

    with tab_facts:
        st.markdown("#### Fact Classes (Ontology)")

        fact_classes = {
            "StressFact": {
                "description": "Base class for all facts in the system",
                "attributes": ["name", "value", "certainty_factor", "timestamp", "source"],
                "colour": "#6C63FF"
            },
            "PhysicalSymptomFact": {
                "description": "Somatic / physical stress manifestations",
                "attributes": ["symptom_type", "severity", "frequency", "duration"],
                "colour": "#FF6584"
            },
            "PsychologicalSymptomFact": {
                "description": "Emotional and cognitive stress indicators",
                "attributes": ["symptom_type", "intensity", "duration", "impact_on_functioning"],
                "colour": "#F7B731"
            },
            "BehavioralSymptomFact": {
                "description": "Observable behavioural changes due to stress",
                "attributes": ["behavior_type", "frequency", "change_magnitude"],
                "colour": "#43E97B"
            },
            "StressorFact": {
                "description": "External or internal sources of stress",
                "attributes": ["stressor_type", "intensity", "controllability", "duration"],
                "colour": "#FC5C65"
            },
            "ProtectiveFactorFact": {
                "description": "Resources that buffer against stress",
                "attributes": ["factor_type", "availability", "effectiveness"],
                "colour": "#3DD6F5"
            },
            "DemographicFact": {
                "description": "Contextual information about the user",
                "attributes": ["age_group", "occupation", "life_stage"],
                "colour": "#9FA8DA"
            },
        }

        for cls_name, info in fact_classes.items():
            col = info["colour"]
            attrs = " | ".join(f'<code>{a}</code>' for a in info["attributes"])
            st.markdown(f"""
            <div class="ss-rule-card" style="border-left-color:{col};">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.88rem;
                           color:{col}; font-weight:600;">{cls_name}</span>
                </div>
                <div style="font-size:0.8rem; color:#9FA8DA; margin-bottom:0.4rem;">
                    {info["description"]}
                </div>
                <div style="font-size:0.78rem; color:#5C6BC0;">Attributes: {attrs}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_inference:
        st.markdown("""
        <div class="ss-card">
            <div class="ss-section-label">🔧 Inference Architecture</div>
            <div style="font-size:0.88rem; color:#9FA8DA; line-height:1.75;">
                StressSense AI uses a <strong style="color:#6C63FF;">hybrid inference architecture</strong>
                combining three complementary mechanisms:
            </div>
        </div>
        """, unsafe_allow_html=True)

        components = [
            {
                "title": "Forward Chaining Engine",
                "icon": "➡️",
                "colour": "#6C63FF",
                "desc": (
                    "Data-driven reasoning that starts with known facts (user inputs) "
                    "and applies rules to derive new conclusions. The engine iterates "
                    "until no new facts can be inferred (fixed-point semantics)."
                ),
                "details": ["65+ hand-crafted clinical rules", "Pattern matching on working memory",
                            "Conflict resolution via salience", "Chained rule firing"]
            },
            {
                "title": "Fuzzy Logic Processor",
                "icon": "〰️",
                "colour": "#FF6584",
                "desc": (
                    "Handles the inherent vagueness in stress assessment. Rather than "
                    "binary yes/no, fuzzy membership functions map symptom clusters to "
                    "degrees of stress severity, capturing the continuous nature of stress."
                ),
                "details": ["Triangular & trapezoidal membership functions",
                            "5 linguistic stress levels", "Mamdani inference model",
                            "Centroid defuzzification"]
            },
            {
                "title": "Certainty Factor Engine",
                "icon": "📐",
                "colour": "#43E97B",
                "desc": (
                    "Adapted from the MYCIN expert system, CFs quantify the degree of "
                    "belief in conclusions given uncertain evidence. CFs propagate through "
                    "rule chains and combine using established algebraic formulas."
                ),
                "details": ["CF range: [-1.0, +1.0]", "Sequential combination algebra",
                            "Threshold-based conclusion acceptance", "Evidence accumulation"]
            },
        ]

        for comp in components:
            col = comp["colour"]
            details = "".join(f'<li style="font-size:0.78rem; color:#9FA8DA;">{d}</li>'
                              for d in comp["details"])
            st.markdown(f"""
            <div class="ss-card" style="border-left:3px solid {col};">
                <div style="display:flex; gap:0.75rem; align-items:flex-start;">
                    <span style="font-size:1.5rem;">{comp["icon"]}</span>
                    <div>
                        <div style="font-size:1rem; font-weight:700; color:{col};
                                    margin-bottom:0.4rem;">{comp["title"]}</div>
                        <div style="font-size:0.83rem; color:#C5CAE9; line-height:1.6;
                                    margin-bottom:0.5rem;">{comp["desc"]}</div>
                        <ul style="margin:0; padding-left:1.2rem;">{details}</ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("""
    <div class="ss-hero">
        <div class="ss-hero-title">About StressSense AI</div>
        <div class="ss-hero-subtitle">
            An expert system for stress and mental wellbeing assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="ss-card">
            <div class="ss-section-label">🎯 Purpose</div>
            <div style="font-size:0.88rem; color:#9FA8DA; line-height:1.75;">
                StressSense AI is a knowledge-based expert system designed to assess stress levels
                and provide evidence-based insights. It combines classical AI techniques —
                forward chaining inference, fuzzy logic, and certainty factors — with clinical
                knowledge to offer transparent, explainable assessments.
            </div>
        </div>
        <div class="ss-card">
            <div class="ss-section-label">🔬 Methodology</div>
            <div style="font-size:0.88rem; color:#9FA8DA; line-height:1.75;">
                The knowledge base was developed through systematic review of clinical literature
                including the PSS (Perceived Stress Scale), DASS-21, and occupational stress
                frameworks. Rules encode expert clinical knowledge with associated certainty factors
                reflecting evidence strength.
            </div>
        </div>
        <div class="ss-card" style="border-color:rgba(252,92,101,0.3);">
            <div class="ss-section-label" style="background:rgba(252,92,101,0.1);
                 color:#FC5C65; border-color:rgba(252,92,101,0.3);">⚠️ Disclaimer</div>
            <div style="font-size:0.83rem; color:#9FA8DA; line-height:1.7;">
                This system is for <strong>educational and informational purposes only</strong>.
                It is <strong>not</strong> a medical device, clinical assessment tool, or
                substitute for professional psychological or medical evaluation.
                If you are experiencing significant distress, please consult a qualified
                mental health professional.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="ss-card">
            <div class="ss-section-label">📦 System Stats</div>
            <div style="font-size:0.82rem; color:#9FA8DA; line-height:2;">
                🧠 <strong>65+</strong> inference rules<br>
                🗂️ <strong>7</strong> fact classes<br>
                📊 <strong>5</strong> stress level categories<br>
                🔬 <strong>3</strong> inference mechanisms<br>
                🌊 <strong>Fuzzy</strong> membership functions<br>
                📐 <strong>CF</strong> certainty propagation<br>
                🔗 <strong>Forward</strong> chaining engine<br>
                💡 <strong>WHY / HOW</strong> explanations
            </div>
        </div>
        <div class="ss-card">
            <div class="ss-section-label">📚 References</div>
            <div style="font-size:0.78rem; color:#5C6BC0; line-height:1.8;">
                • Cohen et al. (1983) — PSS Scale<br>
                • Lazarus & Folkman (1984) — Stress Theory<br>
                • Karasek (1979) — Demand-Control Model<br>
                • Kabat-Zinn (1990) — MBSR<br>
                • Shortliffe (1976) — MYCIN / CF<br>
                • Zadeh (1965) — Fuzzy Logic<br>
                • DSM-5 — Diagnostic criteria
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    page = st.session_state.active_page

    if page == "assessment":
        page_assessment()
    elif page == "results":
        page_results()
    elif page == "explanations":
        page_explanations()
    elif page == "recommendations":
        page_recommendations()
    elif page == "knowledge_base":
        page_knowledge_base()
    elif page == "about":
        page_about()
    else:
        page_assessment()

if __name__ == "__main__":
    main()