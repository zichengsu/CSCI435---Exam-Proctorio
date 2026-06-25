"""
app.py -- ARGUS | Intelligent Exam Integrity Monitoring
=======================================================
Streamlit frontend, custom UI/CSS, SQLite database & module integration.

ARGUS (the hundred-eyed watchman of Greek myth) is the user-facing envelope
around the four CSCI435 vision modules. It provides:
    * Three input modalities: live webcam, uploaded image, uploaded video
    * A live monitor with bounding boxes + a colour-coded status system
    * An analytics dashboard (KPIs + charts) driven by SQLite/Pandas
    * A searchable session history

Custom HTML/CSS is injected via st.markdown so the UI is fully styled on top
of the Streamlit framework.

Run:  streamlit run app.py
"""

import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import database as db
from proctor_engine import ProctorEngine

# ----------------------------------------------------------------- config ----
st.set_page_config(
    page_title="Argus | Exam Proctoring",
    layout="wide",
)
db.init_db()

STATUS_COLORS = {"Violation": "#ef4444", "Warning": "#f59e0b", "Normal": "#22c55e"}

# --------------------------------------------------------------- styling -----
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"], [data-testid] { font-family: 'Space Grotesk', sans-serif !important; }
/* Never override Streamlit's Material icon font, or ligatures leak as raw
   text (e.g. "keyboard_double_arrow_left" on the sidebar collapse button). */
[data-testid="stIconMaterial"],
.material-icons, .material-icons-round, .material-symbols-rounded,
[class*="material-symbols"], [class*="material-icons"] {
  font-family: 'Material Symbols Rounded', 'Material Icons Round',
               'Material Icons' !important;
}
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; min-height: 0 !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; background: transparent; }

/* full-page background */
.stApp {
  background: #060a14 !important;
  background-image:
    radial-gradient(ellipse 70% 55% at 8% -5%, rgba(6,182,212,0.09) 0%, transparent 65%),
    radial-gradient(ellipse 55% 45% at 92% 105%, rgba(99,102,241,0.07) 0%, transparent 65%) !important;
}
[data-testid="stMain"] { background: transparent !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(6,182,212,0.25); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(6,182,212,0.5); }

/* ===== ANIMATIONS ===== */
@keyframes pulse-dot {
  0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.6), 0 0 8px rgba(74,222,128,0.5); }
  60%       { box-shadow: 0 0 0 8px rgba(74,222,128,0), 0 0 8px rgba(74,222,128,0.5); }
}
@keyframes fade-up {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes scan-line {
  0%   { transform: translateX(-100%); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateX(400%); opacity: 0; }
}
@keyframes border-glow {
  0%, 100% { box-shadow: 0 0 12px rgba(6,182,212,0.15); }
  50%       { box-shadow: 0 0 24px rgba(6,182,212,0.3); }
}

/* ===== HEADER ===== */
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(135deg, #08111f 0%, #060d1a 60%, #070e1e 100%);
  border: 1px solid rgba(6,182,212,0.22);
  border-top: 1px solid rgba(6,182,212,0.35);
  border-radius: 18px;
  padding: 20px 32px;
  margin-bottom: 22px;
  position: relative; overflow: hidden;
  box-shadow:
    0 0 0 1px rgba(6,182,212,0.04),
    0 1px 0 rgba(6,182,212,0.2),
    0 32px 80px rgba(0,0,0,0.7),
    inset 0 1px 0 rgba(255,255,255,0.04);
  animation: border-glow 4s ease infinite;
}
/* dot grid */
.app-header::before {
  content: '';
  position: absolute; inset: 0;
  background-image: radial-gradient(circle, rgba(6,182,212,0.09) 1px, transparent 1px);
  background-size: 20px 20px;
  pointer-events: none;
}
/* sweep scan line */
.app-header::after {
  content: '';
  position: absolute; top: 0; left: 0; width: 25%; height: 100%;
  background: linear-gradient(90deg, transparent 0%, rgba(6,182,212,0.06) 50%, transparent 100%);
  animation: scan-line 6s ease-in-out infinite;
  pointer-events: none;
}
/* left cyan bloom */
.app-header-glow {
  position: absolute; left: -40px; top: 50%; transform: translateY(-50%);
  width: 200px; height: 120px;
  background: radial-gradient(ellipse, rgba(6,182,212,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.app-brand { display: flex; align-items: center; gap: 20px; position: relative; z-index: 1; }
.app-logo {
  display: flex; align-items: center; justify-content: center;
  width: 64px; height: 64px;
  background: rgba(6,182,212,0.07);
  border: 1px solid rgba(6,182,212,0.28);
  border-radius: 16px;
  box-shadow: 0 0 28px rgba(6,182,212,0.14), 0 0 0 4px rgba(6,182,212,0.04), inset 0 1px 0 rgba(6,182,212,0.12);
}
.app-title {
  color: #f0f9ff;
  font-size: 33px; font-weight: 800; letter-spacing: 8px; line-height: 1;
  text-shadow: 0 0 60px rgba(6,182,212,0.5), 0 0 20px rgba(6,182,212,0.3);
}
.app-subtitle {
  color: rgba(6,182,212,0.6);
  font-size: 10px; font-weight: 600; margin-top: 7px;
  letter-spacing: 2.5px; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important;
}
.app-right { display: flex; align-items: center; gap: 12px; position: relative; z-index: 1; }
.app-stat {
  text-align: center;
  padding: 8px 16px;
  border: 1px solid rgba(6,182,212,0.1);
  border-radius: 10px;
  background: rgba(6,182,212,0.04);
  font-family: 'JetBrains Mono', monospace !important;
}
.app-stat-val { font-size: 15px; font-weight: 700; color: #22d3ee; line-height: 1; }
.app-stat-lbl { font-size: 9px; font-weight: 600; color: rgba(6,182,212,0.5); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 3px; }
.app-badge {
  display: flex; align-items: center; gap: 9px;
  background: rgba(6,182,212,0.06);
  color: rgba(6,182,212,0.9);
  padding: 10px 22px; border-radius: 999px;
  font-size: 10px; font-weight: 700;
  border: 1px solid rgba(6,182,212,0.22);
  letter-spacing: 2.5px; text-transform: uppercase;
  box-shadow: 0 0 20px rgba(6,182,212,0.1), inset 0 1px 0 rgba(6,182,212,0.08);
  font-family: 'JetBrains Mono', monospace !important;
}
.dot-live {
  width: 8px; height: 8px; border-radius: 50%;
  background: #4ade80;
  animation: pulse-dot 2.2s ease infinite;
}

/* ===== SECTION TITLES ===== */
.section-title {
  font-size: 18px; font-weight: 700; color: #e2e8f0;
  margin: 10px 0 8px;
  padding: 10px 16px;
  border-left: 3px solid #06b6d4;
  background: linear-gradient(90deg, rgba(6,182,212,0.06) 0%, transparent 70%);
  border-radius: 0 8px 8px 0;
  line-height: 1.3;
  animation: fade-up 0.4s ease;
}
.section-sub {
  font-size: 9.5px; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase;
  color: rgba(6,182,212,0.55);
  margin: 20px 0 8px;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ===== KPI CARDS ===== */
.kpi-grid {
  display: grid; grid-template-columns: repeat(6,1fr); gap: 10px;
  margin: 6px 0 12px;
  animation: fade-up 0.35s ease;
}
.kpi-card {
  background: rgba(8,14,26,0.92);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(6,182,212,0.09);
  border-radius: 14px; padding: 15px 16px;
  border-top: 2px solid var(--accent, #6366f1);
  transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
  position: relative; overflow: hidden;
}
/* inner top shimmer */
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.07) 50%, transparent 100%);
}
/* bottom data bar */
.kpi-card::after {
  content: '';
  position: absolute; bottom: 0; left: 16px; right: 16px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(6,182,212,0.08), transparent);
}
.kpi-card:hover {
  transform: translateY(-3px);
  border-color: rgba(6,182,212,0.25);
  box-shadow: 0 10px 36px rgba(0,0,0,0.5), 0 0 0 1px rgba(6,182,212,0.1), 0 0 20px rgba(6,182,212,0.05);
}
.kpi-label {
  font-size: 9px; font-weight: 600; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(6,182,212,0.5);
  font-family: 'JetBrains Mono', monospace !important;
}
.kpi-value {
  font-size: 24px; font-weight: 700; color: #f0f9ff;
  margin-top: 5px; line-height: 1.1;
}
.kpi-sub {
  font-size: 10.5px; font-weight: 500;
  color: rgba(148,163,184,0.55); margin-left: 3px;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ===== VIOLATION CHIPS ===== */
.viol-bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 14px; }
.viol-chip {
  background: rgba(239,68,68,0.08); color: #fca5a5;
  border: 1px solid rgba(239,68,68,0.25);
  padding: 5px 16px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.3px;
  box-shadow: 0 0 12px rgba(239,68,68,0.1);
}
.viol-none {
  background: rgba(34,197,94,0.07); color: #86efac;
  border: 1px solid rgba(34,197,94,0.2);
  padding: 5px 16px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600;
  box-shadow: 0 0 12px rgba(34,197,94,0.08);
}
.enh-chip {
  background: rgba(6,182,212,0.08); color: #67e8f9;
  border: 1px solid rgba(6,182,212,0.2);
  padding: 5px 16px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600;
}

/* ===== DASHBOARD CARDS ===== */
.dash-grid {
  display: grid; grid-template-columns: repeat(4,1fr); gap: 14px;
  margin: 8px 0 14px;
  animation: fade-up 0.35s ease;
}
.dash-card {
  background: rgba(8,14,26,0.92);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(6,182,212,0.09);
  border-left: 3px solid #06b6d4;
  border-radius: 16px; padding: 22px 24px;
  position: relative; overflow: hidden;
  transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.dash-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, rgba(6,182,212,0.2), transparent 60%);
}
.dash-card::after {
  content: '';
  position: absolute; top: 16px; right: 16px; width: 40px; height: 40px;
  background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
  border-radius: 50%;
}
.dash-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(6,182,212,0.12);
  border-left-color: #22d3ee;
}
.dash-num {
  font-size: 36px; font-weight: 800; color: #06b6d4; line-height: 1.1;
  font-family: 'JetBrains Mono', monospace !important;
  text-shadow: 0 0 30px rgba(6,182,212,0.3);
}
.dash-lbl {
  font-size: 10px; color: rgba(148,163,184,0.65);
  font-weight: 600; margin-top: 6px;
  text-transform: uppercase; letter-spacing: 1px;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060c18 0%, #07101e 100%) !important;
  border-right: 1px solid rgba(6,182,212,0.1) !important;
}
section[data-testid="stSidebar"] .section-sub { margin-top: 14px; }
section[data-testid="stSidebar"] > div { padding-top: 1.2rem !important; }

.sb-card {
  background: linear-gradient(135deg, rgba(6,182,212,0.07) 0%, rgba(8,14,26,0.95) 100%);
  border: 1px solid rgba(6,182,212,0.2);
  border-top: 1px solid rgba(6,182,212,0.35);
  border-radius: 14px; padding: 16px 18px; margin-bottom: 18px;
  box-shadow: 0 0 30px rgba(6,182,212,0.07), inset 0 1px 0 rgba(6,182,212,0.12);
  position: relative; overflow: hidden;
}
.sb-card::before {
  content: '';
  position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, #06b6d4 0%, #6366f1 100%);
  border-radius: 14px 0 0 14px;
}
.sb-card .t { font-weight: 800; letter-spacing: 6px; font-size: 17px; color: #f0f9ff; padding-left: 10px; }
.sb-card .s {
  font-size: 10px; color: rgba(6,182,212,0.6);
  margin-top: 4px; letter-spacing: 1.5px; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important; padding-left: 10px;
}

/* sidebar widgets */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label {
  color: rgba(226,232,240,0.85) !important;
  font-size: 13px !important; font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stSlider label { color: rgba(226,232,240,0.85) !important; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background: rgba(6,182,212,0.05) !important;
  border: 1px solid rgba(6,182,212,0.18) !important;
  border-radius: 8px !important; color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: #06b6d4 !important;
}

/* ===== ICON CONTROLS (now render as real Material icons) ===== */
/* tint the sidebar collapse / expand toggles to match the theme */
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {
  color: rgba(6,182,212,0.75) !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: rgba(6,10,20,0.8) !important;
  border-radius: 12px !important; padding: 5px !important; gap: 3px !important;
  border: 1px solid rgba(6,182,212,0.12) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border-radius: 8px !important;
  color: rgba(148,163,184,0.65) !important;
  font-weight: 600 !important; font-size: 13px !important;
  padding: 9px 26px !important; border: none !important;
  transition: all 0.18s ease !important; letter-spacing: 0.3px !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(6,182,212,0.1) !important; color: #22d3ee !important;
  box-shadow: inset 0 -2px 0 #06b6d4, 0 0 20px rgba(6,182,212,0.1) !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #e2e8f0 !important; background: rgba(6,182,212,0.06) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ===== BUTTONS ===== */
.stButton > button {
  background: rgba(6,182,212,0.06) !important; color: #22d3ee !important;
  border: 1px solid rgba(6,182,212,0.25) !important; border-radius: 9px !important;
  font-weight: 600 !important; font-size: 13px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  transition: all 0.18s ease !important; letter-spacing: 0.4px !important;
  padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover {
  background: rgba(6,182,212,0.13) !important;
  border-color: rgba(6,182,212,0.45) !important;
  box-shadow: 0 0 22px rgba(6,182,212,0.2), 0 4px 12px rgba(0,0,0,0.3) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
  background: rgba(6,182,212,0.14) !important;
  border-color: #06b6d4 !important;
  box-shadow: 0 0 28px rgba(6,182,212,0.18), inset 0 1px 0 rgba(6,182,212,0.15) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
  border: 1px dashed rgba(6,182,212,0.22) !important;
  border-radius: 12px !important;
  background: rgba(6,182,212,0.03) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(6,182,212,0.4) !important;
  background: rgba(6,182,212,0.05) !important;
}

/* ===== ALERTS / INFO ===== */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid rgba(6,182,212,0.15) !important;
  background: rgba(6,182,212,0.04) !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(6,182,212,0.12) !important;
  border-radius: 12px !important; overflow: hidden !important;
}

/* ===== PROGRESS BAR ===== */
[data-testid="stProgress"] > div > div > div > div {
  background: linear-gradient(90deg, #06b6d4, #6366f1) !important;
  border-radius: 999px !important;
}
[data-testid="stProgress"] > div > div > div {
  background: rgba(6,182,212,0.1) !important; border-radius: 999px !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Live header stats (pulled from the SQLite store)
_sessions_df = db.get_sessions_df()
_viol_df = db.get_violations_df()
_hdr_sessions = len(_sessions_df)
_hdr_viol = len(_viol_df)

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-header-glow"></div>
      <div class="app-brand">
        <div class="app-logo">
          <svg width="42" height="42" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg"
               style="filter:drop-shadow(0 0 9px rgba(6,182,212,0.75))">
            <path d="M2 20 L11 9 L29 9 L38 20 L29 31 L11 31 Z"
                  stroke="#06b6d4" stroke-width="1.4" fill="none"/>
            <path d="M12 20 L15.5 14 L24.5 14 L28 20 L24.5 26 L15.5 26 Z"
                  stroke="rgba(255,255,255,0.5)" stroke-width="0.9"
                  fill="rgba(6,182,212,0.08)"/>
            <line x1="15.5" y1="14" x2="20" y2="20" stroke="rgba(6,182,212,0.38)" stroke-width="0.7"/>
            <line x1="24.5" y1="14" x2="20" y2="20" stroke="rgba(6,182,212,0.38)" stroke-width="0.7"/>
            <line x1="15.5" y1="26" x2="20" y2="20" stroke="rgba(6,182,212,0.38)" stroke-width="0.7"/>
            <line x1="24.5" y1="26" x2="20" y2="20" stroke="rgba(6,182,212,0.38)" stroke-width="0.7"/>
            <circle cx="20" cy="20" r="4.2" fill="#06b6d4" fill-opacity="0.88"/>
            <circle cx="20" cy="20" r="1.9" fill="white"/>
            <line x1="2"  y1="20" x2="11" y2="20" stroke="#06b6d4" stroke-width="1.2" stroke-dasharray="2.5 1.5"/>
            <line x1="29" y1="20" x2="38" y2="20" stroke="#06b6d4" stroke-width="1.2" stroke-dasharray="2.5 1.5"/>
            <line x1="11" y1="9"  x2="13" y2="7"  stroke="rgba(6,182,212,0.55)" stroke-width="1"/>
            <line x1="29" y1="9"  x2="27" y2="7"  stroke="rgba(6,182,212,0.55)" stroke-width="1"/>
            <line x1="11" y1="31" x2="13" y2="33" stroke="rgba(6,182,212,0.55)" stroke-width="1"/>
            <line x1="29" y1="31" x2="27" y2="33" stroke="rgba(6,182,212,0.55)" stroke-width="1"/>
          </svg>
        </div>
        <div>
          <div class="app-title">ARGUS</div>
          <div class="app-subtitle">// Intelligent Exam Integrity &nbsp;&middot;&nbsp; CSCI435</div>
        </div>
      </div>
      <div class="app-right">
        <div class="app-stat">
          <div class="app-stat-val">{_hdr_sessions}</div>
          <div class="app-stat-lbl">Sessions</div>
        </div>
        <div class="app-stat">
          <div class="app-stat-val">{_hdr_viol}</div>
          <div class="app-stat-lbl">Events</div>
        </div>
        <div class="app-badge"><span class="dot-live"></span>&nbsp; SYSTEM ONLINE</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar ---
st.sidebar.markdown(
    """
    <div class="sb-card">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px">
        <svg width="22" height="22" viewBox="0 0 40 40" fill="none"
             style="flex-shrink:0;filter:drop-shadow(0 0 4px rgba(6,182,212,0.6))">
          <path d="M2 20 L11 9 L29 9 L38 20 L29 31 L11 31 Z"
                stroke="#06b6d4" stroke-width="1.6" fill="none"/>
          <path d="M12 20 L15.5 14 L24.5 14 L28 20 L24.5 26 L15.5 26 Z"
                stroke="rgba(255,255,255,0.5)" stroke-width="1"
                fill="rgba(6,182,212,0.08)"/>
          <circle cx="20" cy="20" r="4.2" fill="#06b6d4" fill-opacity="0.85"/>
          <circle cx="20" cy="20" r="1.9" fill="white"/>
          <line x1="2"  y1="20" x2="11" y2="20" stroke="#06b6d4" stroke-width="1.2" stroke-dasharray="2.5 1.5"/>
          <line x1="29" y1="20" x2="38" y2="20" stroke="#06b6d4" stroke-width="1.2" stroke-dasharray="2.5 1.5"/>
        </svg>
        <div class="t">ARGUS</div>
      </div>
      <div class="s">Exam Proctoring Console</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="section-sub">Input source</div>', unsafe_allow_html=True)
source_type = st.sidebar.radio(
    "Input source", ["Live Webcam", "Upload Image", "Upload Video"],
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="section-sub">Vision modules</div>', unsafe_allow_html=True)
enable_object = st.sidebar.checkbox("Phone / Person", value=True)
enable_gaze = st.sidebar.checkbox("Gaze tracking", value=True)
enable_face = st.sidebar.checkbox("Face recognition", value=True)
enable_motion = st.sidebar.checkbox("Motion / camera", value=True)
enable_enhance = st.sidebar.checkbox("Low-light enhancement", value=True)

st.sidebar.markdown('<div class="section-sub">Detection settings</div>', unsafe_allow_html=True)
object_model = st.sidebar.selectbox(
    "YOLO model", ["yolov8s.pt", "best.pt"], index=0,
    help="best.pt = the custom fine-tuned phone detector.",
)
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)


# ----------------------------------------------------------------- helpers ---
def build_engine(fps):
    return ProctorEngine(
        fps=fps, object_model=object_model, conf=conf,
        enable_object=enable_object, enable_gaze=enable_gaze,
        enable_face=enable_face, enable_motion=enable_motion,
        enable_enhance=enable_enhance,
    )


def log_if_needed(session_id, result, last_status):
    status = result["overall_status"]
    if status != "Normal" and status != last_status:
        db.log_violation(
            session_id=session_id,
            frame_number=result.get("frame_number", 0),
            violation_type="; ".join(result["violations"]) or status,
            overall_status=status,
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            details=(f"persons={result['persons']} phones={result['phones']} "
                     f"gaze={result['gaze_direction']} motion={result['motion_violation']}"),
        )
    return status


def metrics_html(result):
    status = result["overall_status"]
    color = STATUS_COLORS.get(status, "#22c55e")
    chips = "".join(f'<span class="viol-chip">{v}</span>' for v in result["violations"])
    if not chips:
        chips = '<span class="viol-none">No active violations</span>'
    if result.get("enhanced"):
        chips = '<span class="enh-chip">Low-light enhancement active</span>' + chips
    return f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent:{color}">
        <div class="kpi-label">Status</div>
        <div class="kpi-value" style="color:{color}">{status}</div></div>
      <div class="kpi-card" style="--accent:{color}">
        <div class="kpi-label">Risk score</div>
        <div class="kpi-value">{result['risk_score']}<span class="kpi-sub">{result['risk_level']}</span></div></div>
      <div class="kpi-card">
        <div class="kpi-label">Persons / Phones</div>
        <div class="kpi-value">{result['persons']} / {result['phones']}</div></div>
      <div class="kpi-card">
        <div class="kpi-label">Gaze</div>
        <div class="kpi-value" style="font-size:18px">{result['gaze_direction']}</div></div>
      <div class="kpi-card">
        <div class="kpi-label">FPS</div>
        <div class="kpi-value">{result['fps']}</div></div>
      <div class="kpi-card">
        <div class="kpi-label">Latency</div>
        <div class="kpi-value">{result['latency_ms']}<span class="kpi-sub">ms</span></div></div>
    </div>
    <div class="viol-bar">{chips}</div>
    """


tab_live, tab_dash, tab_history = st.tabs(
    ["  Live Monitor  ", "  Dashboard  ", "  Session History  "]
)

# ============================================================ LIVE MONITOR ===
with tab_live:
    # ---------- modality 1: live webcam ----------
    if source_type == "Live Webcam":
        st.markdown('<div class="section-title">Live Webcam Monitor</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("Start", type="primary", use_container_width=True):
            st.session_state.run_webcam = True
        if c2.button("Stop", use_container_width=True):
            st.session_state.run_webcam = False

        metrics_slot = st.empty()
        frame_slot = st.empty()

        if st.session_state.get("run_webcam"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open the webcam (device 0). "
                         "Check System Settings -> Privacy & Security -> Camera.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                engine = build_engine(fps)
                session_id = db.start_session("webcam", "device_0")
                last_status, frames, fps_sum, viol_total = "Normal", 0, 0.0, 0
                while st.session_state.get("run_webcam"):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames += 1
                    result = engine.process_frame(frame)
                    result["frame_number"] = frames
                    fps_sum += result["fps"]
                    prev = last_status
                    last_status = log_if_needed(session_id, result, last_status)
                    if last_status != "Normal" and prev == "Normal":
                        viol_total += 1
                    metrics_slot.markdown(metrics_html(result), unsafe_allow_html=True)
                    frame_slot.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                cap.release()
                db.end_session(session_id, frames, fps_sum / max(frames, 1), viol_total)
                st.success(f"Session saved \u2014 {frames} frames, {viol_total} flagged events.")

    # ---------- modality 2: uploaded image ----------
    elif source_type == "Upload Image":
        st.markdown('<div class="section-title">Single Image Analysis</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            engine = build_engine(fps=1)
            session_id = db.start_session("image", up.name)
            result = engine.process_frame(frame)
            result["frame_number"] = 1
            log_if_needed(session_id, result, "Normal")
            db.end_session(session_id, 1, 0,
                           1 if result["overall_status"] != "Normal" else 0)
            st.markdown(metrics_html(result), unsafe_allow_html=True)
            st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                     channels="RGB", use_container_width=True)

    # ---------- modality 3: uploaded video ----------
    elif source_type == "Upload Video":
        st.markdown('<div class="section-title">Video File Analysis</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if up is not None and st.button("Analyse video", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            engine = build_engine(fps)
            session_id = db.start_session("video", up.name)

            metrics_slot = st.empty()
            frame_slot = st.empty()
            progress = st.progress(0)
            last_status, frames, fps_sum, viol_total = "Normal", 0, 0.0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                result = engine.process_frame(frame)
                result["frame_number"] = frames
                fps_sum += result["fps"]
                prev = last_status
                last_status = log_if_needed(session_id, result, last_status)
                if last_status != "Normal" and prev == "Normal":
                    viol_total += 1
                if frames % 2 == 0:
                    metrics_slot.markdown(metrics_html(result), unsafe_allow_html=True)
                    frame_slot.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                if total:
                    progress.progress(min(frames / total, 1.0))
            cap.release()
            db.end_session(session_id, frames, fps_sum / max(frames, 1), viol_total)
            st.success(f"Analysis complete \u2014 {frames} frames, {viol_total} flagged events. "
                       "See the Dashboard and Session History tabs.")

# ============================================================ DASHBOARD ======
with tab_dash:
    st.markdown('<div class="section-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    sessions = db.get_sessions_df()
    viol = db.get_violations_df()

    if sessions.empty:
        st.info("No data yet. Run a session in the Live Monitor tab to populate the dashboard.")
    else:
        total_sessions = len(sessions)
        total_viol = len(viol)
        if not viol.empty:
            top_type = str(viol["violation_type"].mode().iloc[0])[:28]
            avg_risk = round(float(viol["risk_score"].mean()), 1)
        else:
            top_type, avg_risk = "None", 0

        st.markdown(
            f"""
            <div class="dash-grid">
              <div class="dash-card">
                <div class="dash-num">{total_sessions}</div>
                <div class="dash-lbl">Total sessions</div></div>
              <div class="dash-card">
                <div class="dash-num">{total_viol}</div>
                <div class="dash-lbl">Violation events</div></div>
              <div class="dash-card">
                <div class="dash-num" style="font-size:18px;margin-top:8px">{top_type}</div>
                <div class="dash-lbl">Most frequent violation</div></div>
              <div class="dash-card">
                <div class="dash-num">{avg_risk}</div>
                <div class="dash-lbl">Average risk score</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not viol.empty:
            st.markdown('<div class="section-sub">Violations by type</div>', unsafe_allow_html=True)
            counts = viol["violation_type"].astype(str).str.slice(0, 30).value_counts()
            st.bar_chart(counts, color="#6366f1")

            st.markdown('<div class="section-sub">Recent violation events</div>', unsafe_allow_html=True)
            st.dataframe(
                viol[["timestamp", "violation_type", "overall_status",
                      "risk_score", "risk_level"]].head(15),
                use_container_width=True, hide_index=True,
            )

        st.markdown('<div class="section-sub">Maintenance</div>', unsafe_allow_html=True)
        if st.button("Clear all data (reset dashboard)"):
            db.clear_all()
            st.rerun()

# ============================================================ HISTORY ========
with tab_history:
    st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
    sessions = db.get_sessions_df()
    if sessions.empty:
        st.info("No sessions yet. Run a webcam, image or video in the Live Monitor tab first.")
    else:
        st.dataframe(sessions, use_container_width=True, hide_index=True)
        chosen = st.selectbox("Show violations for session", sessions["id"].tolist())
        st.dataframe(db.get_violations_df(chosen), use_container_width=True, hide_index=True)
