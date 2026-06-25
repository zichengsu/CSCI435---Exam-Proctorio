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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1450px; }

/* ---------- top header banner ---------- */
.app-header{
  display:flex; align-items:center; justify-content:space-between;
  background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#9333ea 100%);
  border-radius:18px; padding:20px 28px; margin-bottom:16px;
  box-shadow:0 12px 32px rgba(79,70,229,.28);
}
.app-brand{ display:flex; align-items:center; gap:16px; }
.app-logo{ display:flex; align-items:center; justify-content:center;
  width:54px; height:54px; background:rgba(255,255,255,.15);
  border:1px solid rgba(255,255,255,.3); border-radius:14px; }
.app-title{ color:#fff; font-size:30px; font-weight:800; letter-spacing:4px; line-height:1; }
.app-subtitle{ color:rgba(255,255,255,.85); font-size:13px; font-weight:500; margin-top:5px; }
.app-badge{ display:flex; align-items:center; gap:8px; background:rgba(255,255,255,.15);
  color:#fff; padding:9px 16px; border-radius:999px; font-size:12px; font-weight:600;
  border:1px solid rgba(255,255,255,.3); }
.dot-live{ width:9px; height:9px; border-radius:50%; background:#4ade80;
  box-shadow:0 0 0 4px rgba(74,222,128,.25); }

/* ---------- section titles ---------- */
.section-title{ font-size:22px; font-weight:700; color:#1f2430; margin:6px 0 2px; }
.section-sub{ font-size:13px; font-weight:600; letter-spacing:.4px; text-transform:uppercase;
  color:#8a93a6; margin:18px 0 6px; }

/* ---------- live KPI cards ---------- */
.kpi-grid{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:4px 0 12px; }
.kpi-card{ background:#fff; border:1px solid #eceef3; border-radius:14px; padding:13px 16px;
  box-shadow:0 2px 6px rgba(16,24,40,.04); border-top:3px solid var(--accent,#6366f1); }
.kpi-label{ font-size:10.5px; font-weight:600; letter-spacing:.6px; text-transform:uppercase; color:#8a93a6; }
.kpi-value{ font-size:23px; font-weight:700; color:#1f2430; margin-top:3px; }
.kpi-sub{ font-size:12px; font-weight:600; color:#8a93a6; margin-left:3px; }

/* ---------- violation chips ---------- */
.viol-bar{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 12px; }
.viol-chip{ background:#fef2f2; color:#b91c1c; border:1px solid #fecaca;
  padding:6px 13px; border-radius:999px; font-size:13px; font-weight:600; }
.viol-none{ background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0;
  padding:6px 13px; border-radius:999px; font-size:13px; font-weight:600; }
.enh-chip{ background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe;
  padding:6px 13px; border-radius:999px; font-size:13px; font-weight:600; }

/* ---------- dashboard big cards ---------- */
.dash-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:8px 0 8px; }
.dash-card{ background:#fff; border:1px solid #eceef3; border-radius:16px; padding:20px 22px;
  box-shadow:0 4px 14px rgba(16,24,40,.05); }
.dash-num{ font-size:34px; font-weight:800; color:#4f46e5; line-height:1.1; }
.dash-lbl{ font-size:12.5px; color:#6b7280; font-weight:500; margin-top:4px; }

/* ---------- sidebar ---------- */
.sb-card{ background:linear-gradient(135deg,#4f46e5,#7c3aed); border-radius:14px;
  padding:16px 18px; margin-bottom:14px; color:#fff; }
.sb-card .t{ font-weight:800; letter-spacing:3px; font-size:18px; }
.sb-card .s{ font-size:11.5px; opacity:.85; margin-top:2px; }
section[data-testid="stSidebar"] .section-sub{ margin-top:14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
      <div class="app-brand">
        <div class="app-logo">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
            <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"
                  stroke="white" stroke-width="1.8"/>
            <circle cx="12" cy="12" r="3.2" fill="white"/>
          </svg>
        </div>
        <div>
          <div class="app-title">ARGUS</div>
          <div class="app-subtitle">Intelligent Exam Integrity Monitoring &middot; CSCI435</div>
        </div>
      </div>
      <div class="app-badge"><span class="dot-live"></span> SYSTEM ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar ---
st.sidebar.markdown(
    '<div class="sb-card"><div class="t">ARGUS</div>'
    '<div class="s">Exam Proctoring Console</div></div>',
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
