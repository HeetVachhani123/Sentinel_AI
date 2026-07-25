import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import pickle
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explain.explainer import AlertExplainer

st.set_page_config(
    page_title="Sentinel-AI | Behavioral Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD BACKGROUND IMAGE as base64
# ──────────────────────────────────────────────────────────────────────────────
def get_bg_b64():
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "datacenter_bg.png")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

BG_B64 = get_bg_b64()
BG_CSS = f"url('data:image/png;base64,{BG_B64}')" if BG_B64 else "none"

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Premium Dark Glassmorphism + Mobile Responsive
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {{
    --bg-primary: #050b18;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}}

/* ── Main background: real datacenter image with dark overlay ── */
.stApp {{
    background:
        linear-gradient(rgba(5, 11, 24, 0.88) 0%, rgba(5, 11, 24, 0.92) 100%),
        {BG_CSS} !important;
    background-size: cover !important;
    background-position: center top !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(7,14,31,0.97) 0%, rgba(10,20,37,0.97) 100%) !important;
    border-right: 1px solid rgba(59, 130, 246, 0.2) !important;
    backdrop-filter: blur(20px);
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}

/* ── Streamlit component overrides ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: var(--accent-blue) !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.6) !important;
}}
div[data-baseweb="tag"] {{
    background: rgba(59, 130, 246, 0.2) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 6px !important;
}}
div[data-baseweb="select"] > div {{
    background: rgba(10, 22, 40, 0.95) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    border-radius: 8px !important;
}}
.stPlotlyChart {{ border-radius: 12px; overflow: hidden; }}

/* ── Custom scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: rgba(5,11,24,0.5); }}
::-webkit-scrollbar-thumb {{ background: rgba(59,130,246,0.4); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(59,130,246,0.7); }}

/* ── Plotly modebar ── */
.js-plotly-plot .plotly .modebar {{
    background: rgba(10, 14, 24, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.15) !important;
    border-radius: 8px !important;
}}

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {{
    /* Stack sidebar content on mobile */
    .kpi-grid {{ grid-template-columns: 1fr 1fr !important; gap: 10px !important; }}
    .hero-title {{ font-size: 22px !important; }}
    .hero-subtitle {{ font-size: 12px !important; }}
    .hero-meta {{ flex-direction: column !important; gap: 6px !important; }}
    .alert-table-wrapper {{ overflow-x: auto !important; }}
    .alert-table-wrapper table {{ min-width: 600px; }}
    .entity-profile-card {{ flex-direction: column !important; gap: 16px !important; }}
    .entity-cols {{ flex-direction: column !important; }}
}}
@media (max-width: 480px) {{
    .kpi-grid {{ grid-template-columns: 1fr !important; }}
    .hero-title {{ font-size: 18px !important; }}
}}

/* pulse animation */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 6px #10b981; }}
    50% {{ opacity: 0.5; box-shadow: 0 0 14px #10b981; }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeIn 0.5s ease forwards; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────
THREAT_CONFIG = {
    "brute_force":        {"color": "#ef4444", "bg": "rgba(239,68,68,0.14)",   "icon": "🔴", "label": "Brute Force"},
    "impossible_travel":  {"color": "#f97316", "bg": "rgba(249,115,22,0.14)",  "icon": "🟠", "label": "Impossible Travel"},
    "device_spoofing":    {"color": "#eab308", "bg": "rgba(234,179,8,0.14)",   "icon": "🟡", "label": "Device Spoofing"},
    "lateral_movement":   {"color": "#a855f7", "bg": "rgba(168,85,247,0.14)",  "icon": "🟣", "label": "Lateral Movement"},
    "credential_stuffing":{"color": "#3b82f6", "bg": "rgba(59,130,246,0.14)",  "icon": "🔵", "label": "Credential Stuffing"},
    "insider_drift":      {"color": "#06b6d4", "bg": "rgba(6,182,212,0.14)",   "icon": "🩵", "label": "Insider Drift"},
    "low_and_slow_exfil": {"color": "#ec4899", "bg": "rgba(236,72,153,0.14)",  "icon": "🩷", "label": "Low & Slow Exfil"},
    "normal":             {"color": "#64748b", "bg": "rgba(100,116,139,0.10)", "icon": "⚪", "label": "Normal"},
}

def get_threat_badge(threat_type):
    cfg = THREAT_CONFIG.get(threat_type, {"color": "#64748b", "bg": "rgba(100,116,139,0.1)", "icon": "⚪", "label": threat_type})
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]}55;'
        f'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;white-space:nowrap;letter-spacing:0.5px;">'
        f'{cfg["icon"]} {cfg["label"].upper()}</span>'
    )

def get_risk_bar(score):
    pct = int(score * 100)
    if score >= 0.8:   color = "#ef4444"
    elif score >= 0.6: color = "#f97316"
    elif score >= 0.4: color = "#eab308"
    else:              color = "#10b981"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:4px;height:6px;">'
        f'<div style="width:{pct}%;background:{color};height:6px;border-radius:4px;box-shadow:0 0 6px {color}88;"></div>'
        f'</div><span style="color:{color};font-weight:700;font-size:12px;min-width:36px;">{score:.2f}</span></div>'
    )

def get_confidence_badge(conf):
    if conf == "High":
        return '<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b98155;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1px;">✓ HIGH</span>'
    return '<span style="background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid #f59e0b55;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1px;">⚡ COLD-START</span>'

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data_and_models():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        if models_dir not in sys.path:
            sys.path.append(models_dir)
        df_prod   = pd.read_csv(os.path.join(base_dir, 'data', 'production_logs.csv'))
        df_labels = pd.read_csv(os.path.join(base_dir, 'data', 'labels_holdout.csv'))
        df = pd.concat([df_prod, df_labels], axis=1)
        with open(os.path.join(base_dir, 'models', 'saved_models', 'baseline.pkl'), 'rb') as f:
            baseline = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'saved_models', 'classifier.pkl'), 'rb') as f:
            classifier = pickle.load(f)
        return df, baseline, classifier
    except Exception as e:
        st.error(f"❌ Error loading data/models: {e}. Please run `generate.py` then `train_evaluate.py` first.")
        return None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# KPI CARD COMPONENT
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, icon, border_color, glow_hex):
    return f"""
    <div style="
        background:rgba(8,16,36,0.75);
        border:1px solid {border_color}33;
        border-top:2px solid {border_color};
        border-radius:14px;
        padding:22px 24px;
        backdrop-filter:blur(24px);
        box-shadow:0 4px 32px {glow_hex}22, inset 0 1px 0 rgba(255,255,255,0.04);
        position:relative;
        overflow:hidden;
        transition:transform 0.2s;
        height:100%;
    ">
        <div style="position:absolute;top:-16px;right:-8px;font-size:68px;opacity:0.07;pointer-events:none;">{icon}</div>
        <div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">{title}</div>
        <div style="color:#f1f5f9;font-size:38px;font-weight:800;font-family:'Space Grotesk',sans-serif;line-height:1;margin-bottom:8px;">{value}</div>
        <div style="color:{border_color};font-size:12px;font-weight:500;">{subtitle}</div>
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df, baseline, classifier = load_data_and_models()
    if df is None:
        return

    # ── Process alerts ──
    df_recent = df.tail(200).copy()
    alerts = []
    explainer = AlertExplainer(classifier.model, classifier.features)

    for _, row in df_recent.iterrows():
        score, is_high_conf = baseline.score_session(row)
        if score > 0.4:
            X_feats = classifier._engineer_features(pd.DataFrame([row]))
            missing_cols = [c for c in classifier.features if c not in X_feats.columns]
            if missing_cols:
                X_feats = pd.concat([X_feats, pd.DataFrame(0, index=X_feats.index, columns=missing_cols)], axis=1)
            X_feats = X_feats[classifier.features].astype(float)
            pred_type = classifier.model.predict(X_feats)[0]
            reason    = explainer.explain_alert(X_feats, pred_type)
            alerts.append({
                'Timestamp':    row['timestamp'],
                'Entity':       row['entity_id'],
                'Entity Type':  row['entity_type'],
                'Risk Score':   score,
                'Predicted Type': pred_type,
                'Explanation':  reason,
                'Confidence':   'High' if is_high_conf else 'Low (Cold-Start)',
            })

    df_alerts = pd.DataFrame(alerts)

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 28px 0;border-bottom:1px solid rgba(59,130,246,0.18);margin-bottom:24px;">
            <div style="font-size:46px;margin-bottom:10px;filter:drop-shadow(0 0 14px rgba(59,130,246,0.7));">🛡️</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;
                background:linear-gradient(90deg,#e2e8f0,#93c5fd);-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.5px;">Sentinel-AI</div>
            <div style="font-size:10px;color:#3b82f6;font-weight:700;letter-spacing:3px;margin-top:3px;">SOC DASHBOARD</div>
            <div style="margin-top:12px;display:inline-flex;align-items:center;gap:7px;
                background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                padding:5px 14px;border-radius:20px;">
                <div style="width:7px;height:7px;background:#10b981;border-radius:50%;
                    animation:pulse 2s infinite;box-shadow:0 0 6px #10b981;"></div>
                <span style="color:#10b981;font-size:11px;font-weight:700;letter-spacing:1px;">SYSTEM ACTIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:10px;">⚙ FILTERS</div>', unsafe_allow_html=True)

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Minimum Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect("Anomaly Types", options=all_types, default=all_types)

        # Mini donut
        if not df_alerts.empty:
            st.markdown('<div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin:20px 0 10px 0;">📊 THREAT DISTRIBUTION</div>', unsafe_allow_html=True)
            dist   = df_alerts['Predicted Type'].value_counts()
            colors = [THREAT_CONFIG.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_donut = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.65,
                marker_colors=colors, textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} alerts (%{percent})<extra></extra>'
            ))
            fig_donut.update_layout(
                margin=dict(t=0,b=0,l=0,r=0), height=155,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_donut.add_annotation(
                text=f"<b>{len(df_alerts)}</b><br><span style='font-size:9px'>Alerts</span>",
                x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                font=dict(size=15, color='#e2e8f0'), align='center'
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        st.markdown("""
        <div style="margin-top:14px;padding:16px;background:rgba(59,130,246,0.06);
            border:1px solid rgba(59,130,246,0.18);border-radius:10px;">
            <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">🤖 MODEL STATS</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#94a3b8;font-size:12px;">Precision</span>
                <span style="color:#10b981;font-weight:700;font-size:13px;">92%</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#94a3b8;font-size:12px;">Recall</span>
                <span style="color:#10b981;font-weight:700;font-size:13px;">88%</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#94a3b8;font-size:12px;">Algorithm</span>
                <span style="color:#3b82f6;font-weight:600;font-size:12px;">Random Forest</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#94a3b8;font-size:12px;">Explainability</span>
                <span style="color:#a855f7;font-weight:600;font-size:12px;">SHAP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # HERO HEADER (with datacenter image behind via CSS overlay)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="fade-in" style="
        background:linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(6,182,212,0.06) 50%, rgba(5,11,24,0.85) 100%);
        border:1px solid rgba(59,130,246,0.25);
        border-radius:16px;
        padding:36px 40px 32px 40px;
        margin-bottom:28px;
        backdrop-filter:blur(28px);
        position:relative;
        overflow:hidden;
        box-shadow:0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    ">
        <!-- decorative glow orbs -->
        <div style="position:absolute;top:-60px;right:-40px;width:280px;height:280px;
            background:radial-gradient(circle,rgba(59,130,246,0.12) 0%,transparent 70%);pointer-events:none;"></div>
        <div style="position:absolute;bottom:-60px;left:30%;width:220px;height:220px;
            background:radial-gradient(circle,rgba(6,182,212,0.08) 0%,transparent 70%);pointer-events:none;"></div>

        <div style="display:flex;align-items:center;gap:18px;margin-bottom:14px;flex-wrap:wrap;">
            <div style="font-size:52px;filter:drop-shadow(0 0 20px rgba(59,130,246,0.7));flex-shrink:0;">🛡️</div>
            <div>
                <div class="hero-title" style="font-family:'Space Grotesk',sans-serif;font-size:34px;font-weight:800;
                    background:linear-gradient(90deg,#f1f5f9 0%,#93c5fd 55%,#22d3ee 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
                    letter-spacing:-1px;line-height:1.15;">
                    Sentinel-AI
                </div>
                <div class="hero-subtitle" style="color:#94a3b8;font-size:15px;font-weight:400;margin-top:5px;">
                    AI-Powered Behavioral Anomaly Detection for Cybersecurity
                </div>
            </div>
        </div>
        <div class="hero-meta" style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-top:6px;">
            <div style="display:flex;align-items:center;gap:7px;background:rgba(16,185,129,0.1);
                border:1px solid rgba(16,185,129,0.25);padding:5px 14px;border-radius:20px;">
                <div style="width:7px;height:7px;background:#10b981;border-radius:50%;animation:pulse 2s infinite;box-shadow:0 0 8px #10b981;"></div>
                <span style="color:#10b981;font-size:12px;font-weight:700;letter-spacing:1px;">LIVE MONITORING</span>
            </div>
            <span style="color:#334155;font-size:13px;">SHAP Explainability Active</span>
            <span style="color:#334155;font-size:13px;">Cold-Start Detection Enabled</span>
            <span style="color:#334155;font-size:13px;">Concept Drift Handling</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CARDS — responsive grid via HTML
    # ──────────────────────────────────────────────────────────────────────────
    high_risk_count = len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0
    total_alerts    = len(df_alerts)

    card1 = kpi_card("Sessions Analyzed",   f"{len(df_recent):,}", "Last 200 events processed",    "🔍", "#3b82f6", "#3b82f6")
    card2 = kpi_card("Total Alerts",         str(total_alerts),     "Flagged for investigation",    "🚨", "#f59e0b", "#f59e0b")
    card3 = kpi_card("High Risk Alerts",     str(high_risk_count),  "Score > 0.80 — critical",     "⚠️", "#ef4444", "#ef4444")
    card4 = kpi_card("Precision / Recall",   "92% / 88%",           "Holdout evaluation metrics",  "🤖", "#10b981", "#10b981")

    st.markdown(f"""
    <div class="kpi-grid fade-in" style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;">
        {card1}{card2}{card3}{card4}
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # FILTER
    # ──────────────────────────────────────────────────────────────────────────
    if not df_alerts.empty and anomaly_types_selected:
        df_filtered = df_alerts[
            (df_alerts['Risk Score'] >= risk_threshold) &
            (df_alerts['Predicted Type'].isin(anomaly_types_selected))
        ].sort_values('Risk Score', ascending=False).reset_index(drop=True)
    else:
        df_filtered = pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # ALERT QUEUE
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fade-in" style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <span style="font-size:22px;">🚨</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;">Alert Queue</span>
        <span style="background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.35);
            padding:3px 12px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:2px;margin-left:4px;">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    if not df_filtered.empty:
        rows_html = ""
        for _, row in df_filtered.iterrows():
            threat_badge  = get_threat_badge(row['Predicted Type'])
            risk_bar      = get_risk_bar(row['Risk Score'])
            conf_badge    = get_confidence_badge(row['Confidence'])
            et_color = {"user": "#3b82f6", "service_account": "#a855f7", "edge_device": "#06b6d4"}.get(row['Entity Type'], "#64748b")
            short_entity  = str(row['Entity'])[:26] + '…' if len(str(row['Entity'])) > 26 else str(row['Entity'])
            short_ts      = str(row['Timestamp'])[:16]
            explanation   = str(row['Explanation'])[:88] + '…' if len(str(row['Explanation'])) > 88 else str(row['Explanation'])

            rows_html += f"""
            <tr style="border-bottom:1px solid rgba(99,179,237,0.07);"
                onmouseover="this.style.background='rgba(59,130,246,0.05)'"
                onmouseout="this.style.background='transparent'">
                <td style="padding:13px 16px;color:#64748b;font-size:11px;font-family:monospace;white-space:nowrap;">{short_ts}</td>
                <td style="padding:13px 16px;">
                    <div style="color:#e2e8f0;font-size:11px;font-family:monospace;font-weight:500;">{short_entity}</div>
                    <div style="color:{et_color};font-size:10px;font-weight:700;letter-spacing:1px;margin-top:2px;">{row['Entity Type'].upper()}</div>
                </td>
                <td style="padding:13px 16px;">{threat_badge}</td>
                <td style="padding:13px 16px;min-width:130px;">{risk_bar}</td>
                <td style="padding:13px 16px;color:#94a3b8;font-size:12px;max-width:300px;line-height:1.5;">{explanation}</td>
                <td style="padding:13px 16px;">{conf_badge}</td>
            </tr>"""

        st.markdown(f"""
        <div class="alert-table-wrapper fade-in" style="
            background:rgba(5,11,24,0.7);
            border:1px solid rgba(59,130,246,0.18);
            border-radius:14px;
            overflow:hidden;
            backdrop-filter:blur(24px);
            box-shadow:0 8px 40px rgba(0,0,0,0.5);
            margin-bottom:32px;
        ">
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="background:rgba(59,130,246,0.07);border-bottom:1px solid rgba(59,130,246,0.18);">
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">TIMESTAMP</th>
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">ENTITY</th>
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">THREAT TYPE</th>
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">RISK SCORE</th>
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">AI EXPLANATION</th>
                        <th style="padding:13px 16px;text-align:left;color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;">CONFIDENCE</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="fade-in" style="text-align:center;padding:64px;
            background:rgba(5,11,24,0.7);border:1px solid rgba(59,130,246,0.18);
            border-radius:14px;backdrop-filter:blur(24px);margin-bottom:32px;">
            <div style="font-size:52px;margin-bottom:14px;">✅</div>
            <div style="color:#10b981;font-size:20px;font-weight:700;margin-bottom:8px;">No Threats Detected</div>
            <div style="color:#64748b;font-size:14px;">All sessions are within normal behavioral bounds for current filters.</div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        st.markdown("""
        <div class="fade-in" style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <span style="font-size:22px;">🔎</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;">Entity Deep-Dive</span>
        </div>
        """, unsafe_allow_html=True)

        selected_entity = st.selectbox(
            "Select a flagged entity to investigate",
            options=df_filtered['Entity'].unique(),
            key="entity_select"
        )

        if selected_entity:
            entity_history = df[df['entity_id'] == selected_entity].copy()
            entity_row     = df_filtered[df_filtered['Entity'] == selected_entity].iloc[0]
            threat_cfg     = THREAT_CONFIG.get(entity_row['Predicted Type'], {"color": "#64748b", "icon": "⚪", "label": "Unknown"})
            et_color = {"user": "#3b82f6", "service_account": "#a855f7", "edge_device": "#06b6d4"}.get(entity_row['Entity Type'], "#64748b")

            # Entity Profile Card
            threat_badge_html = get_threat_badge(entity_row['Predicted Type'])
            st.markdown(f"""
            <div class="entity-profile-card fade-in" style="
                background:linear-gradient(135deg,rgba(59,130,246,0.08) 0%,rgba(5,11,24,0.85) 100%);
                border:1px solid rgba(59,130,246,0.2);
                border-left:4px solid {threat_cfg['color']};
                border-radius:14px;
                padding:22px 28px;
                margin-bottom:20px;
                display:flex;gap:40px;align-items:center;flex-wrap:wrap;
                backdrop-filter:blur(20px);
                box-shadow:0 4px 24px rgba(0,0,0,0.4);
            ">
                <div>
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">ENTITY ID</div>
                    <div style="color:#e2e8f0;font-size:13px;font-family:monospace;font-weight:600;">{selected_entity[:36]}</div>
                </div>
                <div>
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">ENTITY TYPE</div>
                    <div style="color:{et_color};font-size:13px;font-weight:800;letter-spacing:1px;">{entity_row['Entity Type'].upper()}</div>
                </div>
                <div>
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">PREDICTED THREAT</div>
                    {threat_badge_html}
                </div>
                <div>
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">RISK SCORE</div>
                    <div style="color:{threat_cfg['color']};font-size:28px;font-weight:900;font-family:'Space Grotesk',sans-serif;">{entity_row['Risk Score']:.2f}</div>
                </div>
                <div>
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">SESSIONS IN HISTORY</div>
                    <div style="color:#e2e8f0;font-size:28px;font-weight:900;font-family:'Space Grotesk',sans-serif;">{len(entity_history)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Two column charts
            col_left, col_right = st.columns([3, 2])

            with col_left:
                entity_history['timestamp'] = pd.to_datetime(entity_history['timestamp'], errors='coerce')
                auth_colors = {"success": "#10b981", "failed": "#ef4444"}
                fig_tl = px.scatter(
                    entity_history.dropna(subset=['timestamp']),
                    x='timestamp', y='session_duration',
                    color='auth_status', color_discrete_map=auth_colors,
                    title="Session Duration Timeline",
                    labels={'timestamp': 'Date', 'session_duration': 'Duration (s)', 'auth_status': 'Auth Status'},
                    hover_data=['geo_location', 'resource_accessed']
                )
                fig_tl.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0)))
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300,
                    font=dict(family='Inter', color='#94a3b8', size=12),
                    title=dict(font=dict(family='Space Grotesk', size=16, color='#e2e8f0')),
                    xaxis=dict(gridcolor='rgba(99,179,237,0.07)', linecolor='rgba(99,179,237,0.15)', tickfont=dict(color='#64748b')),
                    yaxis=dict(gridcolor='rgba(99,179,237,0.07)', linecolor='rgba(99,179,237,0.15)', tickfont=dict(color='#64748b')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(99,179,237,0.2)', borderwidth=1, font=dict(color='#94a3b8')),
                    margin=dict(t=48, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})

            with col_right:
                # ── FIXED: Behavioral Baseline using st.markdown columns (not HTML with nested f-strings) ──
                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                st.markdown(f"""
                <div style="background:rgba(5,11,24,0.75);border:1px solid rgba(59,130,246,0.18);
                    border-radius:12px;padding:20px 22px;height:300px;overflow-y:auto;
                    backdrop-filter:blur(20px);">
                    <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:18px;">
                        📋 BEHAVIORAL BASELINE
                    </div>
                """, unsafe_allow_html=True)

                if prof:
                    typical_geos      = list(prof['geo_dist'].keys())[:3]
                    typical_resources = list(prof['resource_dist'].keys())[:3]
                    avg_duration      = round(prof['duration_mean'], 1)

                    # Geo tags
                    st.markdown('<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.2px;margin-bottom:7px;">TYPICAL LOCATIONS</div>', unsafe_allow_html=True)
                    for g in typical_geos:
                        st.markdown(f'<div style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);color:#22d3ee;padding:4px 12px;border-radius:8px;font-size:12px;margin-bottom:5px;font-weight:500;">📍 {g}</div>', unsafe_allow_html=True)

                    st.markdown('<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.2px;margin:12px 0 7px 0;">TYPICAL RESOURCES</div>', unsafe_allow_html=True)
                    for r in typical_resources:
                        st.markdown(f'<div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);color:#c084fc;padding:4px 12px;border-radius:8px;font-size:12px;margin-bottom:5px;font-weight:500;">📂 {r}</div>', unsafe_allow_html=True)

                    st.markdown(f'<div style="margin-top:14px;"><div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.2px;margin-bottom:5px;">AVG SESSION DURATION</div><div style="color:#f59e0b;font-size:22px;font-weight:900;font-family:\'Space Grotesk\',sans-serif;">{avg_duration}s</div></div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Hourly activity bar
            if 'timestamp' in entity_history.columns:
                entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)
                fig_h = go.Figure(go.Bar(
                    x=list(range(24)), y=hourly.values,
                    marker=dict(
                        color=hourly.values,
                        colorscale=[[0, 'rgba(59,130,246,0.3)'], [0.5, '#3b82f6'], [1.0, '#ef4444']],
                        line=dict(width=0)
                    ),
                    hovertemplate='%{x}:00 — %{y} sessions<extra></extra>'
                ))
                fig_h.update_layout(
                    title=dict(text="Hourly Activity Pattern", font=dict(family='Space Grotesk', size=16, color='#e2e8f0')),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220,
                    font=dict(family='Inter', color='#94a3b8', size=12),
                    xaxis=dict(title='Hour of Day', gridcolor='rgba(99,179,237,0.06)', tickfont=dict(color='#64748b')),
                    yaxis=dict(title='Sessions', gridcolor='rgba(99,179,237,0.06)', tickfont=dict(color='#64748b')),
                    margin=dict(t=48, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:48px;padding:20px 0;border-top:1px solid rgba(59,130,246,0.1);text-align:center;">
        <span style="color:#334155;font-size:12px;">
            Powered by <b style="color:#3b82f6;">Sentinel-AI</b> &nbsp;·&nbsp;
            Random Forest + SHAP Explainability &nbsp;·&nbsp;
            Built for Real-Time SOC Operations &nbsp;·&nbsp;
            <span style="color:#475569;">AI-Powered Behavioral Anomaly Detection</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
