import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explain.explainer import AlertExplainer

st.set_page_config(
    page_title="Sentinel-AI | Behavioral Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS INJECTION — Premium Dark Glassmorphism Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #050b18;
    --bg-secondary: #0a1628;
    --bg-card: rgba(16, 28, 52, 0.8);
    --border-color: rgba(99, 179, 237, 0.15);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
    --glow-red: 0 0 20px rgba(239, 68, 68, 0.4);
    --glow-green: 0 0 20px rgba(16, 185, 129, 0.3);
}

/* ── Base & Font ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main App Background with Grid Texture ── */
.stApp {
    background: 
        linear-gradient(rgba(59, 130, 246, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.03) 1px, transparent 1px),
        linear-gradient(135deg, #050b18 0%, #0a1628 50%, #060d1f 100%) !important;
    background-size: 40px 40px, 40px 40px, 100% 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070e1f 0%, #0a1425 100%) !important;
    border-right: 1px solid rgba(59, 130, 246, 0.2) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Streamlit default overrides ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-blue) !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.6) !important;
}
div[data-baseweb="tag"] {
    background: rgba(59, 130, 246, 0.2) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(10, 22, 40, 0.9) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    border-radius: 8px !important;
}
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(59, 130, 246, 0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.7); }

/* ── Plotly override ── */
.js-plotly-plot .plotly .modebar {
    background: rgba(10, 14, 24, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.15) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT TAXONOMY — Colors and Icons per attack type
# ──────────────────────────────────────────────────────────────────────────────
THREAT_CONFIG = {
    "brute_force":        {"color": "#ef4444", "bg": "rgba(239,68,68,0.12)",   "icon": "🔴", "label": "Brute Force"},
    "impossible_travel":  {"color": "#f97316", "bg": "rgba(249,115,22,0.12)",  "icon": "🟠", "label": "Impossible Travel"},
    "device_spoofing":    {"color": "#eab308", "bg": "rgba(234,179,8,0.12)",   "icon": "🟡", "label": "Device Spoofing"},
    "lateral_movement":   {"color": "#a855f7", "bg": "rgba(168,85,247,0.12)",  "icon": "🟣", "label": "Lateral Movement"},
    "credential_stuffing":{"color": "#3b82f6", "bg": "rgba(59,130,246,0.12)",  "icon": "🔵", "label": "Credential Stuffing"},
    "insider_drift":      {"color": "#06b6d4", "bg": "rgba(6,182,212,0.12)",   "icon": "🩵", "label": "Insider Drift"},
    "low_and_slow_exfil": {"color": "#ec4899", "bg": "rgba(236,72,153,0.12)",  "icon": "🩷", "label": "Low & Slow Exfil"},
    "normal":             {"color": "#64748b", "bg": "rgba(100,116,139,0.10)", "icon": "⚪", "label": "Normal"},
}

def get_threat_badge(threat_type):
    cfg = THREAT_CONFIG.get(threat_type, {"color": "#64748b", "bg": "rgba(100,116,139,0.1)", "icon": "⚪", "label": threat_type})
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]}44;'
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
        f'<div style="width:{pct}%;background:{color};height:6px;border-radius:4px;box-shadow:0 0 6px {color}88;"></div></div>'
        f'<span style="color:{color};font-weight:600;font-size:12px;min-width:36px;">{score:.2f}</span>'
        f'</div>'
    )

def get_confidence_badge(conf):
    if conf == "High":
        return '<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b98144;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1px;">✓ HIGH</span>'
    return '<span style="background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid #f59e0b44;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1px;">⚡ COLD START</span>'

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
        df_prod = pd.read_csv(os.path.join(base_dir, 'data', 'production_logs.csv'))
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
# COMPONENT: KPI CARD
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, icon, glow_color, border_color):
    st.markdown(f"""
    <div style="
        background: rgba(10,20,42,0.7);
        border: 1px solid {border_color}33;
        border-top: 2px solid {border_color};
        border-radius: 12px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 24px {glow_color}22, inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:-20px;right:-10px;font-size:64px;opacity:0.08;">{icon}</div>
        <div style="color:#94a3b8;font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">{title}</div>
        <div style="color:#f1f5f9;font-size:36px;font-weight:800;font-family:'Space Grotesk',sans-serif;line-height:1;margin-bottom:6px;">{value}</div>
        <div style="color:{border_color};font-size:12px;font-weight:500;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load Data ──
    df, baseline, classifier = load_data_and_models()
    if df is None:
        return

    # ── Process Alerts ──
    df_recent = df.tail(200).copy()
    alerts = []
    explainer = AlertExplainer(classifier.model, classifier.features)

    for _, row in df_recent.iterrows():
        score, is_high_conf = baseline.score_session(row)
        if score > 0.4:
            X_feats = classifier._engineer_features(pd.DataFrame([row]))
            missing_cols = [col for col in classifier.features if col not in X_feats.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=X_feats.index, columns=missing_cols)
                X_feats = pd.concat([X_feats, missing_df], axis=1)
            X_feats = X_feats[classifier.features].astype(float)
            pred_type = classifier.model.predict(X_feats)[0]
            reason = explainer.explain_alert(X_feats, pred_type)
            alerts.append({
                'Timestamp': row['timestamp'],
                'Entity': row['entity_id'],
                'Entity Type': row['entity_type'],
                'Risk Score': score,
                'Predicted Type': pred_type,
                'Explanation': reason,
                'Confidence': 'High' if is_high_conf else 'Low (Cold-Start)',
                '_raw_score': score,
            })

    df_alerts = pd.DataFrame(alerts)

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 24px 0;border-bottom:1px solid rgba(59,130,246,0.2);margin-bottom:24px;">
            <div style="font-size:40px;margin-bottom:8px;">🛡️</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;letter-spacing:-0.5px;">Sentinel-AI</div>
            <div style="font-size:11px;color:#3b82f6;font-weight:500;letter-spacing:2px;margin-top:4px;">SOC DASHBOARD</div>
            <div style="margin-top:10px;display:flex;align-items:center;justify-content:center;gap:6px;">
                <div style="width:8px;height:8px;background:#10b981;border-radius:50%;box-shadow:0 0 8px #10b981;animation:pulse 2s infinite;"></div>
                <span style="color:#10b981;font-size:11px;font-weight:600;">SYSTEM ACTIVE</span>
            </div>
        </div>
        <style>@keyframes pulse {0%,100%{opacity:1}50%{opacity:0.4}}</style>
        """, unsafe_allow_html=True)

        st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:600;letter-spacing:2px;margin-bottom:12px;">⚙ FILTERS</div>', unsafe_allow_html=True)

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Minimum Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect(
                "Anomaly Types",
                options=all_types,
                default=all_types
            )

        # Mini threat distribution donut chart
        if not df_alerts.empty:
            st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:600;letter-spacing:2px;margin:20px 0 12px 0;">📊 THREAT DISTRIBUTION</div>', unsafe_allow_html=True)
            dist = df_alerts['Predicted Type'].value_counts()
            colors = [THREAT_CONFIG.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_donut = go.Figure(go.Pie(
                labels=dist.index,
                values=dist.values,
                hole=0.65,
                marker_colors=colors,
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
            ))
            fig_donut.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=160,
            )
            fig_donut.add_annotation(
                text=f"<b>{len(df_alerts)}</b><br><span style='font-size:10px'>Alerts</span>",
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16, color='#e2e8f0'),
                align='center'
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        # Model stats
        st.markdown("""
        <div style="margin-top:16px;padding:16px;background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.2);border-radius:10px;">
            <div style="color:#94a3b8;font-size:10px;font-weight:600;letter-spacing:2px;margin-bottom:12px;">🤖 MODEL STATS</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#94a3b8;font-size:12px;">Precision</span>
                <span style="color:#10b981;font-weight:700;font-size:12px;">92%</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#94a3b8;font-size:12px;">Recall</span>
                <span style="color:#10b981;font-weight:700;font-size:12px;">88%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#94a3b8;font-size:12px;">Algorithm</span>
                <span style="color:#3b82f6;font-weight:600;font-size:12px;">Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # HERO HEADER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(8,145,178,0.05) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:0;right:0;width:300px;height:300px;
            background:radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
            pointer-events:none;"></div>
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
            <div style="font-size:48px;filter:drop-shadow(0 0 16px rgba(59,130,246,0.6));">🛡️</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:800;
                    background:linear-gradient(90deg,#e2e8f0 0%,#93c5fd 60%,#06b6d4 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
                    letter-spacing:-1px;line-height:1.1;">
                    Sentinel-AI
                </div>
                <div style="color:#94a3b8;font-size:14px;font-weight:400;margin-top:4px;letter-spacing:0.3px;">
                    AI-Powered Behavioral Anomaly Detection for Cybersecurity
                </div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-top:8px;">
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:8px;height:8px;background:#10b981;border-radius:50%;box-shadow:0 0 8px #10b981;"></div>
                <span style="color:#10b981;font-size:12px;font-weight:600;letter-spacing:1px;">LIVE MONITORING</span>
            </div>
            <div style="color:#475569;font-size:12px;">|</div>
            <span style="color:#64748b;font-size:12px;">SHAP Explainability Layer Active</span>
            <div style="color:#475569;font-size:12px;">|</div>
            <span style="color:#64748b;font-size:12px;">Cold-Start Detection Enabled</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CARDS
    # ──────────────────────────────────────────────────────────────────────────
    high_risk_count = len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0
    total_alerts = len(df_alerts)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Sessions Analyzed", f"{len(df_recent):,}", "Last 200 events processed",
                 "🔍", "#3b82f6", "#3b82f6")
    with c2:
        kpi_card("Total Alerts", str(total_alerts),
                 "Flagged for investigation" if total_alerts > 0 else "All clear",
                 "🚨", "#f59e0b", "#f59e0b")
    with c3:
        kpi_card("High Risk Alerts", str(high_risk_count),
                 "Score > 0.80 — critical" if high_risk_count > 0 else "No critical threats",
                 "⚠️", "#ef4444", "#ef4444")
    with c4:
        kpi_card("Model Precision", "92% / 88%", "Precision / Recall on holdout",
                 "🤖", "#10b981", "#10b981")

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # FILTER & PREPARE DATA
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
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
        <span style="font-size:20px;">🚨</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;">Alert Queue</span>
        <span style="background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3);
            padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:1px;margin-left:6px;">
            LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not df_filtered.empty:
        # Build HTML table
        rows_html = ""
        for _, row in df_filtered.iterrows():
            threat_badge = get_threat_badge(row['Predicted Type'])
            risk_bar = get_risk_bar(row['Risk Score'])
            conf_badge = get_confidence_badge(row['Confidence'])
            entity_type_color = {"user": "#3b82f6", "service_account": "#a855f7", "edge_device": "#06b6d4"}.get(row['Entity Type'], "#64748b")
            short_entity = row['Entity'][:24] + '…' if len(str(row['Entity'])) > 24 else row['Entity']
            short_ts = str(row['Timestamp'])[:16]
            explanation = row['Explanation'][:90] + '…' if len(str(row['Explanation'])) > 90 else row['Explanation']

            rows_html += f"""
            <tr style="border-bottom:1px solid rgba(99,179,237,0.08);transition:background 0.2s;"
                onmouseover="this.style.background='rgba(59,130,246,0.05)'"
                onmouseout="this.style.background='transparent'">
                <td style="padding:12px 16px;color:#64748b;font-size:12px;font-family:monospace;">{short_ts}</td>
                <td style="padding:12px 16px;">
                    <div style="color:#e2e8f0;font-size:12px;font-family:monospace;font-weight:500;">{short_entity}</div>
                    <div style="color:{entity_type_color};font-size:10px;font-weight:600;letter-spacing:1px;margin-top:2px;">{row['Entity Type'].upper()}</div>
                </td>
                <td style="padding:12px 16px;">{threat_badge}</td>
                <td style="padding:12px 16px;min-width:140px;">{risk_bar}</td>
                <td style="padding:12px 16px;color:#94a3b8;font-size:12px;max-width:320px;">{explanation}</td>
                <td style="padding:12px 16px;">{conf_badge}</td>
            </tr>"""

        st.markdown(f"""
        <div style="
            background:rgba(8,16,32,0.6);
            border:1px solid rgba(59,130,246,0.15);
            border-radius:14px;
            overflow:hidden;
            backdrop-filter:blur(20px);
            box-shadow:0 4px 24px rgba(0,0,0,0.4);
        ">
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="background:rgba(59,130,246,0.08);border-bottom:1px solid rgba(59,130,246,0.2);">
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">TIMESTAMP</th>
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">ENTITY</th>
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">THREAT TYPE</th>
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">RISK SCORE</th>
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">AI EXPLANATION</th>
                        <th style="padding:14px 16px;text-align:left;color:#64748b;font-size:10px;font-weight:700;letter-spacing:1.5px;">CONFIDENCE</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;background:rgba(8,16,32,0.6);
            border:1px solid rgba(59,130,246,0.15);border-radius:14px;">
            <div style="font-size:48px;margin-bottom:12px;">✅</div>
            <div style="color:#10b981;font-size:18px;font-weight:600;">No Threats Detected</div>
            <div style="color:#64748b;font-size:14px;margin-top:8px;">All sessions are within normal behavioral bounds.</div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <span style="font-size:20px;">🔎</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;">Entity Deep-Dive</span>
        </div>
        """, unsafe_allow_html=True)

        # Styled selectbox container
        selected_entity = st.selectbox(
            "Select a flagged entity to investigate",
            options=df_filtered['Entity'].unique(),
            key="entity_select"
        )

        if selected_entity:
            entity_history = df[df['entity_id'] == selected_entity].copy()
            entity_row = df_filtered[df_filtered['Entity'] == selected_entity].iloc[0]
            threat_cfg = THREAT_CONFIG.get(entity_row['Predicted Type'], {"color": "#64748b", "icon": "⚪", "label": entity_row['Predicted Type']})
            entity_type_color = {"user": "#3b82f6", "service_account": "#a855f7", "edge_device": "#06b6d4"}.get(entity_row['Entity Type'], "#64748b")

            # Entity Profile Card
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(8,16,32,0.8) 100%);
                border:1px solid rgba(59,130,246,0.2);
                border-left:4px solid {threat_cfg['color']};
                border-radius:12px;
                padding:20px 24px;
                margin-bottom:20px;
                display:flex;
                gap:40px;
                align-items:center;
                flex-wrap:wrap;
            ">
                <div>
                    <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:4px;">ENTITY ID</div>
                    <div style="color:#e2e8f0;font-size:14px;font-family:monospace;font-weight:600;">{selected_entity}</div>
                </div>
                <div>
                    <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:4px;">ENTITY TYPE</div>
                    <div style="color:{entity_type_color};font-size:13px;font-weight:700;">{entity_row['Entity Type'].upper()}</div>
                </div>
                <div>
                    <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:4px;">PREDICTED THREAT</div>
                    <div>{get_threat_badge(entity_row['Predicted Type'])}</div>
                </div>
                <div>
                    <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:4px;">RISK SCORE</div>
                    <div style="color:{threat_cfg['color']};font-size:20px;font-weight:800;font-family:'Space Grotesk',sans-serif;">{entity_row['Risk Score']:.2f}</div>
                </div>
                <div>
                    <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:4px;">SESSIONS IN HISTORY</div>
                    <div style="color:#e2e8f0;font-size:20px;font-weight:800;font-family:'Space Grotesk',sans-serif;">{len(entity_history)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Two-column layout for charts ──
            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Timeline chart — custom dark Plotly theme
                auth_colors = {"success": "#10b981", "failed": "#ef4444"}
                entity_history['timestamp'] = pd.to_datetime(entity_history['timestamp'], errors='coerce')
                fig_timeline = px.scatter(
                    entity_history.dropna(subset=['timestamp']),
                    x='timestamp', y='session_duration',
                    color='auth_status',
                    color_discrete_map=auth_colors,
                    size_max=12,
                    title="Session Duration Timeline",
                    labels={'timestamp': 'Date', 'session_duration': 'Duration (s)', 'auth_status': 'Auth Status'},
                    hover_data=['geo_location', 'resource_accessed', 'session_duration']
                )
                fig_timeline.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))
                fig_timeline.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8', size=12),
                    title=dict(font=dict(family='Space Grotesk', size=16, color='#e2e8f0')),
                    xaxis=dict(gridcolor='rgba(99,179,237,0.08)', linecolor='rgba(99,179,237,0.2)', tickfont=dict(color='#64748b')),
                    yaxis=dict(gridcolor='rgba(99,179,237,0.08)', linecolor='rgba(99,179,237,0.2)', tickfont=dict(color='#64748b')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(99,179,237,0.2)', borderwidth=1, font=dict(color='#94a3b8')),
                    margin=dict(t=48, b=0, l=0, r=0),
                    height=300,
                )
                st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})

            with col_right:
                # Baseline Profile Card
                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                if prof:
                    typical_geos = list(prof['geo_dist'].keys())[:3]
                    typical_resources = list(prof['resource_dist'].keys())[:3]
                    avg_duration = round(prof['duration_mean'], 1)

                    geo_items = "".join([f'<div style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.2);color:#06b6d4;padding:3px 10px;border-radius:6px;font-size:11px;margin-bottom:4px;font-weight:500;">📍 {g}</div>' for g in typical_geos])
                    res_items = "".join([f'<div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.2);color:#a855f7;padding:3px 10px;border-radius:6px;font-size:11px;margin-bottom:4px;font-weight:500;">📂 {r}</div>' for r in typical_resources])

                    st.markdown(f"""
                    <div style="background:rgba(8,16,32,0.7);border:1px solid rgba(59,130,246,0.15);border-radius:12px;padding:20px;height:300px;overflow-y:auto;">
                        <div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:16px;">📋 BEHAVIORAL BASELINE</div>
                        
                        <div style="margin-bottom:14px;">
                            <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;">TYPICAL LOCATIONS</div>
                            {geo_items}
                        </div>
                        
                        <div style="margin-bottom:14px;">
                            <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;">TYPICAL RESOURCES</div>
                            {res_items}
                        </div>
                        
                        <div>
                            <div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;">AVG SESSION DURATION</div>
                            <div style="color:#f59e0b;font-size:20px;font-weight:800;font-family:'Space Grotesk',sans-serif;">{avg_duration}s</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Hourly Activity Heatmap ──
            if 'timestamp' in entity_history.columns:
                entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)
                fig_hourly = go.Figure(go.Bar(
                    x=list(range(24)),
                    y=hourly.values,
                    marker=dict(
                        color=hourly.values,
                        colorscale=[[0, 'rgba(59,130,246,0.3)'], [0.5, '#3b82f6'], [1.0, '#ef4444']],
                        line=dict(width=0)
                    ),
                    hovertemplate='Hour %{x}:00<br>Sessions: %{y}<extra></extra>'
                ))
                fig_hourly.update_layout(
                    title=dict(text="Hourly Activity Pattern", font=dict(family='Space Grotesk', size=16, color='#e2e8f0')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#94a3b8', size=12),
                    xaxis=dict(title='Hour of Day', gridcolor='rgba(99,179,237,0.06)', linecolor='rgba(99,179,237,0.2)', tickfont=dict(color='#64748b')),
                    yaxis=dict(title='Sessions', gridcolor='rgba(99,179,237,0.06)', linecolor='rgba(99,179,237,0.2)', tickfont=dict(color='#64748b')),
                    margin=dict(t=48, b=0, l=0, r=0),
                    height=220,
                )
                st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar': False})

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        margin-top:48px;
        padding:20px;
        border-top:1px solid rgba(59,130,246,0.1);
        text-align:center;
    ">
        <span style="color:#334155;font-size:12px;">
            Powered by <b style="color:#3b82f6;">Sentinel-AI</b> &nbsp;·&nbsp; 
            Random Forest + SHAP Explainability &nbsp;·&nbsp; 
            Built for Real-Time SOC Operations
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
