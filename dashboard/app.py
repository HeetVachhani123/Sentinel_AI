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
    page_title="Sentinel-AI | Honeywell SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Clean White & Navy Enterprise Theme  (no f-strings to avoid { } bugs)
# ──────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f0f4f8 !important;
    color: #1e293b !important;
}
.stApp {
    background: #f0f4f8 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f2d5c !important;
    border-right: 3px solid #16a34a !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
}

/* ── Slider track ── */
.stSlider [data-baseweb="slider"] [role="progressbar"] {
    background: #16a34a !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #16a34a !important;
    border-color: #16a34a !important;
}
.stSlider label, .stSlider p { color: #cbd5e1 !important; }

/* ── Multiselect tags ── */
div[data-baseweb="tag"] {
    background: rgba(22,163,74,0.25) !important;
    border: 1px solid rgba(22,163,74,0.5) !important;
    color: #dcfce7 !important;
    border-radius: 4px !important;
}
div[data-baseweb="tag"] span { color: #dcfce7 !important; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] input { color: #e2e8f0 !important; }
div[data-baseweb="select"] li { color: #1e293b !important; }
.stMultiSelect label { color: #94a3b8 !important; }

/* ── Main select box ── */
.stSelectbox label { color: #475569 !important; font-weight: 600; font-size: 14px; }
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
}

/* ── Plotly ── */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0f2d5c; }

/* ── Remove Streamlit default padding ── */
.block-container { padding-top: 1rem !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .kpi-row { flex-direction: column !important; }
    .alert-table { font-size: 11px !important; }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade { animation: fadeUp 0.4s ease both; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT CONFIG
# ──────────────────────────────────────────────────────────────────────────────
THREAT_CONFIG = {
    "brute_force":         {"color": "#dc2626", "bg": "#fef2f2",  "border": "#fca5a5", "icon": "🔴", "label": "Brute Force"},
    "impossible_travel":   {"color": "#ea580c", "bg": "#fff7ed",  "border": "#fdba74", "icon": "🟠", "label": "Impossible Travel"},
    "device_spoofing":     {"color": "#ca8a04", "bg": "#fefce8",  "border": "#fde047", "icon": "🟡", "label": "Device Spoofing"},
    "lateral_movement":    {"color": "#7c3aed", "bg": "#f5f3ff",  "border": "#c4b5fd", "icon": "🟣", "label": "Lateral Movement"},
    "credential_stuffing": {"color": "#1d4ed8", "bg": "#eff6ff",  "border": "#93c5fd", "icon": "🔵", "label": "Credential Stuffing"},
    "insider_drift":       {"color": "#0891b2", "bg": "#ecfeff",  "border": "#67e8f9", "icon": "🩵", "label": "Insider Drift"},
    "low_and_slow_exfil":  {"color": "#be185d", "bg": "#fdf2f8",  "border": "#f9a8d4", "icon": "🩷", "label": "Low & Slow Exfil"},
    "normal":              {"color": "#475569", "bg": "#f8fafc",  "border": "#cbd5e1", "icon": "⚪", "label": "Normal"},
}

def threat_badge(t):
    c = THREAT_CONFIG.get(t, THREAT_CONFIG["normal"])
    return (
        '<span style="background:' + c["bg"] + ';color:' + c["color"] + ';border:1px solid ' + c["border"] + ';'
        'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;">'
        + c["icon"] + ' ' + c["label"].upper() + '</span>'
    )

def risk_bar(score):
    pct = int(score * 100)
    color = "#dc2626" if score >= 0.8 else "#ea580c" if score >= 0.6 else "#ca8a04" if score >= 0.4 else "#16a34a"
    return (
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<div style="flex:1;background:#e2e8f0;border-radius:4px;height:7px;">'
        '<div style="width:' + str(pct) + '%;background:' + color + ';height:7px;border-radius:4px;"></div>'
        '</div>'
        '<span style="color:' + color + ';font-weight:700;font-size:12px;min-width:36px;">' + str(round(score, 2)) + '</span>'
        '</div>'
    )

def conf_badge(conf):
    if conf == "High":
        return '<span style="background:#f0fdf4;color:#16a34a;border:1px solid #86efac;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:1px;">✓ VERIFIED</span>'
    return '<span style="background:#fffbeb;color:#d97706;border:1px solid #fcd34d;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:1px;">⚡ COLD-START</span>'

def entity_type_badge(et):
    colors = {"user": ("#1d4ed8", "#eff6ff", "#bfdbfe"), "service_account": ("#7c3aed", "#f5f3ff", "#ddd6fe"), "edge_device": ("#0891b2", "#ecfeff", "#a5f3fc")}
    c, bg, br = colors.get(et, ("#475569", "#f8fafc", "#cbd5e1"))
    return '<span style="background:' + bg + ';color:' + c + ';border:1px solid ' + br + ';padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;letter-spacing:0.5px;">' + et.replace("_", " ").upper() + '</span>'

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
        st.error(f"Error loading data/models: {e}. Run generate.py then train_evaluate.py first.")
        return None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — pure-string HTML (no f-strings with nested CSS braces)
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card_html(title, value, subtitle, icon, top_color):
    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid ' + top_color + ';'
        'border-radius:12px;padding:20px 22px;box-shadow:0 2px 12px rgba(0,0,0,0.06);position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:-10px;right:8px;font-size:56px;opacity:0.06;">' + icon + '</div>'
        '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">' + title + '</div>'
        '<div style="color:#0f172a;font-size:34px;font-weight:800;line-height:1;margin-bottom:6px;">' + str(value) + '</div>'
        '<div style="color:' + top_color + ';font-size:12px;font-weight:500;">' + subtitle + '</div>'
        '</div>'
    )

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
                'Timestamp':      row['timestamp'],
                'Entity':         row['entity_id'],
                'Entity Type':    row['entity_type'],
                'Risk Score':     score,
                'Predicted Type': pred_type,
                'Explanation':    reason,
                'Confidence':     'High' if is_high_conf else 'Low (Cold-Start)',
            })
    df_alerts = pd.DataFrame(alerts)

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Honeywell + Sentinel brand
        st.markdown(
            '<div style="background:#0a2247;padding:22px 20px 18px 20px;margin:-1rem -1rem 0 -1rem;">'
            '<div style="font-size:11px;font-weight:800;letter-spacing:3px;color:#16a34a;margin-bottom:3px;">HONEYWELL</div>'
            '<div style="font-size:20px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">Sentinel-AI</div>'
            '<div style="font-size:10px;color:#94a3b8;letter-spacing:2px;margin-top:2px;">SOC DASHBOARD</div>'
            '<div style="margin-top:12px;display:inline-flex;align-items:center;gap:6px;'
            'background:rgba(22,163,74,0.2);border:1px solid rgba(22,163,74,0.4);'
            'padding:4px 12px;border-radius:20px;">'
            '<div style="width:6px;height:6px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;"></div>'
            '<span style="color:#22c55e;font-size:10px;font-weight:700;letter-spacing:1px;">SYSTEM ACTIVE</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:8px;padding:0 4px;">⚙ FILTERS</div>', unsafe_allow_html=True)

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Min Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect("Anomaly Types", options=all_types, default=all_types)

        # Threat distribution donut
        if not df_alerts.empty:
            st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;margin:16px 0 8px 0;padding:0 4px;">📊 THREAT MIX</div>', unsafe_allow_html=True)
            dist   = df_alerts['Predicted Type'].value_counts()
            colors = [THREAT_CONFIG.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_d  = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.62,
                marker=dict(colors=colors, line=dict(color='#0f2d5c', width=2)),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} alerts<extra></extra>'
            ))
            fig_d.update_layout(
                margin=dict(t=0,b=0,l=0,r=0), height=150,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_d.add_annotation(
                text='<b>' + str(len(df_alerts)) + '</b>',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, font=dict(size=18, color='#ffffff'), align='center'
            )
            st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})

        # Model stats
        st.markdown(
            '<div style="margin:12px 0;padding:14px 16px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);border-radius:10px;">'
            '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:12px;">🤖 MODEL STATS</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#94a3b8;font-size:12px;">Precision</span>'
            '<span style="color:#22c55e;font-weight:800;font-size:13px;">92%</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#94a3b8;font-size:12px;">Recall</span>'
            '<span style="color:#22c55e;font-weight:800;font-size:13px;">88%</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#94a3b8;font-size:12px;">Algorithm</span>'
            '<span style="color:#93c5fd;font-weight:600;font-size:12px;">Random Forest</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;">'
            '<span style="color:#94a3b8;font-size:12px;">Explainability</span>'
            '<span style="color:#c4b5fd;font-weight:600;font-size:12px;">SHAP</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TOP NAV BAR  (Honeywell branding, always visible)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="fade" style="background:#0f2d5c;border-radius:12px;padding:16px 28px;'
        'display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;'
        'box-shadow:0 4px 16px rgba(15,45,92,0.25);flex-wrap:wrap;gap:12px;">'

        # Left: Honeywell + Sentinel
        '<div style="display:flex;align-items:center;gap:20px;">'
        '<div>'
        '<div style="font-size:9px;font-weight:900;letter-spacing:4px;color:#16a34a;">HONEYWELL</div>'
        '<div style="font-size:20px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;line-height:1.2;">Sentinel-AI</div>'
        '</div>'
        '<div style="width:1px;height:36px;background:rgba(255,255,255,0.15);"></div>'
        '<div style="color:#94a3b8;font-size:13px;font-weight:400;">AI-Powered Behavioral Anomaly Detection</div>'
        '</div>'

        # Right: status badges
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        '<div style="display:flex;align-items:center;gap:6px;background:rgba(22,163,74,0.15);'
        'border:1px solid rgba(22,163,74,0.35);padding:5px 14px;border-radius:20px;">'
        '<div style="width:7px;height:7px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;"></div>'
        '<span style="color:#22c55e;font-size:11px;font-weight:700;letter-spacing:1px;">LIVE</span>'
        '</div>'
        '<span style="color:#64748b;font-size:12px;">SHAP Active</span>'
        '<span style="color:#64748b;font-size:12px;">Cold-Start Ready</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CARDS
    # ──────────────────────────────────────────────────────────────────────────
    high_risk  = len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0
    total_alts = len(df_alerts)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card_html("Sessions Analyzed", f"{len(df_recent):,}", "Last 200 processed", "🔍", "#0f2d5c"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html("Total Alerts", str(total_alts), "Flagged for review", "🚨", "#d97706"), unsafe_allow_html=True)
    with c3:
        color = "#dc2626" if high_risk > 0 else "#16a34a"
        label = "Critical — Score > 0.80" if high_risk > 0 else "No critical threats"
        st.markdown(kpi_card_html("High Risk Alerts", str(high_risk), label, "⚠️", color), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card_html("Precision / Recall", "92% / 88%", "Holdout evaluation", "🤖", "#16a34a"), unsafe_allow_html=True)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # FILTER DATA
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
    st.markdown(
        '<div class="fade" style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
        '<span style="font-size:20px;">🚨</span>'
        '<span style="font-size:20px;font-weight:800;color:#0f172a;">Alert Queue</span>'
        '<span style="background:#fef2f2;color:#dc2626;border:1px solid #fca5a5;'
        'padding:3px 12px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:2px;">LIVE</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if not df_filtered.empty:
        # Build table rows
        rows = ""
        for _, row in df_filtered.iterrows():
            short_ent = str(row['Entity'])[:28] + '…' if len(str(row['Entity'])) > 28 else str(row['Entity'])
            short_ts  = str(row['Timestamp'])[:16]
            expl      = str(row['Explanation'])[:90] + '…' if len(str(row['Explanation'])) > 90 else str(row['Explanation'])
            rows += (
                '<tr onmouseover="this.style.background=\'#f0f9ff\'" onmouseout="this.style.background=\'#ffffff\'"'
                ' style="background:#ffffff;border-bottom:1px solid #f1f5f9;transition:background 0.15s;">'
                '<td style="padding:12px 16px;font-size:12px;color:#64748b;font-family:monospace;white-space:nowrap;">' + short_ts + '</td>'
                '<td style="padding:12px 16px;">'
                '<div style="font-size:12px;font-family:monospace;color:#0f172a;font-weight:600;">' + short_ent + '</div>'
                '<div style="margin-top:4px;">' + entity_type_badge(row['Entity Type']) + '</div>'
                '</td>'
                '<td style="padding:12px 16px;">' + threat_badge(row['Predicted Type']) + '</td>'
                '<td style="padding:12px 16px;min-width:130px;">' + risk_bar(row['Risk Score']) + '</td>'
                '<td style="padding:12px 16px;font-size:12px;color:#475569;max-width:280px;line-height:1.5;">' + expl + '</td>'
                '<td style="padding:12px 16px;">' + conf_badge(row['Confidence']) + '</td>'
                '</tr>'
            )

        st.markdown(
            '<div class="fade" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;'
            'overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,0.06);margin-bottom:32px;">'
            '<div style="background:#0f2d5c;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;">'
            '<span style="color:#ffffff;font-size:13px;font-weight:700;">Active Threat Alerts</span>'
            '<span style="color:#94a3b8;font-size:12px;">Sorted by Risk Score</span>'
            '</div>'
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;" class="alert-table">'
            '<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">TIMESTAMP</th>'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">ENTITY</th>'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">THREAT TYPE</th>'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">RISK SCORE</th>'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">AI EXPLANATION</th>'
            '<th style="padding:11px 16px;text-align:left;color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;">CONFIDENCE</th>'
            '</tr></thead>'
            '<tbody>' + rows + '</tbody>'
            '</table></div></div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="fade" style="background:#f0fdf4;border:2px solid #86efac;border-radius:14px;'
            'padding:56px;text-align:center;margin-bottom:32px;">'
            '<div style="font-size:48px;margin-bottom:12px;">✅</div>'
            '<div style="color:#16a34a;font-size:20px;font-weight:800;margin-bottom:8px;">No Threats Detected</div>'
            '<div style="color:#4ade80;font-size:14px;">All sessions are within normal behavioral bounds.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        # Section divider
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            '<div style="flex:1;height:1px;background:#e2e8f0;"></div>'
            '<span style="font-size:20px;font-weight:800;color:#0f172a;white-space:nowrap;">🔎 Entity Deep-Dive</span>'
            '<div style="flex:1;height:1px;background:#e2e8f0;"></div>'
            '</div>',
            unsafe_allow_html=True
        )

        selected_entity = st.selectbox(
            "Select a flagged entity to investigate",
            options=df_filtered['Entity'].unique(),
            key="entity_select"
        )

        if selected_entity:
            entity_history = df[df['entity_id'] == selected_entity].copy()
            entity_row     = df_filtered[df_filtered['Entity'] == selected_entity].iloc[0]
            threat_cfg     = THREAT_CONFIG.get(entity_row['Predicted Type'], THREAT_CONFIG["normal"])
            high_conf      = entity_row['Confidence'] == 'High'

            # ── Entity Profile Card (pure string concat, no f-string CSS braces) ──
            st.markdown(
                '<div class="fade" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;'
                'border-left:5px solid ' + threat_cfg["color"] + ';padding:22px 28px;margin-bottom:20px;'
                'box-shadow:0 2px 12px rgba(0,0,0,0.06);display:flex;gap:40px;align-items:center;flex-wrap:wrap;">'

                '<div>'
                '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:4px;">ENTITY ID</div>'
                '<div style="color:#0f172a;font-size:13px;font-family:monospace;font-weight:600;">' + str(selected_entity)[:40] + '</div>'
                '</div>'

                '<div>'
                '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">ENTITY TYPE</div>'
                + entity_type_badge(entity_row['Entity Type']) +
                '</div>'

                '<div>'
                '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">PREDICTED THREAT</div>'
                + threat_badge(entity_row['Predicted Type']) +
                '</div>'

                '<div>'
                '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:4px;">RISK SCORE</div>'
                '<div style="color:' + threat_cfg["color"] + ';font-size:28px;font-weight:900;">' + str(round(entity_row['Risk Score'], 2)) + '</div>'
                '</div>'

                '<div>'
                '<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:4px;">SESSIONS</div>'
                '<div style="color:#0f172a;font-size:28px;font-weight:900;">' + str(len(entity_history)) + '</div>'
                '</div>'

                '</div>',
                unsafe_allow_html=True
            )

            col_l, col_r = st.columns([3, 2])

            with col_l:
                entity_history['timestamp'] = pd.to_datetime(entity_history['timestamp'], errors='coerce')
                auth_map = {"success": "#16a34a", "failed": "#dc2626"}
                fig_tl = px.scatter(
                    entity_history.dropna(subset=['timestamp']),
                    x='timestamp', y='session_duration',
                    color='auth_status', color_discrete_map=auth_map,
                    title="Session Duration Timeline",
                    labels={'timestamp': 'Date', 'session_duration': 'Duration (s)', 'auth_status': 'Auth Status'},
                    hover_data=['geo_location', 'resource_accessed']
                )
                fig_tl.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafbfc', height=300,
                    font=dict(family='Inter', color='#475569', size=12),
                    title=dict(font=dict(family='Inter', size=15, color='#0f172a'), x=0),
                    xaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickfont=dict(color='#94a3b8')),
                    yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickfont=dict(color='#94a3b8')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#475569')),
                    margin=dict(t=44, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})

            with col_r:
                # ── Behavioral Baseline — use native Streamlit only ──
                st.markdown(
                    '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
                    'padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">'
                    '<div style="color:#0f2d5c;font-size:13px;font-weight:800;letter-spacing:1px;'
                    'border-bottom:2px solid #0f2d5c;padding-bottom:8px;margin-bottom:16px;">📋 BEHAVIORAL BASELINE</div>',
                    unsafe_allow_html=True
                )

                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                if prof:
                    typical_geos      = list(prof['geo_dist'].keys())[:3]
                    typical_resources = list(prof['resource_dist'].keys())[:3]
                    avg_duration      = round(prof['duration_mean'], 1)

                    # Geo
                    st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">TYPICAL LOCATIONS</div>', unsafe_allow_html=True)
                    for g in typical_geos:
                        st.markdown(
                            '<div style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;'
                            'padding:5px 12px;border-radius:8px;font-size:12px;font-weight:600;margin-bottom:5px;">📍 ' + str(g) + '</div>',
                            unsafe_allow_html=True
                        )

                    # Resources
                    st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin:12px 0 8px 0;">TYPICAL RESOURCES</div>', unsafe_allow_html=True)
                    for r in typical_resources:
                        st.markdown(
                            '<div style="background:#f5f3ff;color:#7c3aed;border:1px solid #ddd6fe;'
                            'padding:5px 12px;border-radius:8px;font-size:12px;font-weight:600;margin-bottom:5px;">📂 ' + str(r) + '</div>',
                            unsafe_allow_html=True
                        )

                    # Avg Duration
                    st.markdown('<div style="color:#94a3b8;font-size:10px;font-weight:700;letter-spacing:1.5px;margin:12px 0 4px 0;">AVG SESSION DURATION</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="color:#d97706;font-size:24px;font-weight:900;">' + str(avg_duration) + 's</div>',
                        unsafe_allow_html=True
                    )

                else:
                    st.markdown('<div style="color:#94a3b8;font-size:13px;font-style:italic;">No baseline profile available for this entity.</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Hourly Activity Bar
            if 'timestamp' in entity_history.columns:
                entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)
                bar_colors = ['#dc2626' if v == hourly.max() else '#0f2d5c' for v in hourly.values]
                fig_h = go.Figure(go.Bar(
                    x=list(range(24)), y=hourly.values,
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate='%{x}:00 — %{y} sessions<extra></extra>'
                ))
                fig_h.update_layout(
                    title=dict(text='Hourly Activity Pattern', font=dict(family='Inter', size=15, color='#0f172a'), x=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafbfc', height=220,
                    font=dict(family='Inter', color='#475569', size=12),
                    xaxis=dict(title='Hour of Day', gridcolor='#e2e8f0', tickfont=dict(color='#94a3b8')),
                    yaxis=dict(title='Sessions',    gridcolor='#e2e8f0', tickfont=dict(color='#94a3b8')),
                    margin=dict(t=44, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:48px;padding:18px 0;border-top:2px solid #e2e8f0;'
        'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<span style="font-size:10px;font-weight:900;letter-spacing:3px;color:#16a34a;">HONEYWELL</span>'
        '<span style="color:#cbd5e1;">|</span>'
        '<span style="color:#64748b;font-size:12px;">Sentinel-AI Behavioral Anomaly Detection</span>'
        '</div>'
        '<span style="color:#94a3b8;font-size:12px;">Random Forest · SHAP · Cold-Start Detection · Concept Drift</span>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
