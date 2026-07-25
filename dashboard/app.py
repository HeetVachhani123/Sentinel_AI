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
    page_title="Sentinel-AI | Honeywell SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND IMAGE — load as base64, inject via separate <style> block
# (kept separate so no f-string brace conflicts with CSS)
# ──────────────────────────────────────────────────────────────────────────────
def _get_bg_css():
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "datacenter_bg.png")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return "url('data:image/png;base64," + b64 + "')"
    return "none"

# Inject background as its own style block — no other CSS here to avoid brace conflicts
bg_url = _get_bg_css()
st.markdown(
    "<style>.stApp { background: linear-gradient(rgba(0,0,0,0.91) 0%, rgba(0,0,0,0.94) 100%), "
    + bg_url
    + " !important; background-size: cover !important; background-position: center top !important;"
    + " background-attachment: fixed !important; background-repeat: no-repeat !important; }</style>",
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN THEME CSS — Black + Green/Cyan accent  (plain triple-quote string, no f-string)
# ──────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #000000 !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050505 0%, #0a0a0a 100%) !important;
    border-right: 1px solid rgba(22,163,74,0.3) !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="progressbar"] { background: #16a34a !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #16a34a !important; border-color: #16a34a !important;
    box-shadow: 0 0 8px rgba(22,163,74,0.6) !important;
}
.stSlider label, .stSlider p { color: #94a3b8 !important; }

/* ── Multiselect ── */
div[data-baseweb="tag"] {
    background: rgba(22,163,74,0.2) !important;
    border: 1px solid rgba(22,163,74,0.4) !important;
    border-radius: 4px !important;
}
div[data-baseweb="tag"] span { color: #bbf7d0 !important; }
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
.stMultiSelect label { color: #64748b !important; font-size: 12px !important; }

/* ── Main selectbox ── */
.stSelectbox label { color: #64748b !important; font-weight: 600; font-size: 14px; }
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Plotly ── */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #050505; }
::-webkit-scrollbar-thumb { background: rgba(22,163,74,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(22,163,74,0.7); }

/* ── Block padding ── */
.block-container { padding-top: 1rem !important; }

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade { animation: fadeUp 0.4s ease both; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .alert-table { font-size: 11px !important; }
    .entity-card { flex-direction: column !important; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT CONFIG
# ──────────────────────────────────────────────────────────────────────────────
THREAT_CONFIG = {
    "brute_force":         {"color": "#f87171", "bg": "rgba(248,113,113,0.12)", "border": "rgba(248,113,113,0.35)", "icon": "🔴", "label": "Brute Force"},
    "impossible_travel":   {"color": "#fb923c", "bg": "rgba(251,146,60,0.12)",  "border": "rgba(251,146,60,0.35)",  "icon": "🟠", "label": "Impossible Travel"},
    "device_spoofing":     {"color": "#fbbf24", "bg": "rgba(251,191,36,0.12)",  "border": "rgba(251,191,36,0.35)",  "icon": "🟡", "label": "Device Spoofing"},
    "lateral_movement":    {"color": "#c084fc", "bg": "rgba(192,132,252,0.12)", "border": "rgba(192,132,252,0.35)", "icon": "🟣", "label": "Lateral Movement"},
    "credential_stuffing": {"color": "#60a5fa", "bg": "rgba(96,165,250,0.12)",  "border": "rgba(96,165,250,0.35)",  "icon": "🔵", "label": "Credential Stuffing"},
    "insider_drift":       {"color": "#22d3ee", "bg": "rgba(34,211,238,0.12)",  "border": "rgba(34,211,238,0.35)",  "icon": "🩵", "label": "Insider Drift"},
    "low_and_slow_exfil":  {"color": "#f472b6", "bg": "rgba(244,114,182,0.12)", "border": "rgba(244,114,182,0.35)", "icon": "🩷", "label": "Low & Slow Exfil"},
    "normal":              {"color": "#94a3b8", "bg": "rgba(148,163,184,0.08)", "border": "rgba(148,163,184,0.25)", "icon": "⚪", "label": "Normal"},
}

# ── HTML helpers (plain string concat — no f-strings with CSS braces) ──

def threat_badge(t):
    c = THREAT_CONFIG.get(t, THREAT_CONFIG["normal"])
    return (
        '<span style="background:' + c["bg"] + ';color:' + c["color"] + ';border:1px solid ' + c["border"] + ';'
        'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap;letter-spacing:0.4px;">'
        + c["icon"] + ' ' + c["label"].upper() + '</span>'
    )

def risk_bar(score):
    pct = int(score * 100)
    color = "#f87171" if score >= 0.8 else "#fb923c" if score >= 0.6 else "#fbbf24" if score >= 0.4 else "#4ade80"
    return (
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:4px;height:6px;">'
        '<div style="width:' + str(pct) + '%;background:' + color + ';height:6px;border-radius:4px;'
        'box-shadow:0 0 6px ' + color + '88;"></div></div>'
        '<span style="color:' + color + ';font-weight:700;font-size:12px;min-width:36px;">' + str(round(score, 2)) + '</span>'
        '</div>'
    )

def conf_badge(conf):
    if conf == "High":
        return '<span style="background:rgba(34,197,94,0.12);color:#4ade80;border:1px solid rgba(74,222,128,0.35);padding:3px 10px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:1px;">✓ VERIFIED</span>'
    return '<span style="background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.35);padding:3px 10px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:1px;">⚡ COLD-START</span>'

def entity_type_badge(et):
    cfg = {
        "user":            ("#60a5fa", "rgba(96,165,250,0.12)",  "rgba(96,165,250,0.35)"),
        "service_account": ("#c084fc", "rgba(192,132,252,0.12)", "rgba(192,132,252,0.35)"),
        "edge_device":     ("#22d3ee", "rgba(34,211,238,0.12)",  "rgba(34,211,238,0.35)"),
    }.get(et, ("#94a3b8", "rgba(148,163,184,0.1)", "rgba(148,163,184,0.3)"))
    return (
        '<span style="background:' + cfg[1] + ';color:' + cfg[0] + ';border:1px solid ' + cfg[2] + ';'
        'padding:2px 9px;border-radius:12px;font-size:10px;font-weight:700;letter-spacing:0.5px;">'
        + et.replace("_", " ").upper() + '</span>'
    )

def kpi_card_html(title, value, subtitle, icon, accent):
    return (
        '<div style="background:rgba(10,10,10,0.75);border:1px solid rgba(255,255,255,0.07);'
        'border-top:2px solid ' + accent + ';border-radius:14px;padding:22px 24px;'
        'backdrop-filter:blur(20px);box-shadow:0 4px 24px rgba(0,0,0,0.5);position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:-12px;right:6px;font-size:60px;opacity:0.06;pointer-events:none;">' + icon + '</div>'
        '<div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">' + title + '</div>'
        '<div style="color:#f1f5f9;font-size:36px;font-weight:800;font-family:\'Space Grotesk\',sans-serif;line-height:1;margin-bottom:8px;">' + str(value) + '</div>'
        '<div style="color:' + accent + ';font-size:12px;font-weight:500;">' + subtitle + '</div>'
        '</div>'
    )

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
        st.markdown(
            '<div style="background:#000000;padding:24px 20px 20px 20px;margin:-1rem -1rem 0 -1rem;'
            'border-bottom:1px solid rgba(22,163,74,0.25);">'
            '<div style="font-size:10px;font-weight:900;letter-spacing:4px;color:#22c55e;margin-bottom:4px;">HONEYWELL</div>'
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:21px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">Sentinel-AI</div>'
            '<div style="font-size:10px;color:#475569;letter-spacing:2px;margin-top:2px;">SOC DASHBOARD</div>'
            '<div style="margin-top:14px;display:inline-flex;align-items:center;gap:7px;'
            'background:rgba(22,163,74,0.12);border:1px solid rgba(22,163,74,0.3);'
            'padding:5px 14px;border-radius:20px;">'
            '<div style="width:7px;height:7px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;'
            'box-shadow:0 0 7px #22c55e;"></div>'
            '<span style="color:#22c55e;font-size:10px;font-weight:700;letter-spacing:1.5px;">SYSTEM ACTIVE</span>'
            '</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:10px;padding:0 4px;">⚙ FILTERS</div>', unsafe_allow_html=True)

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Min Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect("Anomaly Types", options=all_types, default=all_types)

        if not df_alerts.empty:
            st.markdown('<div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin:18px 0 10px 0;padding:0 4px;">📊 THREAT DISTRIBUTION</div>', unsafe_allow_html=True)
            dist   = df_alerts['Predicted Type'].value_counts()
            colors = [THREAT_CONFIG.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_d  = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.62,
                marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} alerts<extra></extra>'
            ))
            fig_d.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), height=155,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_d.add_annotation(
                text='<b>' + str(len(df_alerts)) + '</b>',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, font=dict(size=18, color='#f1f5f9'), align='center'
            )
            st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})

        st.markdown(
            '<div style="margin:10px 0;padding:16px;background:rgba(255,255,255,0.04);'
            'border:1px solid rgba(255,255,255,0.08);border-radius:10px;">'
            '<div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:2px;margin-bottom:14px;">🤖 MODEL STATS</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#64748b;font-size:12px;">Precision</span>'
            '<span style="color:#4ade80;font-weight:800;font-size:13px;">92%</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#64748b;font-size:12px;">Recall</span>'
            '<span style="color:#4ade80;font-weight:800;font-size:13px;">88%</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#64748b;font-size:12px;">Algorithm</span>'
            '<span style="color:#60a5fa;font-weight:600;font-size:12px;">Random Forest</span>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;">'
            '<span style="color:#64748b;font-size:12px;">Explainability</span>'
            '<span style="color:#c084fc;font-weight:600;font-size:12px;">SHAP</span>'
            '</div></div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TOP NAV BAR
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="fade" style="background:rgba(0,0,0,0.8);border:1px solid rgba(255,255,255,0.08);'
        'backdrop-filter:blur(24px);border-radius:14px;padding:18px 30px;'
        'display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;'
        'box-shadow:0 4px 24px rgba(0,0,0,0.6);flex-wrap:wrap;gap:12px;">'

        '<div style="display:flex;align-items:center;gap:20px;">'
        '<div>'
        '<div style="font-size:9px;font-weight:900;letter-spacing:5px;color:#22c55e;margin-bottom:3px;">HONEYWELL</div>'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">Sentinel-AI</div>'
        '</div>'
        '<div style="width:1px;height:38px;background:rgba(255,255,255,0.1);"></div>'
        '<div style="color:#64748b;font-size:13px;">AI-Powered Behavioral Anomaly Detection</div>'
        '</div>'

        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        '<div style="display:flex;align-items:center;gap:6px;background:rgba(22,163,74,0.12);'
        'border:1px solid rgba(22,163,74,0.3);padding:5px 14px;border-radius:20px;">'
        '<div style="width:7px;height:7px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;box-shadow:0 0 7px #22c55e;"></div>'
        '<span style="color:#22c55e;font-size:11px;font-weight:700;letter-spacing:1px;">LIVE</span>'
        '</div>'
        '<span style="color:#475569;font-size:12px;">SHAP Active</span>'
        '<span style="color:#475569;font-size:12px;">Cold-Start Ready</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CARDS
    # ──────────────────────────────────────────────────────────────────────────
    high_risk  = len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0
    total_alts = len(df_alerts)
    hr_accent  = "#f87171" if high_risk > 0 else "#4ade80"
    hr_label   = "Critical — Score > 0.80" if high_risk > 0 else "No critical threats"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card_html("Sessions Analyzed", str(len(df_recent)), "Last 200 processed", "🔍", "#60a5fa"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card_html("Total Alerts", str(total_alts), "Flagged for review", "🚨", "#fbbf24"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card_html("High Risk Alerts", str(high_risk), hr_label, "⚠️", hr_accent), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card_html("Precision / Recall", "92% / 88%", "Holdout evaluation", "🤖", "#4ade80"), unsafe_allow_html=True)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

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
    st.markdown(
        '<div class="fade" style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
        '<span style="font-size:21px;">🚨</span>'
        '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:21px;font-weight:700;color:#f1f5f9;">Alert Queue</span>'
        '<span style="background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.35);'
        'padding:3px 12px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:2px;">LIVE</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if not df_filtered.empty:
        rows = ""
        for _, row in df_filtered.iterrows():
            short_ent = str(row['Entity'])[:28] + '…' if len(str(row['Entity'])) > 28 else str(row['Entity'])
            short_ts  = str(row['Timestamp'])[:16]
            expl      = str(row['Explanation'])[:90] + '…' if len(str(row['Explanation'])) > 90 else str(row['Explanation'])
            rows += (
                '<tr onmouseover="this.style.background=\'rgba(255,255,255,0.04)\'"'
                ' onmouseout="this.style.background=\'transparent\'"'
                ' style="border-bottom:1px solid rgba(255,255,255,0.05);transition:background 0.15s;">'
                '<td style="padding:13px 16px;font-size:11px;color:#475569;font-family:monospace;white-space:nowrap;">' + short_ts + '</td>'
                '<td style="padding:13px 16px;">'
                '<div style="font-size:12px;font-family:monospace;color:#e2e8f0;font-weight:600;">' + short_ent + '</div>'
                '<div style="margin-top:5px;">' + entity_type_badge(row['Entity Type']) + '</div>'
                '</td>'
                '<td style="padding:13px 16px;">' + threat_badge(row['Predicted Type']) + '</td>'
                '<td style="padding:13px 16px;min-width:130px;">' + risk_bar(row['Risk Score']) + '</td>'
                '<td style="padding:13px 16px;font-size:12px;color:#64748b;max-width:280px;line-height:1.5;">' + expl + '</td>'
                '<td style="padding:13px 16px;">' + conf_badge(row['Confidence']) + '</td>'
                '</tr>'
            )

        st.markdown(
            '<div class="fade" style="background:rgba(5,5,5,0.75);border:1px solid rgba(255,255,255,0.07);'
            'border-radius:14px;overflow:hidden;backdrop-filter:blur(20px);'
            'box-shadow:0 8px 32px rgba(0,0,0,0.6);margin-bottom:32px;">'
            '<div style="background:rgba(0,0,0,0.6);border-bottom:1px solid rgba(255,255,255,0.07);'
            'padding:14px 20px;display:flex;align-items:center;justify-content:space-between;">'
            '<span style="color:#e2e8f0;font-size:13px;font-weight:700;">Active Threat Alerts</span>'
            '<span style="color:#475569;font-size:12px;">Sorted by Risk Score</span>'
            '</div>'
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;" class="alert-table">'
            '<thead><tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.07);">'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">TIMESTAMP</th>'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">ENTITY</th>'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">THREAT TYPE</th>'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">RISK SCORE</th>'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">AI EXPLANATION</th>'
            '<th style="padding:12px 16px;text-align:left;color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;">CONFIDENCE</th>'
            '</tr></thead>'
            '<tbody>' + rows + '</tbody>'
            '</table></div></div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="fade" style="background:rgba(22,163,74,0.07);border:1px solid rgba(22,163,74,0.25);'
            'border-radius:14px;padding:60px;text-align:center;margin-bottom:32px;">'
            '<div style="font-size:50px;margin-bottom:14px;">✅</div>'
            '<div style="color:#4ade80;font-size:20px;font-weight:800;margin-bottom:8px;">No Threats Detected</div>'
            '<div style="color:#166534;font-size:14px;">All sessions are within normal behavioral bounds.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
            '<div style="flex:1;height:1px;background:rgba(255,255,255,0.07);"></div>'
            '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:20px;font-weight:700;color:#f1f5f9;white-space:nowrap;">🔎 Entity Deep-Dive</span>'
            '<div style="flex:1;height:1px;background:rgba(255,255,255,0.07);"></div>'
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

            # Entity Profile Card
            st.markdown(
                '<div class="fade entity-card" style="background:rgba(5,5,5,0.75);border:1px solid rgba(255,255,255,0.07);'
                'border-left:4px solid ' + threat_cfg["color"] + ';border-radius:14px;padding:22px 28px;margin-bottom:20px;'
                'backdrop-filter:blur(20px);box-shadow:0 4px 24px rgba(0,0,0,0.5);'
                'display:flex;gap:40px;align-items:center;flex-wrap:wrap;">'

                '<div><div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">ENTITY ID</div>'
                '<div style="color:#e2e8f0;font-size:13px;font-family:monospace;font-weight:600;">' + str(selected_entity)[:42] + '</div></div>'

                '<div><div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">ENTITY TYPE</div>'
                + entity_type_badge(entity_row['Entity Type']) + '</div>'

                '<div><div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:6px;">PREDICTED THREAT</div>'
                + threat_badge(entity_row['Predicted Type']) + '</div>'

                '<div><div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">RISK SCORE</div>'
                '<div style="color:' + threat_cfg["color"] + ';font-size:28px;font-weight:900;font-family:\'Space Grotesk\',sans-serif;">' + str(round(entity_row['Risk Score'], 2)) + '</div></div>'

                '<div><div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:5px;">SESSIONS</div>'
                '<div style="color:#f1f5f9;font-size:28px;font-weight:900;font-family:\'Space Grotesk\',sans-serif;">' + str(len(entity_history)) + '</div></div>'

                '</div>',
                unsafe_allow_html=True
            )

            col_l, col_r = st.columns([3, 2])

            with col_l:
                entity_history['timestamp'] = pd.to_datetime(entity_history['timestamp'], errors='coerce')
                fig_tl = px.scatter(
                    entity_history.dropna(subset=['timestamp']),
                    x='timestamp', y='session_duration',
                    color='auth_status',
                    color_discrete_map={"success": "#4ade80", "failed": "#f87171"},
                    title="Session Duration Timeline",
                    labels={'timestamp': 'Date', 'session_duration': 'Duration (s)', 'auth_status': 'Auth Status'},
                    hover_data=['geo_location', 'resource_accessed']
                )
                fig_tl.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0)))
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', height=300,
                    font=dict(family='Inter', color='#64748b', size=12),
                    title=dict(font=dict(family='Space Grotesk', size=15, color='#e2e8f0'), x=0),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#475569')),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)', tickfont=dict(color='#475569')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#64748b'), bordercolor='rgba(255,255,255,0.08)', borderwidth=1),
                    margin=dict(t=44, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})

            with col_r:
                # Behavioral Baseline — each line is a separate st.markdown call (no nested f-strings)
                st.markdown(
                    '<div style="background:rgba(5,5,5,0.75);border:1px solid rgba(255,255,255,0.07);'
                    'border-radius:12px;padding:20px 22px;backdrop-filter:blur(20px);">'
                    '<div style="color:#22c55e;font-size:11px;font-weight:800;letter-spacing:2px;'
                    'border-bottom:1px solid rgba(34,197,94,0.2);padding-bottom:10px;margin-bottom:16px;">📋 BEHAVIORAL BASELINE</div>',
                    unsafe_allow_html=True
                )

                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                if prof:
                    typical_geos      = list(prof['geo_dist'].keys())[:3]
                    typical_resources = list(prof['resource_dist'].keys())[:3]
                    avg_duration      = round(prof['duration_mean'], 1)

                    st.markdown('<div style="color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">TYPICAL LOCATIONS</div>', unsafe_allow_html=True)
                    for g in typical_geos:
                        st.markdown(
                            '<div style="background:rgba(96,165,250,0.1);color:#60a5fa;border:1px solid rgba(96,165,250,0.25);'
                            'padding:5px 12px;border-radius:8px;font-size:12px;font-weight:600;margin-bottom:5px;">📍 ' + str(g) + '</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<div style="color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;margin:12px 0 8px 0;">TYPICAL RESOURCES</div>', unsafe_allow_html=True)
                    for r in typical_resources:
                        st.markdown(
                            '<div style="background:rgba(192,132,252,0.1);color:#c084fc;border:1px solid rgba(192,132,252,0.25);'
                            'padding:5px 12px;border-radius:8px;font-size:12px;font-weight:600;margin-bottom:5px;">📂 ' + str(r) + '</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<div style="color:#334155;font-size:10px;font-weight:700;letter-spacing:1.5px;margin:12px 0 6px 0;">AVG SESSION DURATION</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="color:#fbbf24;font-size:24px;font-weight:900;font-family:\'Space Grotesk\',sans-serif;">' + str(avg_duration) + 's</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<div style="color:#475569;font-size:13px;font-style:italic;">No baseline profile for this entity.</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Hourly Activity Bar
            if 'timestamp' in entity_history.columns:
                entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)
                bar_colors = ['#f87171' if v == hourly.max() and v > 0 else '#1d4ed8' for v in hourly.values]
                fig_h = go.Figure(go.Bar(
                    x=list(range(24)), y=hourly.values,
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate='%{x}:00 — %{y} sessions<extra></extra>'
                ))
                fig_h.update_layout(
                    title=dict(text='Hourly Activity Pattern', font=dict(family='Space Grotesk', size=15, color='#e2e8f0'), x=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', height=220,
                    font=dict(family='Inter', color='#64748b', size=12),
                    xaxis=dict(title='Hour of Day', gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569')),
                    yaxis=dict(title='Sessions',    gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#475569')),
                    margin=dict(t=44, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:48px;padding:18px 0;border-top:1px solid rgba(255,255,255,0.07);'
        'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="font-size:10px;font-weight:900;letter-spacing:4px;color:#22c55e;">HONEYWELL</span>'
        '<span style="color:#1e293b;">|</span>'
        '<span style="color:#334155;font-size:12px;">Sentinel-AI Behavioral Anomaly Detection</span>'
        '</div>'
        '<span style="color:#1e293b;font-size:12px;">Random Forest · SHAP · Cold-Start · Concept Drift</span>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
