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
    page_title="Sentinel-AI | Honeywell SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# Deep Navy (#0a192f) / Charcoal (#18181b) | Burnt Orange (#ea580c) accent
# JetBrains Mono for labels | Inter -0.02em for headings | Sharp 2px corners
# ──────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── HIDE Streamlit chrome (Deploy button, menu, footer) ── */
#MainMenu          { visibility: hidden !important; }
header             { visibility: hidden !important; }
footer             { visibility: hidden !important; }
.stDeployButton    { display: none !important; }

/* ── CSS Variables ── */
:root {
    --bg-base:        #0a192f;
    --bg-raised:      #112240;
    --bg-overlay:     #0d1f3c;
    --border:         #1e3a5f;
    --border-subtle:  #172a45;
    --accent:         #ea580c;
    --accent-dim:     rgba(234,88,12,0.15);
    --accent-border:  rgba(234,88,12,0.4);
    --text-h:         #ffffff;
    --text-body:      #cbd5e1;
    --text-muted:     #64748b;
    --text-dim:       #334155;
    --green:          #22c55e;
    --red:            #dc2626;
    --amber:          #d97706;
}

/* ── Base & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-body) !important;
}

/* Grid-line technical background — no blobs, no gradients */
.stApp {
    background-color: var(--bg-base) !important;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px) !important;
    background-size: 36px 36px !important;
}

/* ── Sidebar — charcoal, left accent rule ── */
section[data-testid="stSidebar"] {
    background: #12121a !important;
    border-right: 2px solid var(--accent) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-body) !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }

/* ── Slider — orange track ── */
.stSlider [data-baseweb="slider"] [role="progressbar"] { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    border-radius: 2px !important;
}
.stSlider label, .stSlider p { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* ── Multiselect ── */
div[data-baseweb="tag"] {
    background: var(--accent-dim) !important;
    border: 1px solid var(--accent-border) !important;
    border-radius: 2px !important;
}
div[data-baseweb="tag"] span { color: #fdba74 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }
div[data-baseweb="select"] > div {
    background: var(--bg-overlay) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text-body) !important;
}
.stMultiSelect label { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 1.5px !important; }

/* ── Main selectbox ── */
.stSelectbox label { color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 1.5px !important; }
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-overlay) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text-body) !important;
}

/* ── Block container padding ── */
.block-container { padding-top: 1.5rem !important; }

/* ── Scrollbar — thin, orange thumb ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 0px; }

/* ── Plotly chart container ── */
.stPlotlyChart { border-radius: 0px; overflow: hidden; }

/* ── Animations ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: translateX(0); }
}
.slide-in { animation: slideIn 0.35s ease both; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .alert-tbl { font-size: 10px !important; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT TAXONOMY — Sharp, flat badges; specific non-neon colors
# ──────────────────────────────────────────────────────────────────────────────
THREATS = {
    "brute_force":         {"color": "#dc2626", "bg": "rgba(220,38,38,0.12)",   "border": "rgba(220,38,38,0.4)",   "icon": "●", "label": "Brute Force"},
    "impossible_travel":   {"color": "#ea580c", "bg": "rgba(234,88,12,0.12)",   "border": "rgba(234,88,12,0.4)",   "icon": "●", "label": "Impossible Travel"},
    "device_spoofing":     {"color": "#d97706", "bg": "rgba(217,119,6,0.12)",   "border": "rgba(217,119,6,0.4)",   "icon": "●", "label": "Device Spoofing"},
    "lateral_movement":    {"color": "#7c3aed", "bg": "rgba(124,58,237,0.12)",  "border": "rgba(124,58,237,0.4)",  "icon": "●", "label": "Lateral Movement"},
    "credential_stuffing": {"color": "#1d4ed8", "bg": "rgba(29,78,216,0.12)",   "border": "rgba(29,78,216,0.4)",   "icon": "●", "label": "Credential Stuffing"},
    "insider_drift":       {"color": "#0891b2", "bg": "rgba(8,145,178,0.12)",   "border": "rgba(8,145,178,0.4)",   "icon": "●", "label": "Insider Drift"},
    "low_and_slow_exfil":  {"color": "#be185d", "bg": "rgba(190,24,93,0.12)",   "border": "rgba(190,24,93,0.4)",   "icon": "●", "label": "Low & Slow Exfil"},
    "normal":              {"color": "#475569", "bg": "rgba(71,85,105,0.1)",    "border": "rgba(71,85,105,0.3)",   "icon": "○", "label": "Normal"},
}

# ── HTML helpers — plain string concat, no f-string CSS braces ──

def threat_badge(t):
    c = THREATS.get(t, THREATS["normal"])
    return (
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        'background:' + c["bg"] + ';color:' + c["color"] + ';'
        'border:1px solid ' + c["border"] + ';'
        'padding:3px 9px;border-radius:2px;'
        'font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:600;white-space:nowrap;letter-spacing:0.5px;">'
        + c["icon"] + ' ' + c["label"].upper() + '</span>'
    )

def risk_bar(score):
    pct   = int(score * 100)
    color = "#dc2626" if score >= 0.8 else "#ea580c" if score >= 0.6 else "#d97706" if score >= 0.4 else "#22c55e"
    return (
        '<div style="display:flex;align-items:center;gap:8px;">'
        '<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:0px;height:5px;">'
        '<div style="width:' + str(pct) + '%;background:' + color + ';height:5px;border-radius:0px;"></div>'
        '</div>'
        '<span style="color:' + color + ';font-weight:700;font-size:11px;min-width:34px;'
        'font-family:\'JetBrains Mono\',monospace;">' + str(round(score, 2)) + '</span>'
        '</div>'
    )

def conf_badge(conf):
    if conf == "High":
        return (
            '<span style="background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.35);'
            'padding:3px 9px;border-radius:2px;font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:600;letter-spacing:1px;">VERIFIED</span>'
        )
    return (
        '<span style="background:rgba(217,119,6,0.1);color:#d97706;border:1px solid rgba(217,119,6,0.35);'
        'padding:3px 9px;border-radius:2px;font-family:\'JetBrains Mono\',monospace;font-size:10px;font-weight:600;letter-spacing:1px;">COLD-START</span>'
    )

def et_badge(et):
    cfg = {
        "user":            ("#60a5fa", "rgba(96,165,250,0.1)",  "rgba(96,165,250,0.3)"),
        "service_account": ("#a78bfa", "rgba(167,139,250,0.1)", "rgba(167,139,250,0.3)"),
        "edge_device":     ("#22d3ee", "rgba(34,211,238,0.1)",  "rgba(34,211,238,0.3)"),
    }.get(et, ("#64748b", "rgba(100,116,139,0.1)", "rgba(100,116,139,0.3)"))
    return (
        '<span style="background:' + cfg[1] + ';color:' + cfg[0] + ';border:1px solid ' + cfg[2] + ';'
        'padding:2px 7px;border-radius:2px;font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:600;letter-spacing:0.8px;">'
        + et.replace("_", "_").upper() + '</span>'
    )

def kpi_card(title, value, subtitle, accent_color):
    return (
        '<div style="background:#112240;border:1px solid #1e3a5f;border-top:2px solid ' + accent_color + ';'
        'border-radius:2px;padding:20px 22px;position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;right:0;width:3px;height:100%;background:' + accent_color + ';opacity:0.3;"></div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;color:#64748b;font-size:9px;font-weight:600;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">' + title + '</div>'
        '<div style="color:#ffffff;font-size:32px;font-weight:800;font-family:\'Inter\',sans-serif;'
        'letter-spacing:-0.02em;line-height:1;margin-bottom:8px;">' + str(value) + '</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;color:' + accent_color + ';font-size:11px;">' + subtitle + '</div>'
        '</div>'
    )

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data_and_models():
    try:
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    alerts    = []
    explainer = AlertExplainer(classifier.model, classifier.features)

    for _, row in df_recent.iterrows():
        score, is_high_conf = baseline.score_session(row)
        if score > 0.4:
            X_feats = classifier._engineer_features(pd.DataFrame([row]))
            missing = [c for c in classifier.features if c not in X_feats.columns]
            if missing:
                X_feats = pd.concat([X_feats, pd.DataFrame(0, index=X_feats.index, columns=missing)], axis=1)
            X_feats   = X_feats[classifier.features].astype(float)
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
    # SIDEBAR — charcoal #12121a, orange left border
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Brand block
        st.markdown(
            '<div style="background:#0d0d14;padding:24px 20px 20px 20px;margin:-1rem -1rem 0 -1rem;'
            'border-bottom:1px solid #1e3a5f;">'

            # HONEYWELL label
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;'
            'letter-spacing:4px;color:#ea580c;margin-bottom:6px;">HONEYWELL</div>'

            # Product name
            '<div style="font-family:\'Inter\',sans-serif;font-size:22px;font-weight:800;'
            'color:#ffffff;letter-spacing:-0.02em;line-height:1.1;">Sentinel<span style="color:#ea580c;">-AI</span></div>'

            # Tag
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#334155;'
            'letter-spacing:2px;margin-top:4px;">SOC // THREAT DETECTION</div>'

            # Status pill
            '<div style="margin-top:16px;display:inline-flex;align-items:center;gap:8px;'
            'border:1px solid rgba(34,197,94,0.3);padding:5px 12px;border-radius:2px;'
            'background:rgba(34,197,94,0.06);">'
            '<div style="width:6px;height:6px;background:#22c55e;border-radius:0px;'
            'animation:blink 2s infinite;"></div>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#22c55e;'
            'font-size:9px;font-weight:700;letter-spacing:2px;">SYSTEM_ACTIVE</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # Section label
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
            'font-weight:700;letter-spacing:2.5px;margin-bottom:12px;padding:0 4px;">// FILTERS</div>',
            unsafe_allow_html=True
        )

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Min Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect("Anomaly Types", options=all_types, default=all_types)

        # Threat distribution donut
        if not df_alerts.empty:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'font-weight:700;letter-spacing:2.5px;margin:20px 0 10px 0;padding:0 4px;">// THREAT_MIX</div>',
                unsafe_allow_html=True
            )
            dist   = df_alerts['Predicted Type'].value_counts()
            colors = [THREATS.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_d  = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.60,
                marker=dict(colors=colors, line=dict(color='#12121a', width=2)),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} events<extra></extra>'
            ))
            fig_d.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), height=148,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_d.add_annotation(
                text='<b>' + str(len(df_alerts)) + '</b>',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, font=dict(size=17, color='#ffffff', family='JetBrains Mono'), align='center'
            )
            st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})

        # Model stats block
        st.markdown(
            '<div style="margin:8px 0;padding:14px 16px;background:#0d1f3c;'
            'border:1px solid #1e3a5f;border-left:2px solid #ea580c;border-radius:2px;">'
            '<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
            'font-weight:700;letter-spacing:2.5px;margin-bottom:14px;">// MODEL_STATS</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#64748b;font-size:11px;">precision</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#22c55e;font-weight:700;font-size:12px;">0.92</span>'
            '</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#64748b;font-size:11px;">recall</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#22c55e;font-weight:700;font-size:12px;">0.88</span>'
            '</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#64748b;font-size:11px;">model</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#60a5fa;font-size:11px;">RandomForest</span>'
            '</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#64748b;font-size:11px;">xai</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#a78bfa;font-size:11px;">SHAP</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TOP HEADER BAR — left-aligned, asymmetric, technical
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slide-in" style="background:#0d1f3c;border:1px solid #1e3a5f;'
        'border-left:3px solid #ea580c;border-radius:2px;padding:18px 28px;'
        'display:flex;align-items:center;justify-content:space-between;'
        'margin-bottom:20px;flex-wrap:wrap;gap:14px;">'

        # Left block
        '<div style="display:flex;align-items:center;gap:24px;">'
        '<div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:700;'
        'letter-spacing:5px;color:#ea580c;margin-bottom:5px;">HONEYWELL</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:24px;font-weight:800;'
        'color:#ffffff;letter-spacing:-0.02em;">Sentinel<span style="color:#ea580c;">-AI</span></div>'
        '</div>'
        '<div style="width:1px;height:40px;background:#1e3a5f;"></div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:11px;">'
        'Behavioral Anomaly Detection System</div>'
        '</div>'

        # Right block — status tags
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        '<div style="display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(34,197,94,0.35);'
        'background:rgba(34,197,94,0.06);padding:6px 14px;border-radius:2px;">'
        '<div style="width:6px;height:6px;background:#22c55e;border-radius:0px;animation:blink 2s infinite;"></div>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#22c55e;font-size:9px;'
        'font-weight:700;letter-spacing:2px;">LIVE</span>'
        '</div>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:10px;">SHAP_ACTIVE</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:10px;">ZERO_TRUST</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # KPI CARDS — sharp, left accent rule, monospace labels
    # ──────────────────────────────────────────────────────────────────────────
    high_risk  = len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0
    total_alts = len(df_alerts)
    hr_color   = "#dc2626" if high_risk > 0 else "#22c55e"
    hr_sub     = str(high_risk) + " critical events" if high_risk > 0 else "within threshold"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("sessions_analyzed", str(len(df_recent)), "last 200 events", "#60a5fa"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("total_alerts", str(total_alts), "flagged for review", "#ea580c"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("high_risk", str(high_risk), hr_sub, hr_color), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("precision / recall", "92% / 88%", "holdout evaluation", "#22c55e"), unsafe_allow_html=True)

    st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)

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
    alert_count = str(len(df_filtered)) if not df_filtered.empty else "0"
    st.markdown(
        '<div class="slide-in" style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        '<span style="font-family:\'Inter\',sans-serif;font-size:18px;font-weight:800;'
        'color:#ffffff;letter-spacing:-0.02em;">Alert Queue</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;background:rgba(220,38,38,0.12);'
        'color:#dc2626;border:1px solid rgba(220,38,38,0.4);'
        'padding:3px 10px;border-radius:2px;font-size:9px;font-weight:700;letter-spacing:2px;">LIVE</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:10px;">'
        + alert_count + ' events</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if not df_filtered.empty:
        rows = ""
        for _, row in df_filtered.iterrows():
            short_ent = str(row['Entity'])[:28] + '…' if len(str(row['Entity'])) > 28 else str(row['Entity'])
            short_ts  = str(row['Timestamp'])[:16]
            expl      = str(row['Explanation'])[:88] + '…' if len(str(row['Explanation'])) > 88 else str(row['Explanation'])

            rows += (
                '<tr onmouseover="this.style.background=\'rgba(234,88,12,0.04)\'"'
                ' onmouseout="this.style.background=\'transparent\'"'
                ' style="border-bottom:1px solid #0f2944;transition:background 0.12s;">'
                '<td style="padding:12px 16px;font-family:\'JetBrains Mono\',monospace;'
                'font-size:11px;color:#334155;white-space:nowrap;">' + short_ts + '</td>'
                '<td style="padding:12px 16px;">'
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#e2e8f0;font-weight:600;">' + short_ent + '</div>'
                '<div style="margin-top:5px;">' + et_badge(row['Entity Type']) + '</div>'
                '</td>'
                '<td style="padding:12px 16px;">' + threat_badge(row['Predicted Type']) + '</td>'
                '<td style="padding:12px 16px;min-width:130px;">' + risk_bar(row['Risk Score']) + '</td>'
                '<td style="padding:12px 16px;font-size:12px;color:#64748b;max-width:280px;line-height:1.55;">' + expl + '</td>'
                '<td style="padding:12px 16px;">' + conf_badge(row['Confidence']) + '</td>'
                '</tr>'
            )

        st.markdown(
            '<div class="slide-in" style="background:#0d1f3c;border:1px solid #1e3a5f;'
            'border-radius:2px;overflow:hidden;margin-bottom:32px;">'

            # Table header bar
            '<div style="background:#112240;border-bottom:1px solid #1e3a5f;'
            'padding:12px 20px;display:flex;align-items:center;justify-content:space-between;">'
            '<span style="font-family:\'Inter\',sans-serif;color:#ffffff;font-size:13px;font-weight:700;letter-spacing:-0.01em;">Active Threats</span>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:10px;">sorted_by: risk_score DESC</span>'
            '</div>'

            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;" class="alert-tbl">'
            '<thead><tr style="background:#0a192f;border-bottom:1px solid #1e3a5f;">'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">TIMESTAMP</th>'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">ENTITY</th>'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">THREAT_TYPE</th>'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">RISK_SCORE</th>'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">AI_EXPLANATION</th>'
            '<th style="padding:10px 16px;text-align:left;font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;font-weight:700;letter-spacing:1.5px;">CONFIDENCE</th>'
            '</tr></thead>'
            '<tbody>' + rows + '</tbody>'
            '</table></div></div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="slide-in" style="background:#0d1f3c;border:1px solid rgba(34,197,94,0.3);'
            'border-left:3px solid #22c55e;border-radius:2px;padding:52px;text-align:left;margin-bottom:32px;">'
            '<div style="font-family:\'JetBrains Mono\',monospace;color:#22c55e;font-size:11px;'
            'letter-spacing:2px;margin-bottom:8px;">// STATUS</div>'
            '<div style="font-family:\'Inter\',sans-serif;color:#ffffff;font-size:18px;font-weight:700;'
            'letter-spacing:-0.02em;margin-bottom:6px;">No Threats Detected</div>'
            '<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:11px;">'
            'All sessions within normal behavioral bounds for active filter set.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        # Section label with ruled line
        st.markdown(
            '<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
            '<span style="font-family:\'Inter\',sans-serif;font-size:18px;font-weight:800;'
            'color:#ffffff;letter-spacing:-0.02em;white-space:nowrap;">Entity Deep-Dive</span>'
            '<div style="flex:1;height:1px;background:#1e3a5f;"></div>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;letter-spacing:2px;">ZERO_TRUST_VIEW</span>'
            '</div>',
            unsafe_allow_html=True
        )

        selected_entity = st.selectbox(
            "SELECT ENTITY",
            options=df_filtered['Entity'].unique(),
            key="entity_select"
        )

        if selected_entity:
            entity_history = df[df['entity_id'] == selected_entity].copy()
            entity_row     = df_filtered[df_filtered['Entity'] == selected_entity].iloc[0]
            threat_cfg     = THREATS.get(entity_row['Predicted Type'], THREATS["normal"])

            # Entity profile card — left-aligned, sharp, left border accent
            st.markdown(
                '<div class="slide-in" style="background:#0d1f3c;border:1px solid #1e3a5f;'
                'border-left:3px solid ' + threat_cfg["color"] + ';border-radius:2px;'
                'padding:20px 26px;margin-bottom:18px;'
                'display:flex;gap:40px;align-items:flex-start;flex-wrap:wrap;">'

                '<div><div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'letter-spacing:1.5px;margin-bottom:5px;">entity_id</div>'
                '<div style="font-family:\'JetBrains Mono\',monospace;color:#e2e8f0;font-size:12px;font-weight:600;">'
                + str(selected_entity)[:40] + '</div></div>'

                '<div><div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'letter-spacing:1.5px;margin-bottom:6px;">entity_type</div>'
                + et_badge(entity_row['Entity Type']) + '</div>'

                '<div><div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'letter-spacing:1.5px;margin-bottom:6px;">predicted_threat</div>'
                + threat_badge(entity_row['Predicted Type']) + '</div>'

                '<div><div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'letter-spacing:1.5px;margin-bottom:5px;">risk_score</div>'
                '<div style="font-family:\'JetBrains Mono\',monospace;color:' + threat_cfg["color"] + ';'
                'font-size:26px;font-weight:700;">' + str(round(entity_row['Risk Score'], 2)) + '</div></div>'

                '<div><div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;'
                'letter-spacing:1.5px;margin-bottom:5px;">session_count</div>'
                '<div style="font-family:\'JetBrains Mono\',monospace;color:#e2e8f0;'
                'font-size:26px;font-weight:700;">' + str(len(entity_history)) + '</div></div>'
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
                    color_discrete_map={"success": "#22c55e", "failed": "#dc2626"},
                    title="session_duration_timeline",
                    labels={'timestamp': '', 'session_duration': 'duration (s)', 'auth_status': ''},
                    hover_data=['geo_location', 'resource_accessed']
                )
                fig_tl.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0a192f', height=290,
                    font=dict(family='JetBrains Mono', color='#334155', size=11),
                    title=dict(font=dict(family='JetBrains Mono', size=12, color='#64748b'), x=0),
                    xaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f', tickfont=dict(color='#334155'), showgrid=True),
                    yaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f', tickfont=dict(color='#334155')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#64748b', family='JetBrains Mono')),
                    margin=dict(t=36, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})

            with col_r:
                # Behavioral Baseline — individual markdown calls, zero nesting issues
                st.markdown(
                    '<div style="background:#0d1f3c;border:1px solid #1e3a5f;'
                    'border-left:2px solid #ea580c;border-radius:2px;padding:18px 20px;">'
                    '<div style="font-family:\'JetBrains Mono\',monospace;color:#ea580c;font-size:9px;'
                    'font-weight:700;letter-spacing:2.5px;border-bottom:1px solid #1e3a5f;'
                    'padding-bottom:10px;margin-bottom:16px;">// BEHAVIORAL_BASELINE</div>',
                    unsafe_allow_html=True
                )

                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                if prof:
                    typical_geos      = list(prof['geo_dist'].keys())[:3]
                    typical_resources = list(prof['resource_dist'].keys())[:3]
                    avg_duration      = round(prof['duration_mean'], 1)

                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;letter-spacing:1.5px;margin-bottom:8px;">geo_locations[]</div>', unsafe_allow_html=True)
                    for g in typical_geos:
                        st.markdown(
                            '<div style="background:rgba(29,78,216,0.08);color:#60a5fa;border:1px solid rgba(29,78,216,0.3);'
                            'padding:5px 11px;border-radius:2px;font-family:\'JetBrains Mono\',monospace;'
                            'font-size:11px;margin-bottom:4px;">' + str(g) + '</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;letter-spacing:1.5px;margin:12px 0 8px 0;">resources[]</div>', unsafe_allow_html=True)
                    for r in typical_resources:
                        st.markdown(
                            '<div style="background:rgba(124,58,237,0.08);color:#a78bfa;border:1px solid rgba(124,58,237,0.3);'
                            'padding:5px 11px;border-radius:2px;font-family:\'JetBrains Mono\',monospace;'
                            'font-size:11px;margin-bottom:4px;">' + str(r) + '</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:9px;letter-spacing:1.5px;margin:12px 0 5px 0;">avg_session_duration</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-family:\'JetBrains Mono\',monospace;color:#d97706;font-size:22px;font-weight:700;">' + str(avg_duration) + 's</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:11px;">// no_baseline_profile_found</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Hourly Activity
            if 'timestamp' in entity_history.columns:
                entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)
                max_v  = hourly.max()
                bar_colors = ['#ea580c' if v == max_v and v > 0 else '#1d4ed8' for v in hourly.values]

                fig_h = go.Figure(go.Bar(
                    x=list(range(24)), y=hourly.values,
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    hovertemplate='%{x}:00 → %{y} sessions<extra></extra>'
                ))
                fig_h.update_layout(
                    title=dict(text='hourly_activity_pattern', font=dict(family='JetBrains Mono', size=12, color='#64748b'), x=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0a192f', height=210,
                    font=dict(family='JetBrains Mono', color='#334155', size=11),
                    xaxis=dict(title='', gridcolor='#1e3a5f', tickfont=dict(color='#334155')),
                    yaxis=dict(title='', gridcolor='#1e3a5f', tickfont=dict(color='#334155')),
                    margin=dict(t=36, b=0, l=0, r=0),
                    bargap=0.25,
                )
                st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER — left-aligned, monospace, minimal
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:48px;padding:16px 0;border-top:1px solid #1e3a5f;'
        'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;font-weight:900;'
        'letter-spacing:5px;color:#ea580c;">HONEYWELL</span>'
        '<span style="color:#1e3a5f;">|</span>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#334155;font-size:10px;">'
        'sentinel-ai // behavioral-anomaly-detection</span>'
        '</div>'
        '<span style="font-family:\'JetBrains Mono\',monospace;color:#1e3a5f;font-size:10px;">'
        'RF · SHAP · cold-start · concept-drift</span>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
