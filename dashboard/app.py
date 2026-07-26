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
# Enterprise Software / Cyber Security Company Theme
# Minimalist, clean, functional, high-trust.
# ──────────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── HIDE Streamlit chrome ── */
#MainMenu          { visibility: hidden !important; }
header             { visibility: hidden !important; }
footer             { visibility: hidden !important; }
.stDeployButton    { display: none !important; }

/* ── CSS Variables ── */
:root {
    --bg-base:        #0F172A;
    --bg-surface:     #111827;
    --border:         #1F2937;
    --primary:        #2563EB;
    --success:        #10B981;
    --warning:        #F59E0B;
    --danger:         #EF4444;
    --text-main:      #F9FAFB;
    --text-muted:     #94A3B8;
}

/* ── Base & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-main) !important;
}

.stApp {
    background-color: var(--bg-base) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="progressbar"] { background: var(--primary) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #ffffff !important;
    border: 2px solid var(--primary) !important;
    border-radius: 50% !important;
    width: 14px !important;
    height: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stSlider label, .stSlider p { color: var(--text-muted) !important; font-size: 13px !important; font-weight: 500 !important; }

/* ── Multiselect ── */
div[data-baseweb="tag"] {
    background-color: #1E293B !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
div[data-baseweb="tag"] span { color: var(--text-main) !important; font-size: 12px !important; }
div[data-baseweb="select"] > div {
    background-color: var(--bg-base) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-main) !important;
}
.stMultiSelect label { color: var(--text-muted) !important; font-size: 13px !important; font-weight: 500 !important; margin-bottom: 4px !important; }

/* ── Main selectbox ── */
.stSelectbox label { color: var(--text-muted) !important; font-size: 13px !important; font-weight: 500 !important; margin-bottom: 4px !important; }
.stSelectbox [data-baseweb="select"] > div {
    background-color: var(--bg-base) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-main) !important;
}

/* ── Block container padding ── */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── Plotly chart container ── */
.stPlotlyChart { border-radius: 6px; overflow: hidden; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .alert-tbl { font-size: 11px !important; }
}

/* ── Typography and utility classes ── */
.mono { font-family: 'JetBrains Mono', monospace; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREAT TAXONOMY — Professional, understated colors
# ──────────────────────────────────────────────────────────────────────────────
THREATS = {
    "brute_force":         {"color": "#EF4444", "bg": "rgba(239, 68, 68, 0.1)",   "border": "rgba(239, 68, 68, 0.2)",   "label": "Brute Force"},
    "impossible_travel":   {"color": "#F97316", "bg": "rgba(249, 115, 22, 0.1)",  "border": "rgba(249, 115, 22, 0.2)",  "label": "Impossible Travel"},
    "device_spoofing":     {"color": "#EAB308", "bg": "rgba(234, 179, 8, 0.1)",   "border": "rgba(234, 179, 8, 0.2)",   "label": "Device Spoofing"},
    "lateral_movement":    {"color": "#8B5CF6", "bg": "rgba(139, 92, 246, 0.1)",  "border": "rgba(139, 92, 246, 0.2)",  "label": "Lateral Movement"},
    "credential_stuffing": {"color": "#3B82F6", "bg": "rgba(59, 130, 246, 0.1)",  "border": "rgba(59, 130, 246, 0.2)",  "label": "Credential Stuffing"},
    "insider_drift":       {"color": "#14B8A6", "bg": "rgba(20, 184, 166, 0.1)",  "border": "rgba(20, 184, 166, 0.2)",  "label": "Insider Drift"},
    "low_and_slow_exfil":  {"color": "#F43F5E", "bg": "rgba(244, 63, 94, 0.1)",   "border": "rgba(244, 63, 94, 0.2)",   "label": "Low & Slow Exfil"},
    "normal":              {"color": "#94A3B8", "bg": "rgba(148, 163, 184, 0.1)", "border": "rgba(148, 163, 184, 0.2)", "label": "Unclassified Borderline"},
}

# ── HTML helpers — clean, minimal markup ──

def threat_badge(t):
    c = THREATS.get(t, THREATS["normal"])
    return (
        '<span style="display:inline-flex;align-items:center;background:' + c["bg"] + ';color:' + c["color"] + ';'
        'border:1px solid ' + c["border"] + ';padding:2px 8px;border-radius:4px;'
        'font-size:12px;font-weight:500;">'
        + c["label"] + '</span>'
    )

def risk_bar(score):
    pct   = int(score * 100)
    color = "#EF4444" if score >= 0.8 else "#F59E0B" if score >= 0.6 else "#FBBF24" if score >= 0.4 else "#10B981"
    return (
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<div style="flex:1;background:#334155;border-radius:2px;height:4px;overflow:hidden;">'
        '<div style="width:' + str(pct) + '%;background:' + color + ';height:100%;"></div>'
        '</div>'
        '<span class="mono" style="color:' + color + ';font-weight:500;font-size:12px;width:32px;text-align:right;">' + f"{score:.2f}" + '</span>'
        '</div>'
    )

def conf_badge(conf):
    if conf == "High":
        return (
            '<span style="display:inline-flex;align-items:center;gap:4px;color:#10B981;font-size:12px;font-weight:500;">'
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
            'Verified</span>'
        )
    return (
        '<span style="display:inline-flex;align-items:center;gap:4px;color:#F59E0B;font-size:12px;font-weight:500;">'
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
        'Cold-Start</span>'
    )

def et_badge(et):
    return (
        '<span style="color:#94A3B8;font-size:12px;font-weight:500;">'
        + et.replace("_", " ").title() + '</span>'
    )

def kpi_card(title, value, subtitle):
    return (
        '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:20px;'
        'box-shadow:0 1px 2px rgba(0,0,0,0.1);display:flex;flex-direction:column;gap:8px;">'
        '<div style="color:#94A3B8;font-size:13px;font-weight:500;">' + title + '</div>'
        '<div style="color:#F9FAFB;font-size:28px;font-weight:600;line-height:1.2;">' + str(value) + '</div>'
        '<div style="color:#64748B;font-size:12px;">' + subtitle + '</div>'
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
@st.cache_data(show_spinner=False)
def process_recent_alerts(df_recent, _baseline, _classifier):
    alerts = []
    for _, row in df_recent.iterrows():
        score, is_high_conf = _baseline.score_session(row)
        if score > 0.4:
            X_feats = _classifier._engineer_features(pd.DataFrame([row]))
            missing = [c for c in _classifier.features if c not in X_feats.columns]
            if missing:
                X_feats = pd.concat([X_feats, pd.DataFrame(0, index=X_feats.index, columns=missing)], axis=1)
            X_feats   = X_feats[_classifier.features].astype(float)
            pred_type = _classifier.model.predict(X_feats)[0]
            alerts.append({
                'Timestamp':      row['timestamp'],
                'Entity':         row['entity_id'],
                'Entity Type':    row['entity_type'],
                'Risk Score':     score,
                'Predicted Type': pred_type,
                'Confidence':     'High' if is_high_conf else 'Low (Cold-Start)',
                'RowData':        row
            })
    return pd.DataFrame(alerts)

def main():
    df, baseline, classifier = load_data_and_models()
    if df is None:
        return

    # ── Process alerts ──
    df_recent = df.tail(200).copy()
    df_alerts = process_recent_alerts(df_recent, baseline, classifier)

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Brand block
        st.markdown(
            '<div style="padding:24px 20px 24px 20px;border-bottom:1px solid #1F2937;">'
            '<div style="font-size:24px;font-weight:800;color:#F9FAFB;letter-spacing:1px;margin-bottom:2px;">Sentinel-AI</div>'
            '<div style="font-size:14px;font-weight:500;color:#94A3B8;line-height:1.2;">Built for Honeywell Campus Connect</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="color:#F9FAFB;font-size:14px;font-weight:600;margin-bottom:16px;padding:0 4px;">Filters</div>',
            unsafe_allow_html=True
        )

        risk_threshold = 0.4
        anomaly_types_selected = []

        if not df_alerts.empty:
            risk_threshold = st.slider("Minimum Risk Score", 0.0, 1.0, 0.4, step=0.05)
            all_types = sorted(df_alerts['Predicted Type'].unique())
            anomaly_types_selected = st.multiselect("Anomaly Types", options=all_types, default=all_types)

        # Threat distribution donut
        if not df_alerts.empty:
            st.markdown(
                '<div style="color:#F9FAFB;font-size:14px;font-weight:600;margin:32px 0 16px 0;padding:0 4px;">Threat Distribution</div>',
                unsafe_allow_html=True
            )
            dist   = df_alerts['Predicted Type'].value_counts()
            colors = [THREATS.get(t, {"color": "#64748b"})["color"] for t in dist.index]
            fig_d  = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.7,
                marker=dict(colors=colors, line=dict(color='#111827', width=2)),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>%{value} events<extra></extra>'
            ))
            fig_d.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), height=180,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
            )
            fig_d.add_annotation(
                text='<span style="font-size:24px;font-weight:600;color:#F9FAFB;">' + str(len(df_alerts)) + '</span><br><span style="font-size:12px;color:#94A3B8;">Alerts</span>',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, align='center'
            )
            st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})

        # Model stats block
        st.markdown(
            '<div style="margin-top:24px;padding:20px;background-color:#1E293B;border-radius:8px;">'
            '<div style="color:#F9FAFB;font-size:13px;font-weight:600;margin-bottom:16px;">Detection Telemetry</div>'
            
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#94A3B8;font-size:13px;">Precision</span>'
            '<span style="color:#F9FAFB;font-weight:500;font-size:13px;">24.1%</span>'
            '</div>'
            
            '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            '<span style="color:#94A3B8;font-size:13px;">Recall</span>'
            '<span style="color:#F9FAFB;font-weight:500;font-size:13px;">31.7%</span>'
            '</div>'
            
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
            '<span style="color:#94A3B8;font-size:13px;">Algorithm</span>'
            '<span style="color:#F9FAFB;font-weight:500;font-size:13px;">Ensemble Fusion</span>'
            '</div>'
            
            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            '<span style="color:#94A3B8;font-size:13px;">Explainability</span>'
            '<span style="color:#F9FAFB;font-weight:500;font-size:13px;">SHAP</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # TOP HEADER BAR
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid #1F2937;">'
        '<div>'
        '<div style="font-size:15px;font-weight:600;color:#3B82F6;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">AI-Powered Behavioral Anomaly Detection for Cybersecurity</div>'
        '<h1 style="font-size:24px;font-weight:600;margin:0;color:#F9FAFB;">System Overview</h1>'
        '<div style="font-size:14px;color:#94A3B8;margin-top:4px;">Monitor and analyze behavioral anomalies across the infrastructure.<br>'
        '<span style="color:#10B981;font-weight:500;">Business Impact:</span> At a 5% alert budget, analysts review only high-risk sessions — an estimated 20x reduction in manual triage time.</div>'
        '</div>'
        
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<span style="display:inline-flex;align-items:center;gap:6px;background-color:rgba(16, 185, 129, 0.1);color:#10B981;padding:6px 12px;border-radius:6px;font-size:13px;font-weight:500;">'
        '<div style="width:6px;height:6px;background-color:#10B981;border-radius:50%;"></div>'
        'System Healthy'
        '</span>'
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
        st.markdown(kpi_card("Processed Events", f"{len(df_recent):,}", "Last 24 hours"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Active Alerts", str(total_alts), "Pending investigation"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Critical Threats", str(high_risk), "Risk score > 0.80"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Detection Recall", "31.7%", "Top 5% Alert Budget"), unsafe_allow_html=True)

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

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
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
        '<div style="font-size:18px;font-weight:600;color:#F9FAFB;">Alert Queue</div>'
        '<div style="font-size:13px;color:#94A3B8;">' + str(len(df_filtered)) + ' items found</div>'
        '</div>',
        unsafe_allow_html=True
    )

    if not df_filtered.empty:
        df_display = df_filtered.head(15)
        from explain.explainer import AlertExplainer
        explainer = AlertExplainer(classifier.model, classifier.features)
        
        rows = ""
        for _, row_display in df_display.iterrows():
            # Generate explanation on the fly
            X_feats = classifier._engineer_features(pd.DataFrame([row_display['RowData']]))
            missing = [c for c in classifier.features if c not in X_feats.columns]
            if missing:
                X_feats = pd.concat([X_feats, pd.DataFrame(0, index=X_feats.index, columns=missing)], axis=1)
            X_feats = X_feats[classifier.features].astype(float)
            reason = explainer.explain_alert(X_feats, row_display['Predicted Type'])
            
            short_ent = str(row_display['Entity'])[:28] + '…' if len(str(row_display['Entity'])) > 28 else str(row_display['Entity'])
            short_ts  = str(row_display['Timestamp'])[:16]
            expl      = str(reason)[:85] + '…' if len(str(reason)) > 85 else str(reason)

            rows += (
                '<tr style="border-bottom:1px solid #1F2937;transition:background-color 0.15s ease;" onmouseover="this.style.backgroundColor=\'#1E293B\'" onmouseout="this.style.backgroundColor=\'transparent\'">'
                '<td class="mono" style="padding:16px;font-size:12px;color:#94A3B8;white-space:nowrap;">' + short_ts + '</td>'
                '<td style="padding:16px;">'
                '<div class="mono" style="font-size:13px;color:#F9FAFB;margin-bottom:4px;">' + short_ent + '</div>'
                '<div>' + et_badge(row_display['Entity Type']) + '</div>'
                '</td>'
                '<td style="padding:16px;">' + threat_badge(row_display['Predicted Type']) + '</td>'
                '<td style="padding:16px;width:140px;">' + risk_bar(row_display['Risk Score']) + '</td>'
                '<td style="padding:16px;font-size:13px;color:#94A3B8;max-width:300px;line-height:1.5;">' + expl + '</td>'
                '<td style="padding:16px;">' + conf_badge(row_display['Confidence']) + '</td>'
                '</tr>'
            )

        st.markdown(
            '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;overflow:hidden;margin-bottom:48px;">'
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;text-align:left;" class="alert-tbl">'
            '<thead><tr style="border-bottom:1px solid #1F2937;background-color:#1E293B;">'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">Timestamp</th>'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">Entity</th>'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">Threat Type</th>'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">Risk Score</th>'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">AI Explanation</th>'
            '<th style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;">Status</th>'
            '</tr></thead>'
            '<tbody>' + rows + '</tbody>'
            '</table></div></div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:48px;text-align:center;margin-bottom:48px;">'
            '<div style="color:#94A3B8;font-size:15px;">No active alerts matching the current filter criteria.</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ENTITY DEEP-DIVE
    # ──────────────────────────────────────────────────────────────────────────
    if not df_filtered.empty:
        st.markdown(
            '<div style="font-size:18px;font-weight:600;color:#F9FAFB;margin-bottom:16px;">Entity Investigation</div>',
            unsafe_allow_html=True
        )

        selected_entity = st.selectbox(
            "Select Entity ID",
            options=df_filtered['Entity'].unique(),
            key="entity_select",
            label_visibility="collapsed"
        )

        if selected_entity:
            entity_history = df[df['entity_id'] == selected_entity].copy()
            entity_row     = df_filtered[df_filtered['Entity'] == selected_entity].iloc[0]
            
            # Key Details Cards
            st.markdown(
                '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:16px;margin-bottom:24px;">'
                
                '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:20px;">'
                '<div style="color:#94A3B8;font-size:12px;margin-bottom:8px;">Entity ID</div>'
                '<div class="mono" style="color:#F9FAFB;font-size:14px;word-break:break-all;">' + str(selected_entity) + '</div>'
                '</div>'
                
                '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:20px;">'
                '<div style="color:#94A3B8;font-size:12px;margin-bottom:8px;">Entity Type</div>'
                '<div style="color:#F9FAFB;font-size:14px;font-weight:500;">' + entity_row['Entity Type'].replace("_", " ").title() + '</div>'
                '</div>'
                
                '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:20px;">'
                '<div style="color:#94A3B8;font-size:12px;margin-bottom:8px;">Primary Threat</div>'
                '<div>' + threat_badge(entity_row['Predicted Type']) + '</div>'
                '</div>'
                
                '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:20px;">'
                '<div style="color:#94A3B8;font-size:12px;margin-bottom:8px;">Total Sessions</div>'
                '<div style="color:#F9FAFB;font-size:24px;font-weight:600;">' + str(len(entity_history)) + '</div>'
                '</div>'
                
                '</div>',
                unsafe_allow_html=True
            )

            col_l, col_r = st.columns([2, 1], gap="large")

            with col_l:
                st.markdown('<div style="font-size:15px;font-weight:600;color:#F9FAFB;margin-bottom:16px;">Session Timeline</div>', unsafe_allow_html=True)
                
                entity_history['timestamp'] = pd.to_datetime(entity_history['timestamp'], errors='coerce')
                fig_tl = px.scatter(
                    entity_history.dropna(subset=['timestamp']),
                    x='timestamp', y='session_duration',
                    color='auth_status',
                    color_discrete_map={"success": "#10B981", "failed": "#EF4444"},
                    labels={'timestamp': '', 'session_duration': 'Duration (s)', 'auth_status': 'Authentication'},
                    hover_data=['geo_location', 'resource_accessed']
                )
                fig_tl.update_traces(marker=dict(size=8, opacity=0.9, line=dict(width=0)))
                fig_tl.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300,
                    font=dict(family='Inter', color='#94A3B8', size=12),
                    xaxis=dict(gridcolor='#1F2937', linecolor='#1F2937', tickfont=dict(color='#94A3B8')),
                    yaxis=dict(gridcolor='#1F2937', linecolor='#1F2937', tickfont=dict(color='#94A3B8')),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#F9FAFB'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})
                
                st.markdown('<div style="font-size:15px;font-weight:600;color:#F9FAFB;margin-top:32px;margin-bottom:16px;">Activity Distribution</div>', unsafe_allow_html=True)
                
                if 'timestamp' in entity_history.columns:
                    entity_history['hour'] = pd.to_datetime(entity_history['timestamp'], errors='coerce').dt.hour
                    hourly = entity_history.groupby('hour').size().reindex(range(24), fill_value=0)

                    fig_h = go.Figure(go.Bar(
                        x=list(range(24)), y=hourly.values,
                        marker=dict(color='#3B82F6', opacity=0.8, line=dict(width=0)),
                        hovertemplate='%{x}:00 - %{y} sessions<extra></extra>'
                    ))
                    fig_h.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220,
                        font=dict(family='Inter', color='#94A3B8', size=12),
                        xaxis=dict(title='Hour of Day (UTC)', gridcolor='#1F2937', tickfont=dict(color='#94A3B8'), tickmode='linear', tick0=0, dtick=4),
                        yaxis=dict(title='Session Count', gridcolor='#1F2937', tickfont=dict(color='#94A3B8')),
                        margin=dict(t=0, b=0, l=0, r=0),
                        bargap=0.2,
                    )
                    st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})

            with col_r:
                st.markdown('<div style="font-size:15px;font-weight:600;color:#F9FAFB;margin-bottom:16px;">Behavioral Profile</div>', unsafe_allow_html=True)
                
                profile_html = '<div style="background-color:#111827;border:1px solid #1F2937;border-radius:8px;padding:24px;">'

                prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
                if prof:
                    typical_geos      = list(prof['geo_dist'].keys())[:4]
                    typical_resources = list(prof['resource_dist'].keys())[:4]
                    avg_duration      = round(prof['duration_mean'], 1)

                    profile_html += '<div style="color:#94A3B8;font-size:13px;font-weight:500;margin-bottom:12px;">Frequent Locations</div>'
                    for g in typical_geos:
                        profile_html += (
                            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                            '<div style="width:4px;height:4px;background-color:#64748B;border-radius:50%;"></div>'
                            '<div style="color:#F9FAFB;font-size:14px;">' + str(g) + '</div>'
                            '</div>'
                        )

                    profile_html += '<div style="color:#94A3B8;font-size:13px;font-weight:500;margin:24px 0 12px 0;">Accessed Resources</div>'
                    for r in typical_resources:
                        profile_html += (
                            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                            '<div style="width:4px;height:4px;background-color:#64748B;border-radius:50%;"></div>'
                            '<div style="color:#F9FAFB;font-size:14px;">' + str(r) + '</div>'
                            '</div>'
                        )

                    profile_html += '<div style="color:#94A3B8;font-size:13px;font-weight:500;margin:24px 0 8px 0;">Average Duration</div>'
                    profile_html += '<div style="color:#F9FAFB;font-size:20px;font-weight:600;">' + str(avg_duration) + 's</div>'
                else:
                    profile_html += '<div style="color:#94A3B8;font-size:14px;">No historical profile established for this entity.</div>'

                profile_html += '</div>'
                st.markdown(profile_html, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:64px;padding-top:24px;border-top:1px solid #1F2937;display:flex;justify-content:space-between;color:#64748B;font-size:13px;">'
        '<div>&copy; 2026 Sentinel-AI — Honeywell Campus Connect Hackathon Prototype</div>'
        '<div>v1.0.0</div>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
