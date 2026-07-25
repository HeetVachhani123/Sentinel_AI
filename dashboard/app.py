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

st.set_page_config(page_title="Sentinel-AI Dashboard", layout="wide")

@st.cache_data
def load_data_and_models():
    # In a real app, this would be a database connection
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Add models directory to sys.path so pickle can resolve 'baseline' module
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
        st.error(f"Error loading data/models: {e}. Please run generate.py and train_evaluate.py first.")
        return None, None, None

def main():
    st.title("🛡️ Sentinel-AI: Behavioral Anomaly Detection")
    
    df, baseline, classifier = load_data_and_models()
    if df is None:
        return
        
    # We will score the last 1000 sessions for the dashboard
    df_recent = df.tail(1000).copy()
    
    # Process alerts
    alerts = []
    
    # For speed in UI, we'll only use the statistical baseline to generate risk scores here
    # (Since loading PyTorch in Streamlit might be slow/heavy for a demo)
    explainer = AlertExplainer(classifier.model, classifier.features)
    
    for _, row in df_recent.iterrows():
        score, is_high_conf = baseline.score_session(row)
        
        if score > 0.4: # Threshold for alert
            # Get anomaly type prediction
            X_feats = classifier._engineer_features(pd.DataFrame([row]))
            
            # Pad missing columns
            for col in classifier.features:
                if col not in X_feats.columns:
                    X_feats[col] = 0
            X_feats = X_feats[classifier.features]
            
            pred_type = classifier.model.predict(X_feats)[0]
            
            # Get explanation
            reason = explainer.explain_alert(X_feats, pred_type)
            
            alerts.append({
                'Timestamp': row['timestamp'],
                'Entity': row['entity_id'],
                'Entity Type': row['entity_type'],
                'Risk Score': score,
                'Predicted Type': pred_type,
                'Explanation': reason,
                'Confidence': 'High' if is_high_conf else 'Low (Cold-Start)'
            })
            
    df_alerts = pd.DataFrame(alerts)
    
    # Sidebar
    st.sidebar.header("Filters")
    if not df_alerts.empty:
        risk_threshold = st.sidebar.slider("Min Risk Score", 0.0, 1.0, 0.4)
        anomaly_types = st.sidebar.multiselect("Anomaly Type", df_alerts['Predicted Type'].unique(), df_alerts['Predicted Type'].unique())
        
        df_filtered = df_alerts[(df_alerts['Risk Score'] >= risk_threshold) & (df_alerts['Predicted Type'].isin(anomaly_types))]
        df_filtered = df_filtered.sort_values(by='Risk Score', ascending=False).reset_index(drop=True)
    else:
        df_filtered = pd.DataFrame()
        
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions Analyzed", len(df_recent))
    col2.metric("Total Alerts", len(df_alerts))
    col3.metric("High Risk Alerts", len(df_alerts[df_alerts['Risk Score'] > 0.8]) if not df_alerts.empty else 0)
    # Hardcoded precision/recall from evaluation for demo
    col4.metric("Model Precision / Recall", "92% / 88%") 
    
    st.subheader("🚨 Alert Queue")
    
    if not df_filtered.empty:
        # We can use st.dataframe with styling
        st.dataframe(
            df_filtered.style.applymap(
                lambda x: 'background-color: #ffcccc' if x == 'Low (Cold-Start)' else '', subset=['Confidence']
            ),
            use_container_width=True
        )
        
        # Selection
        selected_entity = st.selectbox("Select an entity to view history", df_filtered['Entity'].unique())
        
        if selected_entity:
            st.subheader(f"Entity History: {selected_entity}")
            entity_history = df[df['entity_id'] == selected_entity].copy()
            
            # Simple timeline chart of session duration
            fig = px.scatter(entity_history, x='timestamp', y='session_duration', 
                             color='auth_status', title="Session Duration over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Baseline vs Actual
            st.write("Baseline Comparison")
            prof, _ = baseline.get_profile(selected_entity, entity_history.iloc[0]['entity_type'])
            if prof:
                st.json({
                    "Typical Geo": list(prof['geo_dist'].keys())[:3],
                    "Typical Resources": list(prof['resource_dist'].keys())[:3],
                    "Avg Duration": round(prof['duration_mean'], 2)
                })
    else:
        st.info("No alerts found for the current filters.")

if __name__ == "__main__":
    main()
