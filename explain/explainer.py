import shap
import numpy as np
import pandas as pd

class AlertExplainer:
    def __init__(self, classifier_model, feature_names):
        self.classifier = classifier_model
        self.feature_names = feature_names
        # We'll use TreeExplainer since the classifier is a RandomForest
        # For fast execution, we initialize it once.
        self.explainer = shap.TreeExplainer(self.classifier)
        
    def explain_alert(self, session_features_df, anomaly_type):
        """
        Takes a single row dataframe of engineered features for a flagged session,
        and the predicted anomaly type.
        Returns a short human-readable reason string based on top SHAP values.
        """
        # Ensure correct column order
        X = session_features_df[self.feature_names]
        
        # Calculate SHAP values (disable additivity check to prevent precision-related crashes)
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        
        # For multiclass, shap_values is a list of arrays (one per class)
        # Or an array of shape (samples, features, classes) depending on SHAP version
        # Let's handle both dynamically
        
        if isinstance(shap_values, list):
            # Find index of predicted anomaly type
            try:
                class_idx = list(self.classifier.classes_).index(anomaly_type)
                vals = shap_values[class_idx][0]
            except ValueError:
                vals = shap_values[0][0] # Fallback
        elif len(shap_values.shape) == 3:
            try:
                class_idx = list(self.classifier.classes_).index(anomaly_type)
                vals = shap_values[0, :, class_idx]
            except ValueError:
                vals = shap_values[0, :, 0]
        else:
            vals = shap_values[0]
            
        # Get indices of top 2 contributing features
        top_indices = np.argsort(-np.abs(vals))[:2]
        
        top_features = []
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            # Make it human readable
            if feat_name == 'is_failed_auth':
                feat_name = 'failed authentication'
            elif feat_name == 'is_off_hours':
                feat_name = 'off-hours access'
            elif feat_name.startswith('geo_location_'):
                feat_name = f"unusual location ({feat_name.split('_')[-1]})"
            elif feat_name.startswith('resource_accessed_'):
                feat_name = f"unusual resource ({feat_name.split('_')[-1]})"
            elif feat_name.startswith('command_sequence_'):
                cmd_part = feat_name.replace('command_sequence_', '')
                feat_name = f"unusual command sequence involving '{cmd_part}'"
            elif feat_name == 'session_duration':
                feat_name = 'unusual session duration'
            elif feat_name == 'data_transferred':
                feat_name = 'abnormal data transfer volume'
            
            top_features.append(feat_name)
            
        reason = f"Flagged as {anomaly_type} due to {top_features[0]} and {top_features[1]}"
        return reason
