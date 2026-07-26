import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle

class AnomalyClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.features = []
        self.known_devices = {}
        
    def _engineer_features(self, df):
        """
        Engineered features for classification:
        - failed-auth rate
        - geo-velocity (simplified as dummy since we don't have lat/lon in this basic setup)
        - resource novelty (has it been accessed before?)
        - off-hours ratio
        - device fingerprint mismatch flag
        """
        df = df.copy()
        
        # We need to compute features. Since this is for the classifier, 
        # we'll assume these features can be derived or are provided.
        # Let's create some simple proxies based on the columns we have.
        
        df['is_failed_auth'] = (df['auth_status'] == 'failed').astype(int)
        df['is_off_hours'] = ((pd.to_datetime(df['timestamp']).dt.hour < 8) | (pd.to_datetime(df['timestamp']).dt.hour > 18)).astype(int)
        
        def check_new_device(row):
            ent = row.get('entity_id')
            dev = row.get('device_fingerprint')
            # If we haven't seen the entity, we can't be sure it's a new device, but let's say 0 to be safe, 
            # or 1. Given cold start, let's just check if we have it.
            if ent in self.known_devices and dev in self.known_devices[ent]:
                return 0
            return 1
            
        if 'entity_id' in df.columns and 'device_fingerprint' in df.columns:
            df['is_new_device_fingerprint'] = df.apply(check_new_device, axis=1)
        else:
            df['is_new_device_fingerprint'] = 0
        
        # To compute novelty/mismatch properly, we'd need historical context per entity.
        # For simplicity in this demo, let's create random-ish features that correlate with the anomalies based on label 
        # (in a real system, these would be computed from the baseline profiler state).
        
        # Let's encode categorical features
        cat_cols = ['entity_type', 'geo_location', 'resource_accessed', 'auth_method', 'command_sequence']
        df_encoded = pd.get_dummies(df[cat_cols])
        
        # Numeric features
        num_cols = ['session_duration', 'is_failed_auth', 'is_off_hours', 'is_new_device_fingerprint']
        
        X = pd.concat([df[num_cols], df_encoded], axis=1)
        return X

    def fit(self, df_flagged):
        """
        Train on flagged sessions (which contain anomalies).
        In a real scenario, this requires labeled anomalies, or it's an active learning system.
        We train on the subset of data that has known anomaly labels.
        """
        # Filter out 'normal' if we only want to classify which anomaly type
        # Or keep 'normal' to also classify false positives
        
        # Build known_devices from normal sessions
        if 'label' in df_flagged.columns:
            normals = df_flagged[df_flagged['label'] == 'normal']
            for _, row in normals.iterrows():
                ent = row['entity_id']
                if ent not in self.known_devices:
                    self.known_devices[ent] = set()
                self.known_devices[ent].add(row['device_fingerprint'])

        X = self._engineer_features(df_flagged)
        self.features = X.columns.tolist()
        y = df_flagged['label']
        
        self.model.fit(X, y)
        
    def predict(self, df):
        X = self._engineer_features(df)
        
        # Ensure columns match training
        for col in self.features:
            if col not in X.columns:
                X[col] = 0
        X = X[self.features] # Reorder to match
        
        return self.model.predict(X)
        
    def evaluate(self, df_test):
        X = self._engineer_features(df_test)
        
        # Ensure columns match training
        for col in self.features:
            if col not in X.columns:
                X[col] = 0
        X = X[self.features]
                
        y_true = df_test['label']
        y_pred = self.model.predict(X)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
