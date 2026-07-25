import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor

class SequenceAnomalyModel:
    """
    Fallback sequence detector using scikit-learn's MLP (Multi-Layer Perceptron)
    because PyTorch DLLs are missing on this Windows machine.
    """
    def __init__(self, seq_len=5, hidden_layer_sizes=(32, 16), max_iter=200, **kwargs):
        self.seq_len = seq_len
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def _preprocess(self, df, fit=False):
        df_processed = pd.DataFrame()
        cats = ['entity_type', 'geo_location', 'resource_accessed', 'auth_method', 'auth_status']
        for cat in cats:
            if fit:
                self.encoders[cat] = LabelEncoder()
                df_processed[cat] = self.encoders[cat].fit_transform(df[cat].astype(str))
            else:
                le = self.encoders[cat]
                classes = le.classes_.tolist()
                df_processed[cat] = df[cat].astype(str).map(lambda s: s if s in classes else '<unknown>')
                if '<unknown>' not in classes:
                    classes.append('<unknown>')
                    le.classes_ = np.array(classes)
                df_processed[cat] = le.transform(df_processed[cat])
                
        nums = ['session_duration']
        df_processed['session_duration'] = df['session_duration'].values
        df_processed['hour'] = pd.to_datetime(df['timestamp']).dt.hour.values
        nums.append('hour')
        
        if fit:
            df_processed[nums] = self.scaler.fit_transform(df_processed[nums])
        else:
            df_processed[nums] = self.scaler.transform(df_processed[nums])
            
        for col in df_processed.columns:
            min_val = df_processed[col].min()
            max_val = df_processed[col].max()
            if max_val > min_val:
                df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
            else:
                df_processed[col] = 0.5
                
        return df_processed.values

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            # Flatten the sequence for MLP
            X.append(data[i:i+self.seq_len].flatten())
            y.append(data[i+self.seq_len]) 
        return np.array(X), np.array(y)

    def fit(self, df_normal):
        df_normal = df_normal.sort_values(by=['entity_id', 'timestamp'])
        processed_data = self._preprocess(df_normal, fit=True)
        
        X_all, y_all = [], []
        entities = df_normal['entity_id'].values
        current_entity = entities[0]
        start_idx = 0
        
        for i in range(1, len(entities)):
            if entities[i] != current_entity or i == len(entities) - 1:
                end_idx = i if entities[i] != current_entity else i + 1
                entity_data = processed_data[start_idx:end_idx]
                if len(entity_data) > self.seq_len:
                    X_ent, y_ent = self._create_sequences(entity_data)
                    X_all.append(X_ent)
                    y_all.append(y_ent)
                start_idx = i
                current_entity = entities[i]
                
        if len(X_all) == 0:
            print("Not enough data to create sequences.")
            return
            
        X = np.vstack(X_all)
        y = np.vstack(y_all)
        
        self.model.fit(X, y)
                
    def score_sequences(self, df_test):
        df_test = df_test.sort_values(by=['entity_id', 'timestamp'])
        processed_data = self._preprocess(df_test, fit=False)
        entities = df_test['entity_id'].values
        timestamps = df_test['timestamp'].values
        
        results = []
        current_entity = entities[0]
        start_idx = 0
        
        for i in range(1, len(entities)):
            if entities[i] != current_entity or i == len(entities) - 1:
                end_idx = i if entities[i] != current_entity else i + 1
                entity_data = processed_data[start_idx:end_idx]
                
                if len(entity_data) > self.seq_len:
                    X_ent, y_ent = self._create_sequences(entity_data)
                    preds = self.model.predict(X_ent)
                    
                    mse = np.mean((preds - y_ent) ** 2, axis=1)
                    seq_timestamps = timestamps[start_idx + self.seq_len:end_idx]
                    
                    for ts, error in zip(seq_timestamps, mse):
                        results.append({
                            'entity_id': current_entity,
                            'timestamp': ts,
                            'anomaly_score': float(error)
                        })
                        
                start_idx = i
                current_entity = entities[i]
                
        return pd.DataFrame(results)
