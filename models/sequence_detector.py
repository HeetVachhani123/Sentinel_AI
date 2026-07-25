import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class SequenceDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(SequenceDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim) # Predict next step features
        
    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        # We want to predict the next step, so we use the output to predict the features
        out = self.fc(out)
        return out

class SequenceAnomalyModel:
    def __init__(self, seq_len=5, hidden_dim=32, epochs=10, batch_size=128):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
        self.scaler = StandardScaler()
        self.encoders = {}
        self.input_dim = 0
        
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
        df_processed['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        nums.append('hour')
        
        if fit:
            df_processed[nums] = self.scaler.fit_transform(df[nums])
        else:
            df_processed[nums] = self.scaler.transform(df[nums])
            
        for col in df_processed.columns:
            min_val = df_processed[col].min()
            max_val = df_processed[col].max()
            if max_val > min_val:
                df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
            else:
                df_processed[col] = 0.5
                
        return df_processed.values

    def _create_sequences(self, data):
        # Creates sequences for LSTM. (num_samples, seq_len, features)
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+1:i+self.seq_len+1]) # Target is shifted by 1
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def fit(self, df_normal):
        # To train on sequences, we should really group by entity_id and sort by timestamp
        df_normal = df_normal.sort_values(by=['entity_id', 'timestamp'])
        processed_data = self._preprocess(df_normal, fit=True)
        self.input_dim = processed_data.shape[1]
        
        X_all = []
        y_all = []
        
        # We need to build sequences per entity
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
            
        X = torch.cat(X_all, dim=0)
        y = torch.cat(y_all, dim=0)
        
        self.model = SequenceDetector(self.input_dim, self.hidden_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
    def score_sequences(self, df_test):
        """
        Returns reconstruction error for the LAST element in each sequence.
        To do this properly, df_test should be sorted by entity_id and timestamp.
        Returns a DataFrame with entity_id, timestamp, and anomaly_score.
        """
        self.model.eval()
        df_test = df_test.sort_values(by=['entity_id', 'timestamp'])
        processed_data = self._preprocess(df_test, fit=False)
        entities = df_test['entity_id'].values
        timestamps = df_test['timestamp'].values
        
        results = []
        
        current_entity = entities[0]
        start_idx = 0
        
        with torch.no_grad():
            for i in range(1, len(entities)):
                if entities[i] != current_entity or i == len(entities) - 1:
                    end_idx = i if entities[i] != current_entity else i + 1
                    entity_data = processed_data[start_idx:end_idx]
                    
                    if len(entity_data) > self.seq_len:
                        X_ent, y_ent = self._create_sequences(entity_data)
                        preds = self.model(X_ent)
                        
                        # Calculate MSE for the last step in each sequence
                        # Preds shape: (batch, seq_len, features)
                        # We care about the prediction of the last element compared to the actual last element
                        last_step_preds = preds[:, -1, :]
                        last_step_actuals = y_ent[:, -1, :]
                        mse = torch.mean((last_step_preds - last_step_actuals) ** 2, dim=1).numpy()
                        
                        # The corresponding timestamps are from start_idx + seq_len to end_idx
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
