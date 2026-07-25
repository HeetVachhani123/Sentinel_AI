import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class SessionAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super(SessionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # assuming normalized input [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderBaseline:
    def __init__(self, encoding_dim=16, epochs=20, batch_size=256):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.input_dim = 0
        
    def _preprocess(self, df, fit=False):
        # We need to turn categorical variables into numeric and scale numerical variables
        df_processed = pd.DataFrame()
        
        # Categoricals
        cats = ['entity_type', 'geo_location', 'resource_accessed', 'auth_method', 'auth_status']
        for cat in cats:
            if fit:
                self.encoders[cat] = LabelEncoder()
                df_processed[cat] = self.encoders[cat].fit_transform(df[cat].astype(str))
            else:
                # Handle unknown labels
                df_processed[cat] = df[cat].astype(str).map(lambda s: s if s in self.encoders[cat].classes_ else '<unknown>')
                
                # We need a hack for unknown values: append it temporarily to classes if not present
                le = self.encoders[cat]
                classes = le.classes_.tolist()
                if '<unknown>' not in classes:
                    classes.append('<unknown>')
                    le.classes_ = np.array(classes)
                df_processed[cat] = le.transform(df_processed[cat])
                
        # Numericals
        nums = ['session_duration']
        # Add hour
        df_processed['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        nums.append('hour')
        
        if fit:
            df_processed[nums] = self.scaler.fit_transform(df[nums])
        else:
            df_processed[nums] = self.scaler.transform(df[nums])
            
        # MinMax scale to [0, 1] for Sigmoid
        for col in df_processed.columns:
            min_val = df_processed[col].min()
            max_val = df_processed[col].max()
            if max_val > min_val:
                df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
            else:
                df_processed[col] = 0.5
                
        return torch.tensor(df_processed.values, dtype=torch.float32)

    def fit(self, df_normal):
        X = self._preprocess(df_normal, fit=True)
        self.input_dim = X.shape[1]
        
        self.model = SessionAutoencoder(self.input_dim, self.encoding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        dataset = torch.utils.data.TensorDataset(X, X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader)}")
            
    def score_session(self, session_df):
        """Score a session or multiple sessions. Returns array of reconstruction errors."""
        self.model.eval()
        X = self._preprocess(session_df, fit=False)
        with torch.no_grad():
            reconstructed = self.model(X)
            mse = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
        return mse
