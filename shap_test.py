import sys
import os
import pickle
import pandas as pd
import numpy as np
import shap

sys.path.append(os.path.abspath('models'))
from classifier import AnomalyClassifier

with open('models/saved_models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

print(f"Total features: {len(classifier.features)}")

df = pd.read_csv('data/production_logs.csv').head(500)
X = classifier._engineer_features(df)
for col in classifier.features:
    if col not in X.columns:
        X[col] = 0
X = X[classifier.features]

explainer = shap.TreeExplainer(classifier.model)
shap_values = explainer.shap_values(X, check_additivity=False)

print('\nClasses:', classifier.model.classes_)
try:
    ds_idx = list(classifier.model.classes_).index('device_spoofing')
    if isinstance(shap_values, list):
        vals = shap_values[ds_idx]
    else:
        vals = shap_values[:, :, ds_idx]
        
    mean_abs_shap = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    print('\nTop 5 features for device_spoofing:')
    for idx in top_indices:
        print(f"{classifier.features[idx]}: {mean_abs_shap[idx]:.4f}")
except Exception as e:
    print('Error:', e)
