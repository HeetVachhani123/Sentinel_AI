import pandas as pd
import numpy as np
import sys
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Lock in all sources of randomness for identical reproducible results
np.random.seed(42)
random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except Exception:
    pass

# Add parent directory to path to allow importing data_gen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline import StatisticalBaselineProfiler
try:
    from sequence_detector import SequenceAnomalyModel
except OSError as e:
    print(f"Warning: PyTorch failed to load due to missing Windows DLLs ({e}). Using Scikit-Learn fallback model instead.")
    from sequence_detector_fallback import SequenceAnomalyModel
from classifier import AnomalyClassifier

def evaluate_detector(y_true, y_scores, top_percent=0.01):
    # Sort by score descending
    df_eval = pd.DataFrame({'true': y_true, 'score': y_scores})
    df_eval = df_eval.sort_values(by='score', ascending=False)
    
    # Threshold for top 1%
    threshold_idx = int(len(df_eval) * top_percent)
    threshold = df_eval.iloc[threshold_idx]['score'] if threshold_idx < len(df_eval) else 0
    
    y_pred = (df_eval['score'] >= threshold).astype(int)
    y_true_binary = (df_eval['true'] != 'normal').astype(int)
    
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    
    # Try ROC AUC
    try:
        roc_auc = roc_auc_score(y_true_binary, df_eval['score'])
    except ValueError:
        roc_auc = 0.5
        
    # False positive rate (FPR) = FP / N
    fp = ((y_pred == 1) & (y_true_binary == 0)).sum()
    tn = ((y_pred == 0) & (y_true_binary == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"--- Detector Evaluation (Top {top_percent*100}% budget) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print("-------------------------------------------------")
    
    return y_pred

if __name__ == "__main__":
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prod_path = os.path.join(base_dir, 'data', 'production_logs.csv')
    labels_path = os.path.join(base_dir, 'data', 'labels_holdout.csv')
    
    if not os.path.exists(prod_path):
        print("Data not found. Please run data_gen/generate.py first.")
        sys.exit(1)
        
    df_prod = pd.read_csv(prod_path)
    df_labels = pd.read_csv(labels_path)
    df_full = pd.concat([df_prod, df_labels], axis=1)
    
    # Split into normal training data (first 70% of chronological data, roughly)
    # Actually, we should train only on normal data for the baseline
    df_normal = df_full[df_full['label'] == 'normal']
    train_size = int(len(df_normal) * 0.7)
    df_train_normal = df_normal.iloc[:train_size]
    
    # For testing, we use the remaining 30% of time + all anomalies
    cutoff_time = df_train_normal['timestamp'].max()
    df_test = df_full[df_full['timestamp'] > cutoff_time]
    
    print(f"Training Baseline on {len(df_train_normal)} normal sessions...")
    baseline = StatisticalBaselineProfiler()
    baseline.fit(df_train_normal)
    
    print(f"Training Sequence Detector on {len(df_train_normal)} normal sessions...")
    seq_detector = SequenceAnomalyModel(epochs=2) # Keep epochs low for speed
    seq_detector.fit(df_train_normal)
    
    print("Scoring test set with Baseline and Sequence Detector...")
    
    # Score with Baseline
    baseline_scores = []
    for _, row in df_test.iterrows():
        b_score, _ = baseline.score_session(row)
        baseline_scores.append(b_score)
    df_test = df_test.copy()
    df_test['baseline_score'] = baseline_scores

    # Score with sequence detector
    seq_scores_df = seq_detector.score_sequences(df_test)
    
    # We need to align the sequence scores with the original test df
    # Because sequence scoring drops the first `seq_len` items per entity
    df_test_scored = pd.merge(df_test, seq_scores_df, on=['entity_id', 'timestamp'], how='inner')
    
    # Min-max scale sequence scores so they are comparable to baseline scores (0-1)
    seq_min = df_test_scored['anomaly_score'].min()
    seq_max = df_test_scored['anomaly_score'].max()
    if seq_max > seq_min:
        df_test_scored['seq_score_scaled'] = (df_test_scored['anomaly_score'] - seq_min) / (seq_max - seq_min)
    else:
        df_test_scored['seq_score_scaled'] = 0.0

    # Combine scores (Max is great because if either detector is confident, it's anomalous)
    df_test_scored['combined_score'] = df_test_scored[['baseline_score', 'seq_score_scaled']].max(axis=1)
    
    y_true = df_test_scored['label']
    y_scores = df_test_scored['combined_score']
    
    evaluate_detector(y_true, y_scores, top_percent=0.01)
    evaluate_detector(y_true, y_scores, top_percent=0.02)
    evaluate_detector(y_true, y_scores, top_percent=0.03)
    evaluate_detector(y_true, y_scores, top_percent=0.04)
    evaluate_detector(y_true, y_scores, top_percent=0.05)
    
    print("Training Anomaly Classifier on ALL anomalies (supervised)...")
    # In practice, you'd train this on historical confirmed incidents
    df_anomalies = df_full[df_full['label'] != 'normal']
    # Add some normal data too
    df_clf_data = pd.concat([df_anomalies, df_full[df_full['label'] == 'normal'].sample(len(df_anomalies), random_state=42)])
    
    # Proper 80/20 train/test split for the classifier
    df_clf_train, df_clf_test = train_test_split(df_clf_data, test_size=0.2, random_state=42, stratify=df_clf_data['label'])
    
    classifier = AnomalyClassifier()
    classifier.fit(df_clf_train)
    
    print("Evaluating Classifier on held-out test set (20%)...")
    classifier.evaluate(df_clf_test)
    
    print("Saving models...")
    models_dir = os.path.join(base_dir, 'models', 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    import pickle
    with open(os.path.join(models_dir, 'baseline.pkl'), 'wb') as f:
        pickle.dump(baseline, f)
    with open(os.path.join(models_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(classifier, f)
    # Note: Sequence model (PyTorch) needs torch.save, but skipping for simplicity in demo
