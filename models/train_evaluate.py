import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path to allow importing data_gen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline import StatisticalBaselineProfiler
from sequence_detector import SequenceAnomalyModel
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
    prod_path = '../data/production_logs.csv'
    labels_path = '../data/labels_holdout.csv'
    
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
    # Score with sequence detector
    seq_scores_df = seq_detector.score_sequences(df_test)
    
    # We need to align the sequence scores with the original test df
    # Because sequence scoring drops the first `seq_len` items per entity
    df_test_scored = pd.merge(df_test, seq_scores_df, on=['entity_id', 'timestamp'], how='inner')
    
    y_true = df_test_scored['label']
    y_scores = df_test_scored['anomaly_score']
    
    evaluate_detector(y_true, y_scores, top_percent=0.01)
    
    print("Training Anomaly Classifier on ALL anomalies (supervised)...")
    # In practice, you'd train this on historical confirmed incidents
    df_anomalies = df_full[df_full['label'] != 'normal']
    # Add some normal data too
    df_clf_train = pd.concat([df_anomalies, df_full[df_full['label'] == 'normal'].sample(len(df_anomalies))])
    
    classifier = AnomalyClassifier()
    classifier.fit(df_clf_train)
    
    print("Evaluating Classifier on training set (just as a sanity check)...")
    classifier.evaluate(df_clf_train)
    
    print("Saving models...")
    os.makedirs('saved_models', exist_ok=True)
    import pickle
    with open('saved_models/baseline.pkl', 'wb') as f:
        pickle.dump(baseline, f)
    with open('saved_models/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    # Note: Sequence model (PyTorch) needs torch.save, but skipping for simplicity in demo
