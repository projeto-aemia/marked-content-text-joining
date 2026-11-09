"""
Training script for Tier 0: GBDT Classifier
Loads train/val/test splits, trains LightGBM, evaluates metrics, and saves the model.
"""

import os
import sys
import json

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_builder import extract_batch_features
from src.models.gbdt_model import GBDTBoundaryClassifier


def train_and_eval_gbdt(
    data_dir: str = "data",
    model_output_path: str = "saved_models/gbdt_model.pkl"
):
    print("Loading datasets...")
    with open(os.path.join(data_dir, "train_boundaries.json"), "r", encoding="utf-8") as f:
        train_boundaries = json.load(f)
    with open(os.path.join(data_dir, "val_boundaries.json"), "r", encoding="utf-8") as f:
        val_boundaries = json.load(f)
    with open(os.path.join(data_dir, "test_boundaries.json"), "r", encoding="utf-8") as f:
        test_boundaries = json.load(f)
        
    print(f"Extracting features for {len(train_boundaries)} train samples...")
    X_train, y_train = extract_batch_features(train_boundaries)
    print(f"Extracting features for {len(val_boundaries)} val samples...")
    X_val, y_val = extract_batch_features(val_boundaries)
    print(f"Extracting features for {len(test_boundaries)} test samples...")
    X_test, y_test = extract_batch_features(test_boundaries)
    
    print("Training GBDT model...")
    clf = GBDTBoundaryClassifier(n_estimators=250, learning_rate=0.04, max_depth=6)
    clf.fit(X_train, y_train, X_val, y_val)
    
    print("\n--- Validation Set Performance ---")
    val_metrics = clf.evaluate(X_val, y_val)
    print(f"Accuracy:  {val_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1:  {val_metrics['macro_f1']*100:.2f}%")
    print(f"Join F1:   {val_metrics['join_f1']*100:.2f}%")
    print(f"Split F1:  {val_metrics['split_f1']*100:.2f}%")
    print(f"ROC-AUC:   {val_metrics['roc_auc']:.4f}")
    
    print("\n--- Test Set Performance (Held-out Unseen Documents) ---")
    test_metrics = clf.evaluate(X_test, y_test)
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1:  {test_metrics['macro_f1']*100:.2f}%")
    print(f"Join F1:   {test_metrics['join_f1']*100:.2f}%")
    print(f"Split F1:  {test_metrics['split_f1']*100:.2f}%")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix [ [TN, FP], [FN, TP] ]: {test_metrics['confusion_matrix']}")
    
    print("\nTop 8 Most Informative Features:")
    for i, (k, v) in enumerate(list(test_metrics['feature_importances'].items())[:8]):
        print(f"  {i+1}. {k}: {v:.1f}")
        
    clf.save(model_output_path)
    
    # Save evaluation report
    os.makedirs("results", exist_ok=True)
    with open("results/gbdt_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
        
    print(f"Metrics saved to results/gbdt_test_metrics.json")
    return test_metrics


if __name__ == '__main__':
    train_and_eval_gbdt()
