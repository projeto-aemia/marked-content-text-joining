"""
Training Script for Tier 1: Micro Cross-Encoder
Fine-tunes a lightweight transformer sequence classifier for boundary decision.
"""

import os
import sys
import json
import argparse

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.cross_encoder import CrossEncoderClassifier


def train_encoder(
    model_name: str = "prajjwal1/bert-tiny",
    data_dir: str = "data",
    output_dir: str = "saved_models/cross_encoder",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_length: int = 128
):
    print(f"Loading data from {data_dir}...")
    with open(os.path.join(data_dir, "train_boundaries.json"), "r", encoding="utf-8") as f:
        train_boundaries = json.load(f)
    with open(os.path.join(data_dir, "val_boundaries.json"), "r", encoding="utf-8") as f:
        val_boundaries = json.load(f)
    with open(os.path.join(data_dir, "test_boundaries.json"), "r", encoding="utf-8") as f:
        test_boundaries = json.load(f)
        
    print(f"Initializing CrossEncoder: {model_name}...")
    clf = CrossEncoderClassifier(model_name=model_name, max_length=max_length)
    
    clf.fit(
        train_boundaries=train_boundaries,
        val_boundaries=val_boundaries,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        include_layout_cues=True
    )
    
    print("\n--- Test Set Evaluation ---")
    test_metrics = clf.evaluate(test_boundaries, batch_size=batch_size, include_layout_cues=True)
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Macro F1:  {test_metrics['macro_f1']*100:.2f}%")
    print(f"Test Join F1:   {test_metrics['join_f1']*100:.2f}%")
    print(f"Test Split F1:  {test_metrics['split_f1']*100:.2f}%")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    clf.save(output_dir)
    
    with open("results/cross_encoder_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
        
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_encoder(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)
