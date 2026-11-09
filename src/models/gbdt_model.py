"""
Tier 0: Tabular GBDT Classifier (LightGBM)
Fast, client-side boundary decision classifier using 28 geometric and surface lexical features.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.features.feature_builder import FEATURE_NAMES, extract_batch_features


class GBDTBoundaryClassifier:
    """LightGBM-based Marked Content boundary classifier."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            class_weight=class_weight,
            random_state=random_state,
            verbose=-1
        )
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fits the GBDT model."""
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None
        )
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary decision (1 = JOIN, 0 = SPLIT)."""
        if hasattr(self.model, "n_features_in_") and X.shape[1] > self.model.n_features_in_:
            X = X[:, :self.model.n_features_in_]
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities [P(SPLIT), P(JOIN)]."""
        if hasattr(self.model, "n_features_in_") and X.shape[1] > self.model.n_features_in_:
            X = X[:, :self.model.n_features_in_]
        return self.model.predict_proba(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculates comprehensive evaluation metrics."""
        preds = self.predict(X)
        probs = self.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        join_f1 = f1_score(y, preds, pos_label=1)
        split_f1 = f1_score(y, preds, pos_label=0)
        roc_auc = roc_auc_score(y, probs)
        cm = confusion_matrix(y, preds).tolist()
        
        importances = dict(zip(FEATURE_NAMES, [float(v) for v in self.model.feature_importances_]))
        # Sort by importance
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "join_f1": float(join_f1),
            "split_f1": float(split_f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "feature_importances": sorted_imp
        }
        
    def save(self, model_path: str):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
    @classmethod
    def load(cls, model_path: str) -> "GBDTBoundaryClassifier":
        """Loads model from disk."""
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.is_fitted = True
        return instance
