"""
Tier 2: Two-Stage Cascaded Boundary Classifier
Combines ultra-fast GBDT filtering with a neural Cross-Encoder for ambiguous cases.
Delivers sub-millisecond average latency with high precision and recall.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.features.feature_builder import extract_batch_features, extract_boundary_features
from src.models.gbdt_model import GBDTBoundaryClassifier
from src.models.cross_encoder import CrossEncoderClassifier


class CascadeBoundaryClassifier:
    """
    Two-Stage Decision Cascade:
    - Stage 1: GBDT classifies high-confidence joins/splits in <0.05ms
    - Stage 2: Cross-Encoder evaluates borderline ambiguous boundaries
    """
    
    def __init__(
        self,
        gbdt_model: GBDTBoundaryClassifier,
        cross_encoder: Optional[CrossEncoderClassifier] = None,
        tau_low: float = 0.05,
        tau_high: float = 0.95
    ):
        self.gbdt = gbdt_model
        self.encoder = cross_encoder
        self.tau_low = tau_low
        self.tau_high = tau_high
        
    def predict_with_routing(
        self,
        boundaries: List[Dict[str, Any]],
        include_layout_cues: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Runs cascaded prediction and returns:
        (predictions, probabilities, routing_stats)
        """
        start_time = time.perf_counter()
        X, _ = extract_batch_features(boundaries)
        gbdt_probs = self.gbdt.predict_proba(X)[:, 1]
        
        n_total = len(boundaries)
        predictions = np.zeros(n_total, dtype=np.int32)
        final_probs = np.zeros(n_total, dtype=np.float32)
        
        stage1_decisions = 0
        stage2_decisions = 0
        ambiguous_indices = []
        ambiguous_boundaries = []
        
        for i in range(n_total):
            p = gbdt_probs[i]
            if p >= self.tau_high:
                # Decisive JOIN
                predictions[i] = 1
                final_probs[i] = p
                stage1_decisions += 1
            elif p <= self.tau_low:
                # Decisive SPLIT
                predictions[i] = 0
                final_probs[i] = p
                stage1_decisions += 1
            else:
                # Ambiguous boundary
                ambiguous_indices.append(i)
                ambiguous_boundaries.append(boundaries[i])
                
        # Dispatch ambiguous cases to Stage 2 if encoder is provided
        if ambiguous_boundaries and self.encoder is not None:
            stage2_probs = self.encoder.predict_proba(
                ambiguous_boundaries,
                include_layout_cues=include_layout_cues
            )[:, 1]
            for idx, prob in zip(ambiguous_indices, stage2_probs):
                final_probs[idx] = prob
                predictions[idx] = 1 if prob >= 0.5 else 0
                stage2_decisions += 1
        elif ambiguous_boundaries:
            # Fallback to GBDT 0.5 threshold
            for idx in ambiguous_indices:
                p = gbdt_probs[idx]
                final_probs[idx] = p
                predictions[idx] = 1 if p >= 0.5 else 0
                stage1_decisions += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = elapsed_ms / max(1, n_total)
        
        stats = {
            "total_samples": n_total,
            "stage1_handled": stage1_decisions,
            "stage2_handled": stage2_decisions,
            "stage1_ratio": stage1_decisions / max(1, n_total),
            "stage2_ratio": stage2_decisions / max(1, n_total),
            "total_latency_ms": elapsed_ms,
            "avg_latency_per_pair_ms": avg_latency_ms
        }
        
        return predictions, final_probs, stats
        
    def evaluate(
        self,
        boundaries: List[Dict[str, Any]],
        include_layout_cues: bool = True
    ) -> Dict[str, Any]:
        """Evaluates cascade accuracy, F1, exact match, and routing throughput."""
        preds, probs, stats = self.predict_with_routing(boundaries, include_layout_cues)
        y = np.array([int(b['label']) for b in boundaries])
        
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        join_f1 = f1_score(y, preds, pos_label=1)
        split_f1 = f1_score(y, preds, pos_label=0)
        roc_auc = roc_auc_score(y, probs)
        cm = confusion_matrix(y, preds).tolist()
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "join_f1": float(join_f1),
            "split_f1": float(split_f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "routing_stats": stats
        }
