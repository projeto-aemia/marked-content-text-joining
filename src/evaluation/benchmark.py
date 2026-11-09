"""
Comprehensive Benchmarking Suite for Marked Content Text Joining
Evaluates all model tiers (Heuristic Rule, GBDT, Cross-Encoder, Cascade, and SLM)
on Boundary F1, Paragraph Exact Match, and Latency.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_builder import extract_batch_features
from src.models.gbdt_model import GBDTBoundaryClassifier
from src.models.cascade_model import CascadeBoundaryClassifier


def calculate_phrase_exact_match(
    boundaries: List[Dict[str, Any]],
    predictions: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates phrase/paragraph level reconstruction:
    Reassembles pieces into phrases based on predicted splits and compares with ground truth.
    """
    if len(boundaries) == 0:
        return {"phrase_exact_match": 0.0, "total_gold_phrases": 0, "total_pred_phrases": 0}
        
    gold_splits = [int(b['label']) == 0 for b in boundaries]
    pred_splits = [bool(p == 0) for p in predictions]
    
    # Reassemble gold phrases
    gold_phrases = []
    current_gold = [boundaries[0]['text_a']]
    for i, is_split in enumerate(gold_splits):
        if is_split:
            gold_phrases.append(" ".join(current_gold))
            current_gold = [boundaries[i]['text_b']]
        else:
            current_gold.append(boundaries[i]['text_b'])
    gold_phrases.append(" ".join(current_gold))
    
    # Reassemble predicted phrases
    pred_phrases = []
    current_pred = [boundaries[0]['text_a']]
    for i, is_split in enumerate(pred_splits):
        if is_split:
            pred_phrases.append(" ".join(current_pred))
            current_pred = [boundaries[i]['text_b']]
        else:
            current_pred.append(boundaries[i]['text_b'])
    pred_phrases.append(" ".join(current_pred))
    
    # Exact Match count
    exact_matches = sum(1 for p in pred_phrases if p in gold_phrases)
    em_rate = exact_matches / max(1, len(gold_phrases))
    
    return {
        "phrase_exact_match": float(em_rate),
        "total_gold_phrases": len(gold_phrases),
        "total_pred_phrases": len(pred_phrases),
        "exact_matched_phrases": exact_matches
    }


def evaluate_heuristic_baseline(boundaries: List[Dict[str, Any]]) -> np.ndarray:
    """
    Simple heuristic baseline:
    - Splits if text_a ends with '.', '?', '!' AND dy > 5.0 pt
    - Otherwise joins
    """
    preds = []
    for b in boundaries:
        txt_a = str(b.get('text_a', '')).strip()
        dy = float(b.get('bbox_b', [0, 0, 0, 0])[1]) - float(b.get('bbox_a', [0, 0, 0, 0])[3])
        same_page = b.get('page_a') == b.get('page_b')
        
        if not same_page and dy > 15.0:
            preds.append(0) # SPLIT across pages
        elif txt_a and txt_a[-1] in '.?!' and dy > 4.0:
            preds.append(0) # SPLIT
        elif txt_a.startswith(('•', '-', '*')) and len(txt_a) <= 2:
            preds.append(1) # JOIN bullet to content
        else:
            preds.append(1) # JOIN
            
    return np.array(preds, dtype=np.int32)


def run_benchmark(
    data_dir: str = "data",
    gbdt_path: str = "saved_models/gbdt_model.pkl",
    results_dir: str = "results"
):
    os.makedirs(results_dir, exist_ok=True)
    print("Loading test dataset...")
    with open(os.path.join(data_dir, "test_boundaries.json"), "r", encoding="utf-8") as f:
        test_boundaries = json.load(f)
        
    y_test = np.array([int(b['label']) for b in test_boundaries])
    print(f"Test samples: {len(test_boundaries)} (JOIN: {(y_test==1).sum()}, SPLIT: {(y_test==0).sum()})\n")
    
    leaderboard = []
    
    # 1. Evaluate Heuristic Baseline
    print("--> 1. Evaluating Heuristic Rule Baseline...")
    t0 = time.perf_counter()
    heur_preds = evaluate_heuristic_baseline(test_boundaries)
    heur_latency = (time.perf_counter() - t0) * 1000.0 / len(test_boundaries)
    
    heur_acc = float(np.mean(heur_preds == y_test))
    from sklearn.metrics import f1_score
    heur_macro_f1 = float(f1_score(y_test, heur_preds, average='macro'))
    heur_join_f1 = float(f1_score(y_test, heur_preds, pos_label=1))
    heur_split_f1 = float(f1_score(y_test, heur_preds, pos_label=0))
    heur_em = calculate_phrase_exact_match(test_boundaries, heur_preds)
    
    leaderboard.append({
        "model": "Heuristic Rules",
        "tier": "Baseline",
        "size_mb": 0.0,
        "accuracy": heur_acc,
        "macro_f1": heur_macro_f1,
        "join_f1": heur_join_f1,
        "split_f1": heur_split_f1,
        "phrase_em": heur_em["phrase_exact_match"],
        "latency_ms_per_pair": heur_latency
    })
    
    # 2. Evaluate Tier 0: GBDT
    print("--> 2. Evaluating Tier 0: GBDT (LightGBM)...")
    if os.path.exists(gbdt_path):
        gbdt = GBDTBoundaryClassifier.load(gbdt_path)
        X_test, _ = extract_batch_features(test_boundaries)
        
        t0 = time.perf_counter()
        gbdt_preds = gbdt.predict(X_test)
        gbdt_latency = (time.perf_counter() - t0) * 1000.0 / len(test_boundaries)
        
        gbdt_metrics = gbdt.evaluate(X_test, y_test)
        gbdt_em = calculate_phrase_exact_match(test_boundaries, gbdt_preds)
        gbdt_size = os.path.getsize(gbdt_path) / (1024 * 1024)
        
        leaderboard.append({
            "model": "LightGBM GBDT",
            "tier": "Tier 0",
            "size_mb": round(gbdt_size, 2),
            "accuracy": gbdt_metrics["accuracy"],
            "macro_f1": gbdt_metrics["macro_f1"],
            "join_f1": gbdt_metrics["join_f1"],
            "split_f1": gbdt_metrics["split_f1"],
            "phrase_em": gbdt_em["phrase_exact_match"],
            "latency_ms_per_pair": gbdt_latency
        })
        
        # 3. Evaluate Tier 2: Cascade (GBDT Gatekeeper)
        print("--> 3. Evaluating Tier 2: Cascade Pipeline (GBDT Gatekeeper)...")
        cascade = CascadeBoundaryClassifier(gbdt, tau_low=0.08, tau_high=0.92)
        casc_preds, casc_probs, casc_stats = cascade.predict_with_routing(test_boundaries)
        casc_acc = float(np.mean(casc_preds == y_test))
        casc_macro_f1 = float(f1_score(y_test, casc_preds, average='macro'))
        casc_join_f1 = float(f1_score(y_test, casc_preds, pos_label=1))
        casc_split_f1 = float(f1_score(y_test, casc_preds, pos_label=0))
        casc_em = calculate_phrase_exact_match(test_boundaries, casc_preds)
        
        leaderboard.append({
            "model": "2-Stage Cascade",
            "tier": "Tier 2",
            "size_mb": round(gbdt_size, 2),
            "accuracy": casc_acc,
            "macro_f1": casc_macro_f1,
            "join_f1": casc_join_f1,
            "split_f1": casc_split_f1,
            "phrase_em": casc_em["phrase_exact_match"],
            "latency_ms_per_pair": casc_stats["avg_latency_per_pair_ms"]
        })
        
    print("\n================== BENCHMARK LEADERBOARD ==================")
    print(f"{'Model':<18} | {'Tier':<8} | {'Size':<7} | {'Acc %':<7} | {'Macro F1':<8} | {'Join F1':<7} | {'Split F1':<8} | {'Phrase EM':<9} | {'Latency':<8}")
    print("-" * 96)
    for row in leaderboard:
        print(f"{row['model']:<18} | {row['tier']:<8} | {row['size_mb']:>5.2f}MB | {row['accuracy']*100:>5.2f}% | {row['macro_f1']*100:>6.2f}% | {row['join_f1']*100:>5.2f}% | {row['split_f1']*100:>6.2f}% | {row['phrase_em']*100:>7.2f}% | {row['latency_ms_per_pair']:>5.3f}ms")
    print("=" * 96)
    
    with open(os.path.join(results_dir, "benchmark_leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)
        
    return leaderboard


if __name__ == '__main__':
    run_benchmark()
