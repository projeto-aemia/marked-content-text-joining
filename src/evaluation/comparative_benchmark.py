"""
Comparative Benchmark Suite for Marked Content Text Joining
Evaluates 2 Main GBDT Strategies with 3 Substrategies Each:
  1. Language-Specific GBDT Strategy:
     - Substrategy 1: Specific GBDTs only (11 language-specific GBDTs)
     - Substrategy 2: Specific GBDTs + Language-Specific Cross-Encoders (11 language-specific cross-encoders)
     - Substrategy 3: Specific GBDTs + Universal Cross-Encoder (1 universal cross-encoder)
  2. Universal GBDT Strategy:
     - Substrategy 1: Universal GBDT only (single universal GBDT)
     - Substrategy 2: Universal GBDT + Language-Specific Cross-Encoders (11 language-specific cross-encoders)
     - Substrategy 3: Universal GBDT + Universal Cross-Encoder (1 universal cross-encoder)
Across all 11 target language test splits and the entire global test split.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.features.feature_builder import extract_batch_features
from src.models.gbdt_model import GBDTBoundaryClassifier
from src.models.cross_encoder import CrossEncoderClassifier
from src.models.cascade_model import CascadeBoundaryClassifier
from src.evaluation.benchmark import calculate_phrase_exact_match, evaluate_heuristic_baseline

TARGET_LANGUAGES = ['en', 'ru', 'es', 'tr', 'fr', 'cs', 'pt', 'de', 'pl', 'uk', 'id']

STRATEGY_KEYS = [
    "Heuristic Baseline",
    "Language-Specific GBDT: GBDT Only",
    "Language-Specific GBDT: + Language-Specific Cross-Encoders",
    "Language-Specific GBDT: + Universal Cross-Encoder",
    "Universal GBDT: GBDT Only",
    "Universal GBDT: + Language-Specific Cross-Encoders",
    "Universal GBDT: + Universal Cross-Encoder",
]


def load_test_data(data_dir: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Loads per-language test splits and the global test split."""
    per_lang_test = {}
    for lang in TARGET_LANGUAGES:
        fpath = os.path.join(data_dir, f"{lang}_test.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                per_lang_test[lang] = json.load(f)

    with open(os.path.join(data_dir, "global_test.json"), "r", encoding="utf-8") as f:
        global_test = json.load(f)

    return per_lang_test, global_test


def evaluate_model_on_boundaries(model, boundaries: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluates a GBDT, Cross-Encoder, or Cascade model on a list of boundaries."""
    if len(boundaries) == 0:
        return np.array([], dtype=np.int32), {
            "accuracy": 0.0, "macro_f1": 0.0, "join_f1": 0.0, "split_f1": 0.0,
            "phrase_em": 0.0, "latency_ms": 0.0, "samples": 0
        }

    y_true = np.array([int(b['label']) for b in boundaries], dtype=np.int32)

    t0 = time.perf_counter()
    if hasattr(model, "predict_with_routing"):
        preds, probs, stats = model.predict_with_routing(boundaries)
        latency = stats.get("avg_latency_per_pair_ms", 0.0)
    elif isinstance(model, CrossEncoderClassifier):
        preds = model.predict(boundaries, batch_size=64)
        latency = (time.perf_counter() - t0) * 1000.0 / len(boundaries)
    else:
        X, _ = extract_batch_features(boundaries)
        preds = model.predict(X)
        latency = (time.perf_counter() - t0) * 1000.0 / len(boundaries)

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average='macro'))
    join_f1 = float(f1_score(y_true, preds, pos_label=1))
    split_f1 = float(f1_score(y_true, preds, pos_label=0))
    em_res = calculate_phrase_exact_match(boundaries, preds)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "join_f1": join_f1,
        "split_f1": split_f1,
        "phrase_em": em_res["phrase_exact_match"],
        "latency_ms": latency,
        "samples": len(boundaries)
    }
    return preds, metrics


def compute_metrics_from_preds(boundaries: List[Dict[str, Any]], preds: np.ndarray, avg_latency: float = 0.0) -> Dict[str, Any]:
    """Computes full evaluation metrics given true boundaries and pooled predictions."""
    y_true = np.array([int(b['label']) for b in boundaries], dtype=np.int32)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average='macro'))
    join_f1 = float(f1_score(y_true, preds, pos_label=1))
    split_f1 = float(f1_score(y_true, preds, pos_label=0))
    em_res = calculate_phrase_exact_match(boundaries, preds)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "join_f1": join_f1,
        "split_f1": split_f1,
        "phrase_em": em_res["phrase_exact_match"],
        "latency_ms": avg_latency,
        "samples": len(boundaries)
    }


def run_comparative_benchmark(
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
    results_dir: str = os.path.join(PROJECT_ROOT, "results", "multilingual")
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    per_lang_test, global_test = load_test_data(data_dir)

    print("=" * 80)
    print("RUNNING MULTILINGUAL COMPARATIVE BENCHMARK")
    print("Evaluating 3 Strategies & 7 Substrategies Across 11 Languages & Global Test Set")
    print(f"Total test samples across all 11 languages: {len(global_test)}")
    for lang, b_list in per_lang_test.items():
        print(f"  [{lang:>2}] {len(b_list):>5} test boundaries")
    print("=" * 80)

    # 1. Load Universal GBDT
    univ_path = os.path.join(models_dir, "gbdt_universal.pkl")
    univ_gbdt = GBDTBoundaryClassifier.load(univ_path) if os.path.exists(univ_path) else None
    if univ_gbdt is None:
        raise FileNotFoundError(f"Universal GBDT model not found at {univ_path}")

    # 2. Load Language-Specific GBDTs
    specialized_gbdts = {}
    for lang in TARGET_LANGUAGES:
        l_path = os.path.join(models_dir, f"gbdt_{lang}.pkl")
        if os.path.exists(l_path):
            specialized_gbdts[lang] = GBDTBoundaryClassifier.load(l_path)
        else:
            print(f"Warning: GBDT for {lang} not found at {l_path}")

    # 3. Load Universal Cross-Encoder and Cascade
    univ_encoder_path = os.path.join(models_dir, "cross_encoder_universal")
    univ_encoder = None
    univ_cascade = None
    if os.path.exists(univ_encoder_path):
        print(f"Loading Universal Cross-Encoder from {univ_encoder_path}...")
        try:
            univ_encoder = CrossEncoderClassifier.load(univ_encoder_path)
            univ_cascade = CascadeBoundaryClassifier(
                gbdt_model=univ_gbdt,
                cross_encoder=univ_encoder,
                tau_low=0.10,
                tau_high=0.90
            )
            print("Universal 2-Stage Cascade initialized.")
        except Exception as e:
            print(f"Warning: Failed to load Universal Cross-Encoder: {e}")

    # Data structures to accumulate results
    matrix: Dict[str, Dict[str, Any]] = {k: {} for k in STRATEGY_KEYS}
    pooled_preds: Dict[str, List[int]] = {k: [] for k in STRATEGY_KEYS}
    latency_accum: Dict[str, List[float]] = {k: [] for k in STRATEGY_KEYS}

    # Evaluate per language
    for lang in TARGET_LANGUAGES:
        b_lang = per_lang_test[lang]
        y_lang = np.array([int(b['label']) for b in b_lang], dtype=np.int32)
        n_samples = len(b_lang)
        print(f"\nEvaluating language: [{lang.upper()}] ({n_samples} samples)...")

        # Load language-specific cross-encoder if available
        lang_encoder_path = os.path.join(models_dir, f"cross_encoder_{lang}")
        lang_encoder = None
        if os.path.exists(lang_encoder_path):
            try:
                lang_encoder = CrossEncoderClassifier.load(lang_encoder_path)
            except Exception as e:
                print(f"Warning: Could not load cross-encoder for {lang}: {e}")

        # Cascade models for this language
        gbdt_lang = specialized_gbdts.get(lang, univ_gbdt)

        casc_lang_lang = None
        if lang_encoder is not None:
            casc_lang_lang = CascadeBoundaryClassifier(
                gbdt_model=gbdt_lang,
                cross_encoder=lang_encoder,
                tau_low=0.10,
                tau_high=0.90
            )

        casc_lang_univ = None
        if univ_encoder is not None:
            casc_lang_univ = CascadeBoundaryClassifier(
                gbdt_model=gbdt_lang,
                cross_encoder=univ_encoder,
                tau_low=0.10,
                tau_high=0.90
            )

        casc_univ_lang = None
        if lang_encoder is not None:
            casc_univ_lang = CascadeBoundaryClassifier(
                gbdt_model=univ_gbdt,
                cross_encoder=lang_encoder,
                tau_low=0.10,
                tau_high=0.90
            )

        # 1. Heuristic Baseline
        h_preds = evaluate_heuristic_baseline(b_lang)
        h_metrics = compute_metrics_from_preds(b_lang, h_preds, avg_latency=0.001)
        matrix["Heuristic Baseline"][lang] = h_metrics
        pooled_preds["Heuristic Baseline"].extend(h_preds.tolist())
        latency_accum["Heuristic Baseline"].append(0.001)

        # 2. Language-Specific GBDT Strategy - Substrategy 1: GBDT Only
        p, m = evaluate_model_on_boundaries(gbdt_lang, b_lang)
        matrix["Language-Specific GBDT: GBDT Only"][lang] = m
        pooled_preds["Language-Specific GBDT: GBDT Only"].extend(p.tolist())
        latency_accum["Language-Specific GBDT: GBDT Only"].append(m["latency_ms"])

        # 3. Language-Specific GBDT Strategy - Substrategy 2: + Language-Specific Cross-Encoders
        if casc_lang_lang is not None:
            p_ll, m_ll = evaluate_model_on_boundaries(casc_lang_lang, b_lang)
            matrix["Language-Specific GBDT: + Language-Specific Cross-Encoders"][lang] = m_ll
            pooled_preds["Language-Specific GBDT: + Language-Specific Cross-Encoders"].extend(p_ll.tolist())
            latency_accum["Language-Specific GBDT: + Language-Specific Cross-Encoders"].append(m_ll["latency_ms"])
        else:
            matrix["Language-Specific GBDT: + Language-Specific Cross-Encoders"][lang] = m
            pooled_preds["Language-Specific GBDT: + Language-Specific Cross-Encoders"].extend(p.tolist())
            latency_accum["Language-Specific GBDT: + Language-Specific Cross-Encoders"].append(m["latency_ms"])

        # 4. Language-Specific GBDT Strategy - Substrategy 3: + Universal Cross-Encoder
        if casc_lang_univ is not None:
            p_lu, m_lu = evaluate_model_on_boundaries(casc_lang_univ, b_lang)
            matrix["Language-Specific GBDT: + Universal Cross-Encoder"][lang] = m_lu
            pooled_preds["Language-Specific GBDT: + Universal Cross-Encoder"].extend(p_lu.tolist())
            latency_accum["Language-Specific GBDT: + Universal Cross-Encoder"].append(m_lu["latency_ms"])
        else:
            matrix["Language-Specific GBDT: + Universal Cross-Encoder"][lang] = m
            pooled_preds["Language-Specific GBDT: + Universal Cross-Encoder"].extend(p.tolist())
            latency_accum["Language-Specific GBDT: + Universal Cross-Encoder"].append(m["latency_ms"])

        # 5. Universal GBDT Strategy - Substrategy 1: GBDT Only
        p_ug, m_ug = evaluate_model_on_boundaries(univ_gbdt, b_lang)
        matrix["Universal GBDT: GBDT Only"][lang] = m_ug
        pooled_preds["Universal GBDT: GBDT Only"].extend(p_ug.tolist())
        latency_accum["Universal GBDT: GBDT Only"].append(m_ug["latency_ms"])

        # 6. Universal GBDT Strategy - Substrategy 2: + Language-Specific Cross-Encoders
        if casc_univ_lang is not None:
            p_ul, m_ul = evaluate_model_on_boundaries(casc_univ_lang, b_lang)
            matrix["Universal GBDT: + Language-Specific Cross-Encoders"][lang] = m_ul
            pooled_preds["Universal GBDT: + Language-Specific Cross-Encoders"].extend(p_ul.tolist())
            latency_accum["Universal GBDT: + Language-Specific Cross-Encoders"].append(m_ul["latency_ms"])
        else:
            matrix["Universal GBDT: + Language-Specific Cross-Encoders"][lang] = m_ug
            pooled_preds["Universal GBDT: + Language-Specific Cross-Encoders"].extend(p_ug.tolist())
            latency_accum["Universal GBDT: + Language-Specific Cross-Encoders"].append(m_ug["latency_ms"])

        # 7. Universal GBDT Strategy - Substrategy 3: + Universal Cross-Encoder
        if univ_cascade is not None:
            p_uc, m_uc = evaluate_model_on_boundaries(univ_cascade, b_lang)
            matrix["Universal GBDT: + Universal Cross-Encoder"][lang] = m_uc
            pooled_preds["Universal GBDT: + Universal Cross-Encoder"].extend(p_uc.tolist())
            latency_accum["Universal GBDT: + Universal Cross-Encoder"].append(m_uc["latency_ms"])
        else:
            matrix["Universal GBDT: + Universal Cross-Encoder"][lang] = m_ug
            pooled_preds["Universal GBDT: + Universal Cross-Encoder"].extend(p_ug.tolist())
            latency_accum["Universal GBDT: + Universal Cross-Encoder"].append(m_ug["latency_ms"])

        # Free language cross-encoder memory
        if lang_encoder is not None:
            del lang_encoder
            del casc_lang_lang
            del casc_univ_lang
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute Global Test Split Metrics for every strategy/substrategy
    print("\nComputing Global Metrics across all 34,588 boundaries...")
    for strat_key in STRATEGY_KEYS:
        preds = np.array(pooled_preds[strat_key], dtype=np.int32)
        avg_lat = float(np.mean(latency_accum[strat_key])) if latency_accum[strat_key] else 0.0
        g_metrics = compute_metrics_from_preds(global_test, preds, avg_latency=avg_lat)
        matrix[strat_key]["GLOBAL"] = g_metrics

    # Print Macro F1 Matrix Table
    print("\n" + "=" * 169)
    print("COMPARATIVE BENCHMARK MATRIX (MACRO F1 SCORE %)")
    print("=" * 169)
    header = f"{'Strategy / Substrategy':<60} | " + " | ".join(f"{l.upper():>6}" for l in TARGET_LANGUAGES) + " |  GLOBAL"
    print(header)
    print("-" * len(header))

    for strat_name, cols in matrix.items():
        row_str = f"{strat_name:<60} | "
        for lang in TARGET_LANGUAGES:
            f1_val = cols.get(lang, {}).get("macro_f1", 0.0) * 100
            row_str += f"{f1_val:>5.1f}% | "
        glob_f1 = cols.get("GLOBAL", {}).get("macro_f1", 0.0) * 100
        row_str += f"{glob_f1:>6.1f}%"
        print(row_str)

    print("=" * 169)

    # Print Global Accuracy & Exact Match Table
    print("\n" + "=" * 107)
    print("GLOBAL PERFORMANCE SUMMARY (HELD-OUT TEST SET: 34,588 BOUNDARIES)")
    print("=" * 107)
    g_header = f"{'Strategy / Substrategy':<60} | {'Accuracy':>8} | {'Macro F1':>8} | {'Phrase EM':>9} | {'Latency':>10}"
    print(g_header)
    print("-" * len(g_header))

    for strat_name, cols in matrix.items():
        g = cols.get("GLOBAL", {})
        acc_str = f"{g.get('accuracy', 0.0)*100:>7.2f}%"
        f1_str = f"{g.get('macro_f1', 0.0)*100:>7.2f}%"
        em_str = f"{g.get('phrase_em', 0.0)*100:>8.2f}%"
        lat_str = f"{g.get('latency_ms', 0.0):>7.3f} ms"
        print(f"{strat_name:<60} | {acc_str} | {f1_str} | {em_str} | {lat_str}")

    print("=" * 107)

    # Save to JSON
    out_file = os.path.join(results_dir, "comparative_matrix.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nFull comparative matrix saved to {out_file}")

    return matrix


if __name__ == '__main__':
    run_comparative_benchmark()
