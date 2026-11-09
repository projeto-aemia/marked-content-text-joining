"""
Multilingual Model Training Suite for Marked Content Text Joining
Trains:
  1. Language-Specific Strategy: 11 Language-Specific GBDTs (LightGBM)
  2. Universal Strategy: 1 Universal Multilingual GBDT (LightGBM)
  3. Universal Strategy: 1 Universal Multilingual Cross-Encoder (MiniLM-L12 on CUDA)
  4. Language-Specific Strategy: 11 Language-Specific Pretrained Cross-Encoders on CUDA
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.feature_builder import extract_batch_features
from src.models.gbdt_model import GBDTBoundaryClassifier
from src.models.cross_encoder import CrossEncoderClassifier
from src.models.registry import LANGUAGE_SPECIFIC_ENCODERS, UNIVERSAL_MULTILINGUAL_ENCODER

TARGET_LANGUAGES = ['en', 'ru', 'es', 'tr', 'fr', 'cs', 'pt', 'de', 'pl', 'uk', 'id']


def train_single_gbdt(
    lang: str,
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual")
) -> Dict[str, Any]:
    """Trains a language-specific LightGBM model."""
    print(f"\n--- [Strategy A] Training GBDT for {lang.upper()} ---")
    train_file = os.path.join(data_dir, f"{lang}_train.json")
    val_file = os.path.join(data_dir, f"{lang}_val.json")
    test_file = os.path.join(data_dir, f"{lang}_test.json")

    with open(train_file, "r", encoding="utf-8") as f:
        train_b = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_b = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_b = json.load(f)

    X_train, y_train = extract_batch_features(train_b)
    X_val, y_val = extract_batch_features(val_b)
    X_test, y_test = extract_batch_features(test_b)

    clf = GBDTBoundaryClassifier(n_estimators=250, learning_rate=0.04, max_depth=6)
    clf.fit(X_train, y_train, X_val, y_val)

    test_metrics = clf.evaluate(X_test, y_test)
    print(f"[{lang.upper()}] Test Acc: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']*100:.2f}% | Join F1: {test_metrics['join_f1']*100:.2f}% | Split F1: {test_metrics['split_f1']*100:.2f}%")

    model_path = os.path.join(models_dir, f"gbdt_{lang}.pkl")
    clf.save(model_path)
    return test_metrics


def train_universal_gbdt(
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual")
) -> Dict[str, Any]:
    """Trains a single unified LightGBM model on pooled data from all 11 languages."""
    print("\n--- [Strategy B] Training Universal Multilingual GBDT ---")
    train_file = os.path.join(data_dir, "global_train.json")
    val_file = os.path.join(data_dir, "global_val.json")
    test_file = os.path.join(data_dir, "global_test.json")

    with open(train_file, "r", encoding="utf-8") as f:
        train_b = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_b = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_b = json.load(f)

    print(f"Extracting features: {len(train_b)} train, {len(val_b)} val, {len(test_b)} test...")
    X_train, y_train = extract_batch_features(train_b)
    X_val, y_val = extract_batch_features(val_b)
    X_test, y_test = extract_batch_features(test_b)

    clf = GBDTBoundaryClassifier(n_estimators=300, learning_rate=0.04, max_depth=7)
    clf.fit(X_train, y_train, X_val, y_val)

    test_metrics = clf.evaluate(X_test, y_test)
    print(f"[UNIVERSAL GBDT] Test Acc: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']*100:.2f}% | Join F1: {test_metrics['join_f1']*100:.2f}% | Split F1: {test_metrics['split_f1']*100:.2f}%")

    model_path = os.path.join(models_dir, "gbdt_universal.pkl")
    clf.save(model_path)
    return test_metrics


def train_universal_cross_encoder(
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    max_samples: int = 25000
) -> Dict[str, Any]:
    """Fine-tunes the universal multilingual transformer cross-encoder on GPU."""
    print("\n--- [Strategy B] Fine-Tuning Universal Multilingual Cross-Encoder ---")
    hf_model = UNIVERSAL_MULTILINGUAL_ENCODER["hf_model"]
    output_dir = os.path.join(models_dir, "cross_encoder_universal")

    with open(os.path.join(data_dir, "global_train.json"), "r", encoding="utf-8") as f:
        train_b = json.load(f)
    with open(os.path.join(data_dir, "global_val.json"), "r", encoding="utf-8") as f:
        val_b = json.load(f)
    with open(os.path.join(data_dir, "global_test.json"), "r", encoding="utf-8") as f:
        test_b = json.load(f)

    if len(train_b) > max_samples:
        import random
        random.seed(42)
        random.shuffle(train_b)
        train_b = train_b[:max_samples]
        print(f"Sampled {max_samples} balanced boundaries for cross-encoder training.")

    clf = CrossEncoderClassifier(model_name=hf_model, max_length=128)
    clf.fit(
        train_boundaries=train_b,
        val_boundaries=val_b[:2000],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        include_layout_cues=True
    )

    test_metrics = clf.evaluate(test_b[:2000], batch_size=batch_size, include_layout_cues=True)
    print(f"\n[UNIVERSAL ENCODER] Test Acc: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']*100:.2f}%")

    clf.save(output_dir)
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_metrics


def train_single_cross_encoder(
    lang: str,
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_samples: int = 4000
) -> Dict[str, Any]:
    """Fine-tunes a language-specific transformer cross-encoder on GPU."""
    print(f"\n--- [Language-Specific Strategy] Fine-Tuning Cross-Encoder for {lang.upper()} ---")
    if lang not in LANGUAGE_SPECIFIC_ENCODERS:
        raise ValueError(f"Unknown language: {lang}")

    cfg = LANGUAGE_SPECIFIC_ENCODERS[lang]
    hf_model = cfg["hf_model"]
    output_dir = os.path.join(models_dir, f"cross_encoder_{lang}")

    with open(os.path.join(data_dir, f"{lang}_train.json"), "r", encoding="utf-8") as f:
        train_b = json.load(f)
    with open(os.path.join(data_dir, f"{lang}_val.json"), "r", encoding="utf-8") as f:
        val_b = json.load(f)
    with open(os.path.join(data_dir, f"{lang}_test.json"), "r", encoding="utf-8") as f:
        test_b = json.load(f)

    if len(train_b) > max_samples:
        import random
        random.seed(42)
        random.shuffle(train_b)
        train_b = train_b[:max_samples]
        print(f"Sampled {max_samples} balanced boundaries for {lang.upper()} cross-encoder training.")

    clf = CrossEncoderClassifier(model_name=hf_model, max_length=128)
    clf.fit(
        train_boundaries=train_b,
        val_boundaries=val_b[:1000] if len(val_b) > 1000 else val_b,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        include_layout_cues=True
    )

    eval_test = test_b[:1000] if len(test_b) > 1000 else test_b
    test_metrics = clf.evaluate(eval_test, batch_size=batch_size, include_layout_cues=True)
    print(f"\n[{lang.upper()} ENCODER] Test Acc: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']*100:.2f}%")

    clf.save(output_dir)

    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_metrics


def train_all_language_cross_encoders(
    data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    models_dir: str = os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
    languages: Optional[List[str]] = None,
    epochs: int = 2,
    batch_size: int = 16,
    max_samples: int = 4000,
    skip_existing: bool = True
) -> Dict[str, Any]:
    langs = languages or TARGET_LANGUAGES
    results = {}
    for lang in langs:
        out_dir = os.path.join(models_dir, f"cross_encoder_{lang}")
        if skip_existing and os.path.exists(out_dir) and os.path.exists(os.path.join(out_dir, "config.json")):
            print(f"Model for {lang.upper()} already exists in {out_dir}, skipping...")
            continue
        try:
            m = train_single_cross_encoder(
                lang=lang,
                data_dir=data_dir,
                models_dir=models_dir,
                epochs=epochs,
                batch_size=batch_size,
                max_samples=max_samples
            )
            results[lang] = m
        except Exception as e:
            print(f"Error training cross-encoder for {lang}: {e}")
            results[lang] = {"error": str(e)}
    return results


def train_all(data_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "multilingual"), exist_ok=True)
    results = {"gbdt_language_specific": {}, "gbdt_universal": {}, "encoder_language_specific": {}}

    # 1. Train 11 language-specific GBDTs
    for lang in TARGET_LANGUAGES:
        metrics = train_single_gbdt(lang, data_dir=data_dir)
        results["gbdt_language_specific"][lang] = metrics

    # 2. Train 1 Universal GBDT
    results["gbdt_universal"] = train_universal_gbdt(data_dir=data_dir)

    # 3. Train Universal Cross-Encoder
    try:
        results["encoder_universal"] = train_universal_cross_encoder(data_dir=data_dir)
    except Exception as e:
        print(f"Warning: Universal cross-encoder training error: {e}")
        results["encoder_universal"] = {"error": str(e)}

    # 4. Train 11 Language-Specific Cross-Encoders
    results["encoder_language_specific"] = train_all_language_cross_encoders(data_dir=data_dir)

    with open(os.path.join(PROJECT_ROOT, "results", "multilingual", "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAll model training complete! Summary saved to results/multilingual/training_summary.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbdts_only", action="store_true")
    parser.add_argument("--universal_encoder", action="store_true")
    parser.add_argument("--lang_encoders", action="store_true", help="Train all 11 language-specific cross-encoders")
    parser.add_argument("--single_lang_encoder", type=str, help="Train cross-encoder for a specific language (e.g. pt, cs, en)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--no_skip_existing", action="store_true")
    args = parser.parse_args()

    if args.single_lang_encoder:
        train_single_cross_encoder(
            lang=args.single_lang_encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
    elif args.lang_encoders:
        train_all_language_cross_encoders(
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            skip_existing=not args.no_skip_existing
        )
    elif args.gbdts_only:
        for lang in TARGET_LANGUAGES:
            train_single_gbdt(lang)
        train_universal_gbdt()
    elif args.universal_encoder:
        train_universal_cross_encoder()
    else:
        train_all()
