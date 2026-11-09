#!/usr/bin/env python3
"""
Marked Content Text Joining: Unified CLI & Orchestrator

Provides a single command-line interface for:
  - comparative benchmarking across strategies (Universal GBDT vs. Language-Specific GBDTs vs. Cross-Encoders)
  - model training (GBDTs, Universal Cross-Encoder, Language-Specific Encoders)
  - model export (clean client-side JSON for zero-dependency browser execution & INT8 ONNX)
  - multilingual dataset construction and alignment
  - interactive single-pair boundary prediction
"""

import os
import sys
import json
import argparse
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TARGET_LANGUAGES = ['en', 'ru', 'es', 'tr', 'fr', 'cs', 'pt', 'de', 'pl', 'uk', 'id']


def cmd_benchmark(args):
    """Executes the comparative benchmark suite across all strategies and languages."""
    from src.evaluation.comparative_benchmark import run_comparative_benchmark

    print("\n" + "=" * 80)
    print("EXECUTING COMPARATIVE BENCHMARK SUITE")
    print(f"Data directory:   {args.data_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Results output:   {args.results_dir}")
    print("=" * 80 + "\n")

    run_comparative_benchmark(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )


def cmd_train(args):
    """Executes model training for GBDTs or Transformer Cross-Encoders."""
    from src.training.train_multilingual import (
        train_single_gbdt,
        train_universal_gbdt,
        train_universal_cross_encoder,
        train_single_cross_encoder,
        train_all_language_cross_encoders,
        train_all
    )

    strategy = args.strategy

    if strategy == "universal-gbdt":
        print("Training Universal GBDT...")
        train_universal_gbdt(data_dir=args.data_dir, models_dir=args.models_dir)
    elif strategy == "lang-gbdts":
        langs = [args.lang] if args.lang else TARGET_LANGUAGES
        for lang in langs:
            train_single_gbdt(lang=lang, data_dir=args.data_dir, models_dir=args.models_dir)
    elif strategy == "universal-encoder":
        print("Training Universal Cross-Encoder...")
        train_universal_cross_encoder(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples
        )
    elif strategy == "lang-encoders":
        if args.lang:
            train_single_cross_encoder(
                lang=args.lang,
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples
            )
        else:
            train_all_language_cross_encoders(
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                languages=TARGET_LANGUAGES,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                skip_existing=not args.force
            )
    elif strategy == "all":
        train_all(data_dir=args.data_dir)
    else:
        print(f"Unknown strategy: {strategy}")
        sys.exit(1)


def cmd_export(args):
    """Exports trained models to clean client-side JSON or ONNX format."""
    from src.export.export_json import export_gbdt_to_clean_json
    from src.export.export_onnx import export_gbdt_to_web, export_cross_encoder_to_onnx

    target = args.target

    if target in ("json", "all"):
        model_path = args.model or os.path.join(
            PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.pkl"
        )
        output_path = args.output
        if output_path is None or os.path.isdir(output_path):
            out_dir = output_path or os.path.join(PROJECT_ROOT, "saved_models", "multilingual")
            output_path = os.path.join(out_dir, f"{args.name}.json")

        print(f"\n[Export] Converting GBDT ({model_path}) to clean JSON...")
        export_gbdt_to_clean_json(
            model_input=model_path,
            output_path=output_path,
            name=args.name,
            version=args.version,
            minify=not args.pretty
        )

    if target in ("onnx", "all"):
        encoder_path = args.model
        if encoder_path is None:
            local_univ = os.path.join(PROJECT_ROOT, "saved_models", "multilingual", "cross_encoder_universal")
            encoder_path = local_univ if os.path.exists(local_univ) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        out_dir = args.output if (args.output and os.path.isdir(args.output)) else os.path.join(PROJECT_ROOT, "src", "export", "web_demo")
        print(f"\n[Export] Converting Cross-Encoder ({encoder_path}) to ONNX in {out_dir}...")
        export_cross_encoder_to_onnx(
            model_name=encoder_path,
            output_dir=out_dir,
            quantize_int8=not args.no_quantize
        )

    if target == "web-demo":
        out_dir = args.output or os.path.join(PROJECT_ROOT, "src", "export", "web_demo")
        print(f"\n[Export] Generating Web Demo assets into {out_dir}...")
        export_gbdt_to_web(gbdt_path=args.model, output_dir=out_dir, name=args.name)
        export_cross_encoder_to_onnx(
            model_name=args.model,
            output_dir=out_dir,
            quantize_int8=not args.no_quantize
        )


def cmd_build_dataset(args):
    """Builds document-level leak-free Train, Val, Test datasets across languages."""
    from src.training.build_multilingual_dataset import build_multilingual_dataset

    langs = args.languages.split(",") if args.languages else TARGET_LANGUAGES

    build_multilingual_dataset(
        manifest_dir=args.manifest_dir,
        docx_dir=args.docx_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        train_docs_per_lang=args.train_docs,
        val_docs_per_lang=args.val_docs,
        test_docs_per_lang=args.test_docs,
        seed=args.seed
    )


def cmd_manifest(args):
    """Generates per-language manifests by scanning PDF and DOCX directories."""
    from src.extraction.language_manifest import generate_language_manifests

    langs = args.languages.split(",") if args.languages else TARGET_LANGUAGES
    generate_language_manifests(
        docx_dir=args.docx_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        target_languages=langs,
        types=args.types,
        topics=args.topics,
        all_types=getattr(args, 'all_types', False) or getattr(args, 'all', False),
        all_topics=getattr(args, 'all_topics', False) or getattr(args, 'all', False)
    )


def cmd_align(args):
    """Aligns a single DOCX and PDF document pair and displays boundary labels."""
    from src.extraction.aligner import align_pdf_docx

    if not os.path.exists(args.docx):
        print(f"Error: DOCX not found at {args.docx}")
        sys.exit(1)
    if not os.path.exists(args.pdf):
        print(f"Error: PDF not found at {args.pdf}")
        sys.exit(1)

    print(f"Aligning:\n  DOCX: {args.docx}\n  PDF:  {args.pdf}\n")
    boundaries = align_pdf_docx(args.docx, args.pdf)
    joins = sum(1 for b in boundaries if b.get('label') == 1)
    splits = sum(1 for b in boundaries if b.get('label') == 0)

    print(f"Alignment completed: {len(boundaries)} total boundaries extracted.")
    print(f"  JOIN  (1): {joins} ({joins / len(boundaries) * 100:.1f}%)" if boundaries else "")
    print(f"  SPLIT (0): {splits} ({splits / len(boundaries) * 100:.1f}%)" if boundaries else "")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(boundaries, f, indent=2)
        print(f"Saved boundaries to {args.output}")


def cmd_predict(args):
    """Predicts JOIN or SPLIT for two adjacent text pieces."""
    from src.features.feature_builder import extract_boundary_features
    from src.models.gbdt_model import GBDTBoundaryClassifier

    model_path = args.model or os.path.join(
        PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.pkl"
    )
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, "saved_models", "gbdt_model.pkl")

    if not os.path.exists(model_path):
        print(f"Error: GBDT model not found at {model_path}")
        sys.exit(1)

    clf = GBDTBoundaryClassifier.load(model_path)

    boundary = {
        'page_a': 0, 'page_b': 0,
        'bbox_a': [100.0, 100.0, 300.0, 115.0],
        'bbox_b': [100.0 + args.dx, 115.0 + args.dy, 300.0 + args.dx, 130.0 + args.dy],
        'font_size_a': args.font_size, 'font_size_b': args.font_size,
        'font_a': 'Arial', 'font_b': 'Arial',
        'is_bold_a': False, 'is_bold_b': False,
        'text_a': args.text_a,
        'text_b': args.text_b
    }

    vec = extract_boundary_features(boundary).reshape(1, -1)
    prob_join = float(clf.predict_proba(vec)[0, 1])
    decision = "JOIN" if prob_join >= 0.5 else "SPLIT"

    print("\n" + "=" * 60)
    print("BOUNDARY PREDICTION RESULT")
    print("=" * 60)
    print(f"Piece A:  \"{args.text_a}\"")
    print(f"Piece B:  \"{args.text_b}\"")
    print(f"Decision: {decision}")
    print(f"P(JOIN):  {prob_join * 100:.2f}%  |  P(SPLIT): {(1 - prob_join) * 100:.2f}%")
    print(f"Model:    {model_path}")
    print("=" * 60 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-joining",
        description="Marked Content Text Joining: Multilingual Boundary Classification CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # 1. Benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run comparative benchmark matrix across strategies and 11 languages"
    )
    bench_parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "multilingual"),
        help="Path to boundary data directory"
    )
    bench_parser.add_argument(
        "--models-dir",
        default=os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
        help="Path to saved models directory"
    )
    bench_parser.add_argument(
        "--results-dir",
        default=os.path.join(PROJECT_ROOT, "results", "multilingual"),
        help="Path to save benchmark JSON matrix"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # 2. Train
    train_parser = subparsers.add_parser("train", help="Train GBDT or neural Cross-Encoder models")
    train_parser.add_argument(
        "--strategy",
        choices=["universal-gbdt", "lang-gbdts", "universal-encoder", "lang-encoders", "all"],
        default="universal-gbdt",
        help="Training target strategy (default: universal-gbdt)"
    )
    train_parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Specific language code (e.g., 'pt', 'es', 'cs', 'en') for single model training"
    )
    train_parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "multilingual"),
        help="Path to multilingual boundary dataset"
    )
    train_parser.add_argument(
        "--models-dir",
        default=os.path.join(PROJECT_ROOT, "saved_models", "multilingual"),
        help="Output directory to save trained models"
    )
    train_parser.add_argument("--epochs", type=int, default=2, help="Epochs for transformer training")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for neural training")
    train_parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for transformers")
    train_parser.add_argument("--max-samples", type=int, default=25000, help="Max boundaries for encoder training")
    train_parser.add_argument("--force", action="store_true", help="Overwrite existing model directories")
    train_parser.set_defaults(func=cmd_train)

    # 3. Export
    export_parser = subparsers.add_parser("export", help="Export models for client-side or ONNX execution")
    export_parser.add_argument(
        "--target",
        choices=["json", "onnx", "web-demo", "all"],
        default="json",
        help="Export target format (default: json)"
    )
    export_parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to input model (.pkl file for GBDT, or path/name for Cross-Encoder)"
    )
    export_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output destination path (file for JSON, directory for ONNX/web-demo)"
    )
    export_parser.add_argument(
        "--name",
        default="gbdt_universal",
        help="Model metadata name identifier (default: 'gbdt_universal')"
    )
    export_parser.add_argument(
        "--version",
        default="1.0.0",
        help="Model semantic version string (default: '1.0.0')"
    )
    export_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format JSON with indentation instead of minifying"
    )
    export_parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable dynamic INT8 quantization for ONNX export"
    )
    export_parser.set_defaults(func=cmd_export)

    # 4. Build Dataset
    ds_parser = subparsers.add_parser(
        "build-dataset",
        help="Build document-level leak-free Train/Val/Test boundary splits"
    )
    ds_parser.add_argument(
        "--manifest-dir",
        default=os.path.join(PROJECT_ROOT, "manifests"),
        help="Path to language manifest files"
    )
    ds_parser.add_argument(
        "--docx-dir",
        default=r"f:\docx-corpus-educational-education",
        help="Path to source DOCX documents"
    )
    ds_parser.add_argument(
        "--pdf-dir",
        default=r"f:\docx-corpus-educational-education\PDFs",
        help="Path to converted PDF documents"
    )
    ds_parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "multilingual"),
        help="Output directory for generated datasets"
    )
    ds_parser.add_argument("--train-docs", type=int, default=60, help="Train docs per language")
    ds_parser.add_argument("--val-docs", type=int, default=10, help="Val docs per language")
    ds_parser.add_argument("--test-docs", type=int, default=15, help="Test docs per language")
    ds_parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    ds_parser.add_argument("--languages", type=str, default=None, help="Comma-separated language codes")
    ds_parser.set_defaults(func=cmd_build_dataset)

    # 5. Manifest
    man_parser = subparsers.add_parser(
        "manifest",
        help="Scan and generate per-language manifest lists from corpus"
    )
    man_parser.add_argument(
        "--docx-dir",
        default=r"f:\docx-corpus-educational-education",
        help="Path to source DOCX documents"
    )
    man_parser.add_argument(
        "--pdf-dir",
        default=r"f:\docx-corpus-educational-education\PDFs",
        help="Path to converted PDF documents"
    )
    man_parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "manifests"),
        help="Output directory for manifests"
    )
    man_parser.add_argument("--languages", type=str, default=None, help="Comma-separated language codes")
    man_parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="List of document types to filter (space or comma-separated, e.g. 'educational', 'all' for no filter). Default: educational"
    )
    man_parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="List of document topics to filter (space or comma-separated, e.g. 'education', 'all' for no filter). Default: education"
    )
    man_parser.add_argument(
        "--all-types",
        action="store_true",
        help="Include all document types without filtering"
    )
    man_parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Include all document topics without filtering"
    )
    man_parser.add_argument(
        "--all",
        action="store_true",
        help="Include all document types and topics (entire corpus) without filtering"
    )
    man_parser.set_defaults(func=cmd_manifest)

    # 6. Align Single Pair
    align_parser = subparsers.add_parser(
        "align",
        help="Align a single DOCX and PDF document pair"
    )
    align_parser.add_argument("--docx", required=True, help="Path to input DOCX file")
    align_parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    align_parser.add_argument("--output", "-o", default=None, help="Optional output JSON file")
    align_parser.set_defaults(func=cmd_align)

    # 7. Predict
    pred_parser = subparsers.add_parser(
        "predict",
        help="Predict JOIN or SPLIT between two text pieces"
    )
    pred_parser.add_argument("--text-a", required=True, help="Preceding text piece")
    pred_parser.add_argument("--text-b", required=True, help="Succeeding text piece")
    pred_parser.add_argument("--model", default=None, help="Path to GBDT model pickle (.pkl)")
    pred_parser.add_argument("--dy", type=float, default=2.0, help="Vertical gap in points")
    pred_parser.add_argument("--dx", type=float, default=0.0, help="Horizontal gap in points")
    pred_parser.add_argument("--font-size", type=float, default=12.0, help="Font size in points")
    pred_parser.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
