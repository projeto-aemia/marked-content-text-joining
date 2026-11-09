"""
Clean JSON Exporter for LightGBM GBDT Models
Extracts tree structures, strips training metadata, and formats nodes into a minimal,
zero-dependency JSON representation matching the TypeScript/JavaScript GbdtModelJson schema.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Union, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _clean_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively cleans LightGBM node dictionary to essential evaluation fields."""
    if "leaf_value" in node:
        return {"leaf_value": float(node["leaf_value"])}
    return {
        "split_feature": int(node["split_feature"]),
        "threshold": float(node["threshold"]),
        "left_child": _clean_node(node["left_child"]),
        "right_child": _clean_node(node["right_child"])
    }


def export_gbdt_to_clean_json(
    model_input: Union[str, Any],
    output_path: Optional[str] = None,
    name: str = "gbdt_universal",
    version: str = "1.0.0",
    minify: bool = True
) -> Dict[str, Any]:
    """
    Exports a trained GBDT model (file path, GBDTBoundaryClassifier, or LightGBM booster)
    into a clean, minimal JSON object matching the TypeScript GbdtModelJson interface.

    Args:
        model_input: File path to .pkl model or loaded model instance.
        output_path: Optional destination file path (.json). If None, model is returned without saving.
        name: Name tag for the model metadata.
        version: Semantic version string.
        minify: Whether to write single-line minified JSON or formatted indented JSON.

    Returns:
        Dict[str, Any]: The clean serialized model structure.
    """
    import joblib

    if isinstance(model_input, str):
        if not os.path.exists(model_input):
            raise FileNotFoundError(f"Model file not found at: {model_input}")
        loaded = joblib.load(model_input)
    else:
        loaded = model_input

    # Resolve underlying LightGBM booster and number of features
    if hasattr(loaded, "model") and hasattr(loaded.model, "booster_"):
        booster = loaded.model.booster_
        num_features = getattr(loaded.model, "n_features_in_", 30)
    elif hasattr(loaded, "booster_"):
        booster = loaded.booster_
        num_features = getattr(loaded, "n_features_in_", 30)
    elif hasattr(loaded, "dump_model"):
        booster = loaded
        num_features = 30
    else:
        raise ValueError(f"Unsupported model object type: {type(loaded)}")

    raw_dump = booster.dump_model()
    tree_info = raw_dump.get("tree_info", [])
    feature_names = raw_dump.get("feature_names", [f"Column_{i}" for i in range(num_features)])

    clean_dump: Dict[str, Any] = {
        "name": name,
        "version": version,
        "num_features": num_features,
        "feature_names": feature_names,
        "tree_info": [
            {"tree_structure": _clean_node(t["tree_structure"])}
            for t in tree_info
        ]
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if minify:
                json.dump(clean_dump, f, separators=(",", ":"))
            else:
                json.dump(clean_dump, f, indent=2)

        file_size_kb = os.path.getsize(output_path) / 1024.0
        print(f"Exported clean GBDT model JSON to {output_path}")
        print(f"  Trees: {len(tree_info)} | Features: {num_features} | Size: {file_size_kb:.1f} KB")

    return clean_dump


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained LightGBM GBDT model to clean client-side JSON."
    )
    default_input = os.path.join(PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.pkl")
    default_output = os.path.join(PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.json")

    parser.add_argument(
        "-i", "--input", "--model-path",
        dest="input_path",
        default=default_input,
        help=f"Path to trained GBDT model pickle file (default: {default_input})"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        default=default_output,
        help=f"Path to output JSON file (default: {default_output})"
    )
    parser.add_argument(
        "--name",
        default="gbdt_universal",
        help="Model identifier name in metadata header (default: 'gbdt_universal')"
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Model semantic version (default: '1.0.0')"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format output JSON with indentation instead of minifying"
    )

    args = parser.parse_args()

    export_gbdt_to_clean_json(
        model_input=args.input_path,
        output_path=args.output_path,
        name=args.name,
        version=args.version,
        minify=not args.pretty
    )


if __name__ == '__main__':
    main()
