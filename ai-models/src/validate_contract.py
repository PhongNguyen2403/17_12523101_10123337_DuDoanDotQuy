"""Validate the model, schema, metadata, and training feature contract."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from preprocess import ALL_FEATURES, TARGET

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCHEMA_PATH = MODELS_DIR / "schema.json"
METADATA_PATH = MODELS_DIR / "metadata.json"


def main() -> None:
    for path in (MODEL_PATH, SCHEMA_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    schema_features = tuple(feature["name"] for feature in schema["features"])
    if schema_features != tuple(ALL_FEATURES):
        raise AssertionError(f"Schema features do not match training features: {schema_features}")
    if schema.get("target") != TARGET:
        raise AssertionError("Schema target does not match training target.")
    if metadata.get("target") != TARGET or metadata.get("positive_class") != 1:
        raise AssertionError("Metadata target or positive_class is inconsistent.")

    model = joblib.load(MODEL_PATH)
    steps = getattr(model, "named_steps", {})
    if set(steps) != {"preprocess", "clf"}:
        raise AssertionError("model.joblib must contain preprocess and clf pipeline steps.")
    if not hasattr(model, "predict_proba"):
        raise AssertionError("model.joblib must expose predict_proba().")

    actual_versions = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    expected_versions = metadata.get("library_versions", {})
    mismatches = {
        name: (expected_versions.get(name), version)
        for name, version in actual_versions.items()
        if expected_versions.get(name) != version
    }
    if mismatches:
        raise AssertionError(f"Library versions do not match metadata: {mismatches}")

    print("Contract OK")
    print(f"Features: {', '.join(ALL_FEATURES)}")
    print(f"Model: {metadata.get('model_name')}")
    print(f"Versions: {actual_versions}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError) as error:
        print(f"Contract FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
