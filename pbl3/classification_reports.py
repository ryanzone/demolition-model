"""Detailed test results (accuracy, macro scores, per-class report) for every PBL3 model.

Writes two text files to model_comparison_results/:
  * pbl3_classification_reports.txt        - models using the 30 features (with damage grades)
  * photo_only_classification_reports.txt  - photo-input models (24 features, no grades)

Run:  python classification_reports.py   (from the pbl3 folder)
"""
import warnings
from pathlib import Path

import joblib
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

import codd_models  # noqa: F401  (needed to load MixtureOfExperts)
from codd_common import CLASS_NAMES, FEATURE_COLUMNS, load_grouped_splits

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "model_comparison_results"

MAIN_MODELS = {
    "SVM": "codd_svm_pipeline.joblib",
    "CART": "codd_cart_pipeline.joblib",
    "GAM": "codd_gam_pipeline.joblib",
    "Random Forest": "codd_random_forest.joblib",
    "AdaBoost": "codd_adaboost.joblib",
    "Mixture of Experts": "codd_moe_pipeline.joblib",
    "Voting Ensemble": "codd_voting_ensemble.joblib",
    "Stacking": "codd_stacking_top5.joblib",
}
PHOTO_MODELS = {
    "SVM": "photo_svm.joblib",
    "CART": "photo_cart.joblib",
    "GAM": "photo_gam.joblib",
    "Random Forest": "photo_random_forest.joblib",
    "Mixture of Experts": "photo_moe.joblib",
    "Stacking": "photo_stacking.joblib",
}


def format_report(name, y_true, y_pred, setting):
    """Return the test-results text block for one model (same layout as the notebooks)."""
    title = f"{name} Test Results ({setting})"
    lines = [
        title,
        "-" * len(title),
        f"Accuracy : {accuracy_score(y_true, y_pred):.4f}",
        f"Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        f"Recall (macro)   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        f"F1-score (macro) : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "",
        "Classification Report:",
        classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0),
    ]
    return "\n".join(lines)


def build(models, folder, columns, X_test, y_test, setting):
    """Evaluate each saved model on the test split and return the joined text."""
    blocks = []
    for name, filename in models.items():
        path = folder / filename
        if not path.exists():
            print(f"Skipping {name}: {path.name} not found")
            continue
        pred = joblib.load(path).predict(X_test[columns])
        blocks.append(format_report(name, y_test, pred, setting))
    return ("\n" + "=" * 60 + "\n\n").join(blocks)


def main():
    """Write both report files and print them."""
    data = load_grouped_splits(verbose=False)
    OUT.mkdir(exist_ok=True)

    main_text = build(MAIN_MODELS, BASE / "saved_models", FEATURE_COLUMNS,
                      data.X_test, data.y_test, "30 features, with damage grades")
    (OUT / "pbl3_classification_reports.txt").write_text(main_text)

    photo_cols = joblib.load(BASE / "photo_models" / "photo_feature_columns.joblib")
    photo_text = build(PHOTO_MODELS, BASE / "photo_models", photo_cols,
                       data.X_test, data.y_test, "photo input, 24 features")
    (OUT / "photo_only_classification_reports.txt").write_text(photo_text)

    print(main_text)
    print("\n" + "#" * 60 + "\n")
    print(photo_text)


if __name__ == "__main__":
    main()
