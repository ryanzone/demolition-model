# ============================================================
# CODD PBL3 - STACKING CLASSIFIER
# SVM + CART + GAM + Random Forest + MoE
#
# Uses the shared grouped splits and the same 30 features as every
# other PBL3 model (see codd_common.py). The base-model hyperparameters
# are taken from the saved, validation-tuned pipelines. Base models are
# re-fit inside the stack; the meta-learner is trained on out-of-fold
# predictions from folds grouped by Image_ID, so near-duplicate crops of
# one image never sit on both sides of a fold.
# ============================================================

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

import codd_models  # noqa: F401  (needed so saved MixtureOfExperts files unpickle)
from codd_common import (
    CLASS_NAMES, FEATURE_COLUMNS, RANDOM_STATE, assert_feature_set,
    load_grouped_splits,
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "stacking_results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "svm": "codd_svm_pipeline.joblib",
    "cart": "codd_cart_pipeline.joblib",
    "gam": "codd_gam_pipeline.joblib",
    "random_forest": "codd_random_forest.joblib",
    "moe": "codd_moe_pipeline.joblib",
}


def summarize(y_true, y_pred, model, split):
    return {
        "Model": model,
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def load_base_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing base model: {path}. Run its notebook first.")
        model = joblib.load(path)
        # Every base model must have been trained on exactly the shared features.
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None:
            assert_feature_set(cols)
        models[name] = model
        print(f"Loaded {name}: {type(model).__name__}")
    return models


def build_stack(base_models):
    """Unfitted stack: same hyperparameters as the saved base models."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cv, StackingClassifier(
        estimators=[(name, clone(model)) for name, model in base_models.items()],
        final_estimator=LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def fit_stack(base_models, X, y, groups):
    cv, stack = build_stack(base_models)
    # Precomputed image-grouped folds for the out-of-fold meta features.
    stack.set_params(cv=list(cv.split(X, y, groups=groups)))
    return stack.fit(X, y)


def main():
    data = load_grouped_splits()
    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train, data.y_val, data.y_test
    for X in (X_train, X_val, X_test):
        assert_feature_set(X.columns)

    base_models = load_base_models()

    # ---- Validation: stack fit on train only
    print("\nTraining stacking classifier on the training split...")
    stack_val = fit_stack(base_models, X_train, y_train, data.groups_train)
    val_pred = stack_val.predict(X_val)
    val_row = summarize(y_val, val_pred, "Stacking Classifier", "Validation")
    print("\n========== VALIDATION ==========")
    print(classification_report(y_val, val_pred, labels=CLASS_NAMES, zero_division=0))

    # ---- Final: refit on train + validation, evaluate once on test
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    g_tv = pd.concat([data.groups_train, data.groups_val], ignore_index=True)

    print("Refitting on train + validation...")
    final_stack = fit_stack(base_models, X_tv, y_tv, g_tv)
    test_pred = final_stack.predict(X_test)
    test_row = summarize(y_test, test_pred, "Stacking Classifier", "Test")
    print("\n========== TEST ==========")
    report = classification_report(y_test, test_pred, labels=CLASS_NAMES, zero_division=0)
    print(report)

    metrics = pd.DataFrame([val_row, test_row])
    metrics.to_csv(RESULTS_DIR / "stacking_metrics.csv", index=False)
    (RESULTS_DIR / "stacking_test_classification_report.txt").write_text(report)

    joblib.dump(final_stack, MODEL_DIR / "codd_stacking_top5.joblib")

    # Meta-learner weights show how much each base model contributes.
    meta = final_stack.final_estimator_
    names = [f"{n}:{c}" for n in base_models for c in final_stack.classes_]
    pd.DataFrame(meta.coef_.T, index=names, columns=meta.classes_).to_csv(
        RESULTS_DIR / "stacking_meta_coefficients.csv")

    cm = confusion_matrix(y_test, test_pred, labels=CLASS_NAMES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Stacking Classifier - Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stacking_confusion_matrix.png", dpi=200)
    plt.close()

    print(metrics.to_string(index=False))
    print(f"\nSaved model and results under {BASE_DIR}")


if __name__ == "__main__":
    main()
