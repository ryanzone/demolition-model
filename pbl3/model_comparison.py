"""CODD PBL3 model comparison.

Loads the saved, validation-tuned pipelines, re-fits an unfitted copy of each
on the training split for validation scoring, then re-fits on train +
validation for the single test evaluation. All models use the shared grouped
splits and the same 30 features (see codd_common.py).
"""
import traceback
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import codd_models  # noqa: F401  (needed so MixtureOfExperts unpickles)
from codd_common import assert_feature_set, load_grouped_splits

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
FINAL_MODEL_DIR = BASE_DIR / "final_models"
OUTPUT_DIR = BASE_DIR / "model_comparison_results"

MODEL_FILES = {
    "GAM": "codd_gam_pipeline.joblib",
    "CART": "codd_cart_pipeline.joblib",
    "MoE": "codd_moe_pipeline.joblib",
    "Random Forest": "codd_random_forest.joblib",
    "AdaBoost": "codd_adaboost.joblib",
    "SVM": "codd_svm_pipeline.joblib",
}


def fit_model(model, name, X, y):
    """AdaBoost is trained with balanced sample weights, as in its notebook."""
    if name == "AdaBoost":
        return model.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


def evaluate(model, X, y, name, split):
    pred = [str(p) for p in model.predict(X)]
    return {
        "Model": name,
        "Split": split,
        "Accuracy": accuracy_score(y, pred),
        "Macro Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y, pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y, pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y, pred, average="weighted", zero_division=0),
    }


def main():
    FINAL_MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_grouped_splits()
    X_tv = pd.concat([data.X_train, data.X_val], ignore_index=True)
    y_tv = pd.concat([data.y_train, data.y_val], ignore_index=True)

    results = []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"Skipping {name}: {filename} not found")
            continue
        try:
            base = joblib.load(path)
            cols = getattr(base, "feature_names_in_", None)
            if cols is not None:
                assert_feature_set(cols)

            print(f"\nEvaluating {name}...")
            val_model = fit_model(clone(base), name, data.X_train, data.y_train)
            val_row = evaluate(val_model, data.X_val, data.y_val, name, "Validation")

            final_model = fit_model(clone(base), name, X_tv, y_tv)
            test_row = evaluate(final_model, data.X_test, data.y_test, name, "Test")

            joblib.dump(final_model, FINAL_MODEL_DIR / filename)
            results += [val_row, test_row]
            print(f"  Validation macro F1 {val_row['Macro F1']:.4f} | "
                  f"Test macro F1 {test_row['Macro F1']:.4f}")
        except Exception:
            print(f"ERROR in {name}:")
            traceback.print_exc()

    if not results:
        raise RuntimeError("No models could be evaluated.")

    all_df = pd.DataFrame(results)
    validation = (all_df[all_df["Split"] == "Validation"]
                  .sort_values(["Macro F1", "Accuracy"], ascending=False)
                  .reset_index(drop=True))
    validation["Rank"] = range(1, len(validation) + 1)

    test = all_df[all_df["Split"] == "Test"].drop(columns="Split").add_prefix("Test ")
    test = test.rename(columns={"Test Model": "Model"})
    top5 = validation.head(5)
    top5_final = top5.merge(test, on="Model", how="left")

    all_df.to_csv(OUTPUT_DIR / "all_model_split_results.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation_model_ranking.csv", index=False)
    top5.to_csv(OUTPUT_DIR / "top_5_models_validation.csv", index=False)
    top5_final.to_csv(OUTPUT_DIR / "top_5_with_test_results.csv", index=False)

    print("\nVALIDATION RANKING (by Macro F1)")
    print(validation.round(4).to_string(index=False))
    print("\nTOP 5 WITH TEST RESULTS")
    print(top5_final.round(4).to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
