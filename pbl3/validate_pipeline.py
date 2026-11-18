"""Independent check of the saved PBL3 models.

Loads every saved model from disk, confirms they all use the shared 30
features and the grouped splits, and scores them on the test split.
Also trains an annotation-free Random Forest as a reference, because the six
damage annotation columns are only valid inputs if they exist at prediction
time (they are manual grades that drive the label).
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import codd_models  # noqa: F401  (unpickling MixtureOfExperts)
from codd_common import (
    ANNOTATION_FEATURES, FEATURE_COLUMNS, RANDOM_STATE, assert_feature_set,
    load_grouped_splits,
)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
MODELS = {
    "SVM": "codd_svm_pipeline.joblib",
    "CART": "codd_cart_pipeline.joblib",
    "GAM": "codd_gam_pipeline.joblib",
    "Random Forest": "codd_random_forest.joblib",
    "MoE": "codd_moe_pipeline.joblib",
    "Stacking": "codd_stacking_top5.joblib",
}


def row(name, y, p):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, p),
        "Macro Precision": precision_score(y, p, average="macro", zero_division=0),
        "Macro Recall": recall_score(y, p, average="macro", zero_division=0),
        "Macro F1": f1_score(y, p, average="macro", zero_division=0),
        "Weighted F1": f1_score(y, p, average="weighted", zero_division=0),
    }


data = load_grouped_splits()
rows = []
for name, filename in MODELS.items():
    model = joblib.load(BASE / "saved_models" / filename)
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        assert_feature_set(cols)
        status = "30 shared features verified"
    else:
        status = "pipeline column check via predict on 30-feature frame"
    pred = model.predict(data.X_test[FEATURE_COLUMNS])
    rows.append(row(name, data.y_test, pred))
    print(f"{name:14s} {status}")

# Reference: same Random Forest recipe without the six annotation columns.
ref_cols = [c for c in FEATURE_COLUMNS if c not in ANNOTATION_FEATURES]
num = [c for c in data.numeric_features if c in ref_cols]
cat = [c for c in data.categorical_features if c in ref_cols]
ref = Pipeline([
    ("pre", ColumnTransformer([
        ("n", SimpleImputer(strategy="median"), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])),
    ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample",
                                 random_state=RANDOM_STATE, n_jobs=-1)),
])
X_tv = pd.concat([data.X_train, data.X_val])[ref_cols]
y_tv = pd.concat([data.y_train, data.y_val])
ref.fit(X_tv, y_tv)
rows.append(row("RF without annotations (reference)", data.y_test, ref.predict(data.X_test[ref_cols])))

out = pd.DataFrame(rows)
out_dir = BASE / "model_comparison_results"
out_dir.mkdir(exist_ok=True)
out.to_csv(out_dir / "pbl3_final_test_results.csv", index=False)
print()
print(out.round(4).to_string(index=False))
