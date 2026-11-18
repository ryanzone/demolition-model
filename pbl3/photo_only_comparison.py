"""CODD PBL3: photo-only comparison (no manual damage grades).

Same grouped splits, same models and the same validation-based tuning as the
main 30-feature pipeline, but the six manual damage grade columns are removed,
leaving the 24 features that exist for a new photo (material metadata plus
image-derived features). The 30-feature models and results are not touched.
"""
import time
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from codd_common import ANNOTATION_FEATURES, FEATURE_COLUMNS, RANDOM_STATE, load_grouped_splits
from codd_models import MixtureOfExperts

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "model_comparison_results"
OUT.mkdir(exist_ok=True)
MODEL_OUT = Path(__file__).resolve().parent / "photo_models"
MODEL_OUT.mkdir(exist_ok=True)

data = load_grouped_splits()
COLS = [c for c in FEATURE_COLUMNS if c not in ANNOTATION_FEATURES]
assert len(COLS) == 24
X_tr, X_va, X_te = (d[COLS] for d in (data.X_train, data.X_val, data.X_test))
y_tr, y_va, y_te = data.y_train, data.y_val, data.y_test
NUM = [c for c in data.numeric_features if c in COLS]
CAT = [c for c in data.categorical_features if c in COLS]
print(f"Photo-only features: {len(COLS)} ({len(NUM)} numeric, {len(CAT)} categorical)")

cat_pipe = lambda: Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(handle_unknown="ignore"))])


def prep(num_steps, num=NUM, cat=CAT):
    return ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", cat_pipe(), cat)])


med = lambda: ("imp", SimpleImputer(strategy="median"))
scale = lambda: ("sc", StandardScaler())


def metrics(name, y, p):
    return {"Model": name,
            "Accuracy": accuracy_score(y, p),
            "Macro Precision": precision_score(y, p, average="macro", zero_division=0),
            "Macro Recall": recall_score(y, p, average="macro", zero_division=0),
            "Macro F1": f1_score(y, p, average="macro", zero_division=0),
            "Weighted F1": f1_score(y, p, average="weighted", zero_division=0)}


def build(kind, **p):
    if kind == "CART":
        return Pipeline([("pre", prep([med()])), ("m", DecisionTreeClassifier(
            max_depth=p["max_depth"], class_weight="balanced", random_state=RANDOM_STATE))])
    if kind == "SVM":
        return Pipeline([("pre", prep([med(), scale()])), ("m", SVC(
            C=p["C"], gamma=p["gamma"], class_weight="balanced", probability=True,
            random_state=RANDOM_STATE))])
    if kind == "GAM":
        return Pipeline([("pre", prep([med(), ("spl", SplineTransformer(
            n_knots=p["n_knots"], degree=3, knots="quantile", include_bias=False)), scale()])),
            ("m", LogisticRegression(C=p["C"], class_weight="balanced", max_iter=1000,
                                     tol=1e-3, random_state=RANDOM_STATE))])
    if kind == "Random Forest":
        return Pipeline([("pre", prep([med()])), ("m", RandomForestClassifier(
            n_estimators=300, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1))])
    if kind == "MoE":
        return MixtureOfExperts(
            prep([med(), scale()], [c for c in NUM], [c for c in CAT if c != "Material_Type"]),
            gate_col="Material_Type", min_samples=30, random_state=RANDOM_STATE)
    raise ValueError(kind)


GRIDS = {
    "CART": [{"max_depth": d} for d in [3, 5, 8, 12, 16, 20, None]],
    "SVM": [{"C": c, "gamma": g} for c in [0.1, 1, 10, 100] for g in ["scale", "auto"]],
    "GAM": [{"n_knots": k, "C": c} for k in [3, 5] for c in [0.1, 1.0, 10.0]],
    "Random Forest": [{"max_depth": d, "min_samples_leaf": l}
                      for d in [10, 20, None] for l in [1, 2, 5]],
    "MoE": [{}],
}

rows, best_models, best_params, val_rows = [], {}, {}, []
X_tv, y_tv = pd.concat([X_tr, X_va], ignore_index=True), pd.concat([y_tr, y_va], ignore_index=True)
for kind, grid in GRIDS.items():
    t = time.time()
    best = None
    for params in grid:                       # tuning uses validation only
        m = build(kind, **params).fit(X_tr, y_tr)
        f1 = f1_score(y_va, m.predict(X_va), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, params, m)
    best_params[kind] = best[1]
    val_rows.append(metrics(kind, y_va, best[2].predict(X_va)) | {"Split": "Validation"})
    final = build(kind, **best[1]).fit(X_tv, y_tv)   # refit on train + validation
    best_models[kind] = final
    joblib.dump(final, MODEL_OUT / f"photo_{kind.lower().replace(' ', '_')}.joblib")
    rows.append(metrics(kind, y_te, final.predict(X_te)) | {"Split": "Test"})
    print(f"{kind:14s} best={best[1]} val F1={best[0]:.4f} "
          f"test F1={rows[-1]['Macro F1']:.4f} ({time.time()-t:.0f}s)", flush=True)


def fit_stack(X, y, groups):
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups))
    stack = StackingClassifier(
        estimators=[(k.lower().replace(" ", "_"), build(k, **best_params[k])) for k in GRIDS],
        final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced",
                                           random_state=RANDOM_STATE),
        stack_method="predict_proba", cv=cv, n_jobs=1)
    return stack.fit(X, y)


t = time.time()
s_val = fit_stack(X_tr, y_tr, data.groups_train)
val_rows.append(metrics("Stacking", y_va, s_val.predict(X_va)) | {"Split": "Validation"})
g_tv = pd.concat([data.groups_train, data.groups_val], ignore_index=True)
s_final = fit_stack(X_tv, y_tv, g_tv)
rows.append(metrics("Stacking", y_te, s_final.predict(X_te)) | {"Split": "Test"})
print(f"Stacking       val F1={val_rows[-1]['Macro F1']:.4f} test F1={rows[-1]['Macro F1']:.4f} "
      f"({time.time()-t:.0f}s)")

joblib.dump(s_final, MODEL_OUT / "photo_stacking.joblib")
joblib.dump(COLS, MODEL_OUT / "photo_feature_columns.joblib")
test_df = pd.DataFrame(rows).drop(columns="Split")
val_df = pd.DataFrame(val_rows).drop(columns="Split")
test_df.to_csv(OUT / "photo_only_test_results.csv", index=False)
val_df.to_csv(OUT / "photo_only_validation_results.csv", index=False)
pd.Series({k: str(v) for k, v in best_params.items()}).to_csv(OUT / "photo_only_best_params.csv")

# Side-by-side with the 30-feature results already produced by validate_pipeline.py
full_path = OUT / "pbl3_final_test_results.csv"
if full_path.exists():
    full = pd.read_csv(full_path)
    full = full[~full["Model"].str.contains("reference")][["Model", "Accuracy", "Macro F1"]]
    both = full.merge(test_df[["Model", "Accuracy", "Macro F1"]], on="Model",
                      suffixes=(" (30 features)", " (photo-only, 24)"))
    both.to_csv(OUT / "with_vs_without_damage_grades.csv", index=False)
    print("\nWITH vs WITHOUT manual damage grades (test)")
    print(both.round(4).to_string(index=False))
print("\nPHOTO-ONLY TEST RESULTS")
print(test_df.round(4).to_string(index=False))
