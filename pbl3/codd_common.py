"""Shared data loading and feature checks for the CODD PBL3 models.

Every PBL3 model (CART, SVM, GAM, Random Forest, MoE, Stacking) must load its
data through load_grouped_splits() so all of them use the same grouped splits
and exactly the same 30 input features.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent

TRAIN_PATH = DATA_DIR / "training_image_features_grouped.csv"
VAL_PATH = DATA_DIR / "validation_image_features_grouped.csv"
TEST_PATH = DATA_DIR / "testing_image_features_grouped.csv"

TARGET = "Condition_Class"
GROUP_COLUMN = "Image_ID"
CLASS_NAMES = ["Excellent", "Good", "Fair", "Poor"]
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Material_Type",
    "Material_Subtype",
    "Source",
    "Exposure_Type",
    "Crack_Level",
    "Surface_Damage",
    "Breakage_Level",
    "Discoloration_Level",
    "Deformation_Level",
    "Missing_Portion",
    "pixel_obj_width",
    "pixel_obj_height",
    "relative_area",
    "edge_density",
    "fill_ratio",
    "color_std",
    "mean_brightness",
    "std_brightness",
    "contrast",
    "texture_laplacian_var",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "mean_hue",
    "mean_saturation",
    "saturation_std",
    "contour_area",
    "contour_perimeter",
    "circularity",
    "solidity",
    "extent",
]

# Target-derived or downstream columns that must never be model inputs.
FORBIDDEN_COLUMNS = [
    "Condition_Score",
    "Damage_Score",
    "Alternative_Use_Suitability",
    "Structural_Integrity",
    "Recovery_Pathway",
    "Reconditioning_Required",
    "Recovery_Potential",
    "Alternative_Use_Category",
]

# Manually graded damage annotations kept as inputs. They are only valid if
# they are available independently of the label at prediction time.
ANNOTATION_FEATURES = [
    "Crack_Level",
    "Surface_Damage",
    "Breakage_Level",
    "Discoloration_Level",
    "Deformation_Level",
    "Missing_Portion",
]


def assert_feature_set(columns):
    """Raise if `columns` is not exactly the agreed 30-feature list."""
    columns = list(columns)
    if columns != FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in columns]
        extra = [c for c in columns if c not in FEATURE_COLUMNS]
        raise AssertionError(
            f"Feature set mismatch. Missing: {missing}. Unexpected: {extra}."
        )
    bad = set(columns) & (set(FORBIDDEN_COLUMNS) | {TARGET, GROUP_COLUMN, "Sample_ID"})
    if bad:
        raise AssertionError(f"Forbidden columns used as features: {sorted(bad)}")


def load_grouped_splits(verbose=True):
    """Load the grouped train/validation/test CSVs and validate them."""
    frames = {}
    for name, path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Grouped split not found: {path}")
        frames[name] = pd.read_csv(path)

    train_df, val_df, test_df = frames["train"], frames["val"], frames["test"]

    ids = {k: set(v[GROUP_COLUMN].astype(str)) for k, v in frames.items()}
    overlaps = {
        "train-val": len(ids["train"] & ids["val"]),
        "train-test": len(ids["train"] & ids["test"]),
        "val-test": len(ids["val"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Image_ID leakage between splits: {overlaps}")

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_val = val_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    assert_feature_set(X_train.columns)

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in FEATURE_COLUMNS if c not in numeric_features]

    data = SimpleNamespace(
        train_df=train_df, val_df=val_df, test_df=test_df,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=train_df[TARGET].astype(str),
        y_val=val_df[TARGET].astype(str),
        y_test=test_df[TARGET].astype(str),
        feature_columns=list(FEATURE_COLUMNS),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        groups_train=train_df[GROUP_COLUMN].astype(str),
        groups_val=val_df[GROUP_COLUMN].astype(str),
    )

    if verbose:
        print("Grouped splits:",
              "train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
        print("Image_ID overlap (must all be 0):", overlaps)
        print(f"Features: {len(FEATURE_COLUMNS)} "
              f"({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    return data
