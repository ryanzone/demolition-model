"""Shared data setup for the CODD PBL2 notebooks.

All PBL2 notebooks use the grouped splits (no Image_ID shared between splits)
and only features computed from the image itself. The manual damage grades
and the score columns are not used, because the condition label is derived
from them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = DATA_DIR / "training_image_features_grouped.csv"
TEST_PATH = DATA_DIR / "testing_image_features_grouped.csv"

TARGET = "Condition_Class"
STATES = ["Excellent", "Good", "Fair", "Poor"]

# Image-derived features only (no pixel_img_* sizes, no manual grades, no scores).
IMAGE_FEATURES = [
    "pixel_obj_width", "pixel_obj_height", "relative_area", "edge_density",
    "fill_ratio", "color_std", "mean_brightness", "std_brightness", "contrast",
    "texture_laplacian_var", "dark_pixel_ratio", "bright_pixel_ratio",
    "mean_hue", "mean_saturation", "saturation_std", "contour_area",
    "contour_perimeter", "circularity", "solidity", "extent",
]

# Original PBL2 selection: every numeric column except the three score columns.
_head = pd.read_csv(TRAIN_PATH, nrows=500)
ORIGINAL_FEATURES = [
    c for c in _head.select_dtypes(include=np.number).columns
    if c not in {"Condition_Score", "Damage_Score", "Alternative_Use_Suitability"}
]

# Features the HMM / Viterbi notebooks turn into a discrete observation symbol.
OBSERVATION_FEATURES = ("edge_density", "texture_laplacian_var")


def load_train_test():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    overlap = set(train["Image_ID"].astype(str)) & set(test["Image_ID"].astype(str))
    assert not overlap, f"Image_ID overlap between train and test: {len(overlap)}"
    return train, test


class ObservationEncoder:
    """Turns two image features into one discrete symbol (fit on train only)."""

    def __init__(self, features=OBSERVATION_FEATURES, bins=(4, 3)):
        self.features = features
        self.bins = bins

    def fit(self, df):
        self.cuts_, self.medians_ = [], []
        for col, n in zip(self.features, self.bins):
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            qs = np.linspace(0, 1, n + 1)[1:-1]
            self.cuts_.append(np.unique(np.quantile(values.fillna(median), qs)))
            self.medians_.append(median)
        self.n_symbols = int(np.prod(self.bins))
        return self

    def transform(self, df):
        symbol = np.zeros(len(df), dtype=int)
        for col, cuts, median, n in zip(self.features, self.cuts_, self.medians_, self.bins):
            values = pd.to_numeric(df[col], errors="coerce").fillna(median)
            symbol = symbol * n + np.digitize(values, cuts)
        return symbol
