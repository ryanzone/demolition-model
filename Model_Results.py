import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score


# ============================================================
# SETTINGS
# ============================================================

CSV_FILES = [
    "training_image_features.csv",
    "validation_image_features.csv",
    "testing_image_features.csv"
]


# ============================================================
# LOAD DATA
# ============================================================

dataframes = []

for file in CSV_FILES:

    if os.path.exists(file):

        dataframes.append(
            pd.read_csv(file)
        )

        print(
            f"Loaded: {file}"
        )

    else:

        print(
            f"File not found: {file}"
        )


if not dataframes:

    raise FileNotFoundError(
        "No CSV files were found."
    )


df = pd.concat(
    dataframes,
    ignore_index=True
)


# ============================================================
# LINEAR REGRESSION
# DAMAGE SCORE -> CONDITION SCORE
# ============================================================

X_reg = df[
    ["Damage_Score"]
]

y_reg = df[
    "Condition_Score"
]


linear_model = LinearRegression()

linear_model.fit(
    X_reg,
    y_reg
)


condition_prediction = (
    linear_model.predict(
        X_reg
    )
)


linear_r2 = r2_score(
    y_reg,
    condition_prediction
)


# ============================================================
# SVM
# CONDITION CLASSIFICATION
# ============================================================

feature_columns = [

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

    "pixel_obj_width",

    "pixel_obj_height",

    "relative_area"

]


X = df[
    feature_columns
].copy()

y = df[
    "Condition_Class"
]


X = X.fillna(
    X.median(
        numeric_only=True
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# SVM MODEL
# ============================================================

svm_model = Pipeline(

    [

        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                random_state=42
            )
        )

    ]

)


svm_model.fit(
    X_train,
    y_train
)


y_prediction = (
    svm_model.predict(
        X_test
    )
)


svm_accuracy = accuracy_score(
    y_test,
    y_prediction
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print()

print(
    "Linear Regression"
)

print(
    f"R² Score: {linear_r2:.4f}"
)

print()

print(
    "SVM"
)

print(
    f"Test Accuracy: "
    f"{svm_accuracy * 100:.2f}%"
)

print()

print("=" * 70)
print("VALUES FOR KEY FINDINGS")
print("=" * 70)

print()

print(
    f"Linear Regression R² = "
    f"{linear_r2:.4f}"
)

print(
    f"SVM Test Accuracy = "
    f"{svm_accuracy * 100:.2f}%"
)

print()

print("=" * 70)