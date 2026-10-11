import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
CSV_FILES = [
    "training_image_features.csv",
    "validation_image_features.csv",
    "testing_image_features.csv"
]

dataframes = []

for file in CSV_FILES:
    if os.path.exists(file):
        dataframes.append(pd.read_csv(file))

if not dataframes:
    raise FileNotFoundError("No CSV files were found.")

df = pd.concat(
    dataframes,
    ignore_index=True
)

total_samples = len(df)
total_columns = len(df.columns)

condition_counts = df["Condition_Class"].value_counts()
condition_percentages = (
    df["Condition_Class"]
    .value_counts(normalize=True)
    * 100
)

most_common_condition = condition_counts.idxmax()
most_common_condition_percentage = (
    condition_percentages[most_common_condition]
)

least_common_condition = condition_counts.idxmin()
least_common_condition_percentage = (
    condition_percentages[least_common_condition]
)

recovery_counts = df["Recovery_Pathway"].value_counts()
recovery_percentages = (
    df["Recovery_Pathway"]
    .value_counts(normalize=True)
    * 100
)

dominant_recovery = recovery_counts.idxmax()
dominant_recovery_percentage = (
    recovery_percentages[dominant_recovery]
)

damage_mean = df["Damage_Score"].mean()
damage_min = df["Damage_Score"].min()
damage_max = df["Damage_Score"].max()

condition_mean = df["Condition_Score"].mean()
condition_min = df["Condition_Score"].min()
condition_max = df["Condition_Score"].max()

material_counts = df["Material_Type"].value_counts()
material_percentages = (
    df["Material_Type"]
    .value_counts(normalize=True)
    * 100
)

top_materials = material_counts.head(3)

top_material_text = ", ".join(
    [
        f"{material} "
        f"({material_percentages[material]:.2f}%)"
        for material in top_materials.index
    ]
)

print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print()
print(f"Total samples: {total_samples:,}")
print(f"Total columns: {total_columns}")

print()
print("Condition Distribution:")

for condition, count in condition_counts.items():
    print(
        f"{condition}: "
        f"{count:,} "
        f"({condition_percentages[condition]:.2f}%)"
    )

print()
print(
    f"Most common condition: "
    f"{most_common_condition} "
    f"({most_common_condition_percentage:.2f}%)"
)

print(
    f"Least common condition: "
    f"{least_common_condition} "
    f"({least_common_condition_percentage:.2f}%)"
)

print()
print("Recovery Pathway Distribution:")

for pathway, count in recovery_counts.items():
    print(
        f"{pathway}: "
        f"{count:,} "
        f"({recovery_percentages[pathway]:.2f}%)"
    )

print()
print(
    f"Dominant recovery pathway: "
    f"{dominant_recovery} "
    f"({dominant_recovery_percentage:.2f}%)"
)

print()
print("Damage Score:")
print(f"Mean: {damage_mean:.2f}")
print(f"Minimum: {damage_min:.2f}")
print(f"Maximum: {damage_max:.2f}")

print()
print("Condition Score:")
print(f"Mean: {condition_mean:.2f}")
print(f"Minimum: {condition_min:.2f}")
print(f"Maximum: {condition_max:.2f}")

print()
print("Top 3 Materials:")
print(top_material_text)

print()
print("=" * 70)
print("FINAL KEY FINDINGS")
print("=" * 70)

print()

print(
    f"• {most_common_condition} condition was the most common "
    f"classification, accounting for "
    f"{most_common_condition_percentage:.2f}% of the dataset, "
    f"while {least_common_condition} condition accounted for "
    f"{least_common_condition_percentage:.2f}%."
)

print()

print(
    f"• {dominant_recovery} was the dominant recovery pathway, "
    f"selected for {dominant_recovery_percentage:.2f}% "
    f"of the samples."
)

print()

print(
    f"• Damage Score had a mean of {damage_mean:.2f}, "
    f"with values ranging from {damage_min:.2f} to "
    f"{damage_max:.2f}, indicating varying levels of "
    f"material damage."
)

print()

print(
    f"• Condition Score had a mean of {condition_mean:.2f}, "
    f"with values ranging from {condition_min:.2f} to "
    f"{condition_max:.2f}, showing variation in "
    f"material condition."
)

print()

print(
    f"• The three most frequent material types were "
    f"{top_material_text}."
)

print()

print(
    "• Image-derived features such as edge density, "
    "brightness, contrast, and texture variation were "
    "used to support visual condition classification."
)

print()
print("=" * 70)

X_reg = df[["Damage_Score"]]
y_reg = df["Condition_Score"]

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

reg_predictions = reg_model.predict(X_reg)

r2 = r2_score(
    y_reg,
    reg_predictions
)