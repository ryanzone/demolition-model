import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("training_image_features_1.csv")
validation = pd.read_csv("validation_image_features_1.csv")
test = pd.read_csv("testing_image_features_1.csv")

df = pd.concat([train, validation, test], ignore_index=True)

print("DATASET SUMMARY")
print("-" * 40)
print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))
print("Training Rows:", len(train))
print("Validation Rows:", len(validation))
print("Testing Rows:", len(test))

print("\nData Types:")
print("Numerical Columns:", len(df.select_dtypes(include="number").columns))
print("Categorical Columns:", len(df.select_dtypes(exclude="number").columns))

print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

sample_columns = [
    "Material_Type",
    "Material_Subtype",
    "pixel_obj_width",
    "Crack_Level",
    "Damage_Score",
    "Condition_Class",
    "Recovery_Pathway"
]

sample = df[sample_columns].head(5).copy()

sample.columns = [
    "Material",
    "Subtype",
    "Object Width",
    "Crack",
    "Damage",
    "Condition",
    "Recovery"
]

sample["Object Width"] = sample["Object Width"].round(1)
sample["Damage"] = sample["Damage"].round(1)

fig, ax = plt.subplots(figsize=(7.5, 2.0))

ax.axis("off")

table = ax.table(
    cellText=sample.values,
    colLabels=sample.columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.14, 0.14, 0.15, 0.09, 0.10, 0.14, 0.22]
)

table.auto_set_font_size(False)
table.set_fontsize(6.5)
table.scale(1, 1.4)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#233F70")
        cell.get_text().set_color("white")

    cell.set_edgecolor("#999999")
    cell.set_linewidth(0.5)

plt.savefig(
    "dataset_sample_table.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()