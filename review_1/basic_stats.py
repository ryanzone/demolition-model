import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("training_image_features_1.csv")
validation = pd.read_csv("validation_image_features_1.csv")
test = pd.read_csv("testing_image_features_1.csv")

df = pd.concat([train, validation, test], ignore_index=True)

columns = [
    "Damage_Score",
    "Condition_Score",
    "Crack_Level",
    "Surface_Damage",
    "Breakage_Level"
]

comparison = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Median",
        "Mode",
        "Minimum",
        "Maximum",
        "Range",
        "Standard Deviation"
    ]
})

for column in columns:
    data = df[column]

    comparison[column] = [
        data.mean(),
        data.median(),
        data.mode().iloc[0],
        data.min(),
        data.max(),
        data.max() - data.min(),
        data.std()
    ]

comparison[columns] = comparison[columns].round(2)

print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.axis("off")

table = ax.table(
    cellText=comparison.values,
    colLabels=comparison.columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.20, 0.16, 0.16, 0.16, 0.16, 0.16]
)

table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#999999")

    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#233F70")

plt.savefig(
    "basic_statistics_comparison.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()