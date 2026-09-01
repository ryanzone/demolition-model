import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LOAD DATASETS
# ==========================================

train = pd.read_csv("training_image_features.csv")
validation = pd.read_csv("validation_image_features.csv")
test = pd.read_csv("testing_image_features.csv")

# Combine datasets
df = pd.concat([train, validation, test], ignore_index=True)

# Seaborn appearance
sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# ==========================================
# 1. SCATTER PLOT
# Damage Score vs Condition Score
# ==========================================

plt.figure(figsize=(6, 4))

sns.regplot(
    data=df,
    x="Damage_Score",
    y="Condition_Score",
    scatter_kws={"alpha": 0.4, "s": 25},
    line_kws={"linewidth": 2}
)

plt.title("Damage Score vs Condition Score")
plt.xlabel("Damage Score")
plt.ylabel("Condition Score")

plt.tight_layout()
plt.savefig(
    "1_damage_vs_condition_scatter.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 2. HISTOGRAM
# Damage Score Distribution
# ==========================================

plt.figure(figsize=(6, 4))

sns.histplot(
    data=df,
    x="Damage_Score",
    bins=10,
    kde=True
)

plt.title("Distribution of Damage Score")
plt.xlabel("Damage Score")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    "2_damage_score_histogram.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 3. BOX PLOT
# Damage Score
# ==========================================

plt.figure(figsize=(5, 4))

sns.boxplot(
    data=df,
    y="Damage_Score"
)

plt.title("Damage Score Box Plot")
plt.ylabel("Damage Score")

plt.tight_layout()
plt.savefig(
    "3_damage_score_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 4. BAR CHART
# Material Type
# ==========================================

material_order = df["Material_Type"].value_counts().index

plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="Material_Type",
    order=material_order
)

plt.title("Distribution of Material Types")
plt.xlabel("Material Type")
plt.ylabel("Number of Samples")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    "4_material_type_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 5. CONDITION CLASS
# ==========================================

plt.figure(figsize=(6, 4))

condition_order = df["Condition_Class"].value_counts().index

sns.countplot(
    data=df,
    x="Condition_Class",
    order=condition_order
)

plt.title("Distribution of Condition Classes")
plt.xlabel("Condition Class")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.savefig(
    "5_condition_class_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


print("All 5 Seaborn visualizations generated successfully!")