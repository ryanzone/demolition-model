"""
PBL 1: Exploratory Data Analysis & Diagnostic Chart Generation
=============================================================
Generates:
  1. Material distribution (X-axis = material category, Y-axis = record count)
  2. Condition distribution (X-axis = condition class, Y-axis = record count)
  3. Recovery pathway distribution (X-axis = recovery pathway, Y-axis = record count)
  4. Correlation heatmap (Strictly numerical visual features)
  5. Data leakage check & Class Imbalance report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold'
})

def prepare_eda_charts():
    # Identify paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    output_dir = current_dir

    print("[*] Loading datasets for EDA charts...")
    train = pd.read_csv(os.path.join(project_dir, "training_image_features.csv"))
    val = pd.read_csv(os.path.join(project_dir, "validation_image_features.csv"))
    test = pd.read_csv(os.path.join(project_dir, "testing_image_features.csv"))

    # Full structured dataset
    df = pd.concat([train, val, test], ignore_index=True)
    total_records = len(df)
    print(f"[+] Total records combined: {total_records}")

    # =========================================================================
    # CHART 1: Material Distribution
    # =========================================================================
    plt.figure(figsize=(9, 5))
    mat_counts = df["Material_Type"].value_counts()
    ax1 = sns.barplot(x=mat_counts.index, y=mat_counts.values, hue=mat_counts.index, palette="viridis", legend=False)
    
    # Annotate bar counts and percentages
    for p in ax1.patches:
        height = p.get_height()
        pct = (height / total_records) * 100
        ax1.annotate(f"{int(height):,}\n({pct:.1f}%)",
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=8.5, xytext=(0, 3),
                     textcoords='offset points')

    plt.title("Material Category Distribution (Total N = 16,519)", pad=15)
    plt.xlabel("Material Category")
    plt.ylabel("Record Count")
    plt.xticks(rotation=35, ha='right')
    plt.ylim(0, mat_counts.max() * 1.18)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "eda_1_material_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {chart1_path}")

    # =========================================================================
    # CHART 2: Condition Distribution
    # =========================================================================
    plt.figure(figsize=(8, 5))
    condition_order = ["Excellent", "Good", "Fair", "Poor"]
    cond_counts = df["Condition_Class"].value_counts().reindex(condition_order)
    
    colors_cond = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    ax2 = sns.barplot(x=cond_counts.index, y=cond_counts.values, hue=cond_counts.index, palette=colors_cond, legend=False)

    for p in ax2.patches:
        height = p.get_height()
        pct = (height / total_records) * 100
        ax2.annotate(f"{int(height):,}\n({pct:.1f}%)",
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                     textcoords='offset points')

    plt.title("Condition Class Distribution (Severe Class Imbalance)", pad=15)
    plt.xlabel("Condition Class")
    plt.ylabel("Record Count")
    plt.ylim(0, cond_counts.max() * 1.18)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "eda_2_condition_distribution.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {chart2_path}")

    # =========================================================================
    # CHART 3: Recovery Pathway Distribution
    # =========================================================================
    plt.figure(figsize=(9, 5))
    rec_counts = df["Recovery_Pathway"].value_counts()
    colors_rec = ["#8e44ad", "#16a085", "#d35400"]
    ax3 = sns.barplot(x=rec_counts.index, y=rec_counts.values, hue=rec_counts.index, palette=colors_rec, legend=False)

    for p in ax3.patches:
        height = p.get_height()
        pct = (height / total_records) * 100
        ax3.annotate(f"{int(height):,}\n({pct:.1f}%)",
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                     textcoords='offset points')

    plt.title("Recovery Pathway Distribution (Dominant Class Skew)", pad=15)
    plt.xlabel("Recovery Pathway")
    plt.ylabel("Record Count")
    plt.ylim(0, rec_counts.max() * 1.18)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "eda_3_recovery_pathway_distribution.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {chart3_path}")

    # =========================================================================
    # CHART 4: Correlation Heatmap (Strictly Numerical Features)
    # =========================================================================
    numerical_features = [
        'pixel_obj_width', 'pixel_obj_height', 'relative_area', 'edge_density',
        'fill_ratio', 'mean_brightness', 'std_brightness', 'contrast',
        'texture_laplacian_var', 'dark_pixel_ratio', 'bright_pixel_ratio',
        'mean_hue', 'mean_saturation', 'circularity', 'solidity'
    ]
    
    # Calculate exact Pearson correlation matrix
    corr_matrix = df[numerical_features].corr(method='pearson')

    # Mask upper triangle for readability
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"}
    )
    plt.title("Correlation Heatmap: Numerical Computer Vision Features\n(Excluding Derived Target Scores to Prevent Leakage)", pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "eda_4_correlation_heatmap.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {chart4_path}")

    # =========================================================================
    # PRINT SUMMARY STATS & DATA LEAKAGE REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("  DATA LEAKAGE DIAGNOSTIC CHECK")
    print("=" * 70)
    
    # Range overlap check between Condition_Class and Damage/Condition_Score
    cond_leakage = df.groupby("Condition_Class")[["Damage_Score", "Condition_Score"]].agg(['min', 'max', 'count'])
    print("\nCondition_Class vs Continuous Scores:")
    print(cond_leakage)

    rec_leakage = df.groupby("Recovery_Pathway")[["Damage_Score", "Condition_Score"]].agg(['min', 'max', 'count'])
    print("\nRecovery_Pathway vs Continuous Scores:")
    print(rec_leakage)

    # Save distribution counts to CSV for report inclusion
    mat_counts.to_frame("Count").to_csv(os.path.join(output_dir, "summary_material_distribution.csv"))
    cond_counts.to_frame("Count").to_csv(os.path.join(output_dir, "summary_condition_distribution.csv"))
    rec_counts.to_frame("Count").to_csv(os.path.join(output_dir, "summary_recovery_distribution.csv"))
    corr_matrix.to_csv(os.path.join(output_dir, "summary_correlation_matrix.csv"))

    print("\n[SUCCESS] All 4 EDA charts and diagnostic summaries generated successfully!")

if __name__ == "__main__":
    prepare_eda_charts()
