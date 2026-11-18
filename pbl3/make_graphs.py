"""Graphs for the six PBL3 models (SVM, CART, GAM, Random Forest, MoE, Stacking).

Saved to pbl3/graphs/:
  confusion_matrices_photo_input.png / confusion_matrices_with_grades.png  (all 6 in one figure)
  photo_input/<model>.png, with_grades/<model>.png                         (one per model)
  accuracy_macro_f1_comparison.png   with grades vs photo input
  per_class_f1_photo_input.png       per-class F1 for each model

Run:  python make_graphs.py   (from the pbl3 folder)
"""
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import codd_models  # noqa: F401  (needed to load MixtureOfExperts)
from codd_common import CLASS_NAMES, FEATURE_COLUMNS, load_grouped_splits

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "graphs"

MODELS = ["SVM", "CART", "GAM", "Random Forest", "Mixture of Experts", "Stacking"]
MAIN_FILES = {"SVM": "codd_svm_pipeline.joblib", "CART": "codd_cart_pipeline.joblib",
              "GAM": "codd_gam_pipeline.joblib", "Random Forest": "codd_random_forest.joblib",
              "Mixture of Experts": "codd_moe_pipeline.joblib", "Stacking": "codd_stacking_top5.joblib"}
PHOTO_FILES = {"SVM": "photo_svm.joblib", "CART": "photo_cart.joblib", "GAM": "photo_gam.joblib",
               "Random Forest": "photo_random_forest.joblib", "Mixture of Experts": "photo_moe.joblib",
               "Stacking": "photo_stacking.joblib"}

BLUE, ORANGE = "#2a6fb0", "#e07b00"          # with grades / photo input
CLASS_COLORS = ["#7b3fa0", "#2a6fb0", "#3a9d5d", "#d1495b"]  # Excellent, Good, Fair, Poor


def predict_all(files, folder, columns, X, y):
    """Return {model: predictions} for the saved models on the test split."""
    return {name: joblib.load(folder / f).predict(X[columns]) for name, f in files.items()}


def draw_cm(ax, y_true, y_pred, title):
    """Confusion matrix: colour = share of the true class, number = item count."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    share = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(share, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def confusion_figures(preds, y, folder_name, setting):
    """Save one grid with all models and one PNG per model."""
    (OUT / folder_name).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name in zip(axes.ravel(), MODELS):
        acc = accuracy_score(y, preds[name])
        draw_cm(ax, y, preds[name], f"{name}  (accuracy {acc:.3f})")
        single, sax = plt.subplots(figsize=(5.5, 4.5))
        draw_cm(sax, y, preds[name], f"{name} - {setting}\naccuracy {acc:.3f}")
        single.tight_layout()
        single.savefig(OUT / folder_name / f"{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(single)
    fig.suptitle(f"Test confusion matrices - {setting}\n(colour = share of each actual class, numbers = items)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_matrices_{folder_name}.png", dpi=150)
    plt.close(fig)


def comparison_chart(main, photo, y):
    """Grouped bars: accuracy and macro F1, with grades vs photo input."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(MODELS))
    for ax, (label, fn) in zip(axes, [
            ("Accuracy", accuracy_score),
            ("Macro F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0))]):
        a = [fn(y, main[m]) for m in MODELS]
        b = [fn(y, photo[m]) for m in MODELS]
        r1 = ax.bar(x - 0.2, a, 0.4, label="With damage grades (30 features)", color=BLUE)
        r2 = ax.bar(x + 0.2, b, 0.4, label="Photo input (24 features)", color=ORANGE)
        for rect in list(r1) + list(r2):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                    f"{rect.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Mixture of Experts", "MoE").replace("Random Forest", "Random\nForest")
                            for m in MODELS])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{label} on the test split")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "accuracy_macro_f1_comparison.png", dpi=150)
    plt.close(fig)


def per_class_chart(photo, y):
    """Grouped bars: F1 of each condition class for every photo-input model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(MODELS))
    for i, cls in enumerate(CLASS_NAMES):
        vals = [f1_score(y, photo[m], labels=[cls], average=None, zero_division=0)[0] for m in MODELS]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=cls, color=CLASS_COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Photo-input models: F1 per condition class (test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "per_class_f1_photo_input.png", dpi=150)
    plt.close(fig)


def main():
    """Build every graph from the saved models."""
    data = load_grouped_splits(verbose=False)
    y = data.y_test
    OUT.mkdir(exist_ok=True)
    photo_cols = joblib.load(BASE / "photo_models" / "photo_feature_columns.joblib")

    main_preds = predict_all(MAIN_FILES, BASE / "saved_models", FEATURE_COLUMNS, data.X_test, y)
    photo_preds = predict_all(PHOTO_FILES, BASE / "photo_models", photo_cols, data.X_test, y)

    confusion_figures(photo_preds, y, "photo_input", "photo input (24 features)")
    confusion_figures(main_preds, y, "with_grades", "with damage grades (30 features)")
    comparison_chart(main_preds, photo_preds, y)
    per_class_chart(photo_preds, y)
    print("Saved graphs in", OUT)
    for p in sorted(OUT.rglob("*.png")):
        print(" ", p.relative_to(BASE))


if __name__ == "__main__":
    main()
