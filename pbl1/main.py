"""
PBL 1: Machine Learning Comparative Analysis Pipeline
=====================================================
Tasks Covered:
  1. Polynomial Curve Fitting (Single Feature vs Damage Score / Degree 1 to 5, Overfitting vs Underfitting Analysis)
  2. Regression Algorithms (Linear Regression, Ridge, Lasso, Polynomial Regression, Decision Tree, Random Forest)
  3. Classification Algorithms (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors)
  4. Comparative Performance Analysis (Tabular Metrics & Visual Comparison Plots)
  5. Detailed Explanations for each algorithm (Mathematical intuition, strengths, limitations, and use-case fit)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def load_data(project_dir):
    """
    Loads training, validation, and testing CSV files from the project root.
    """
    train_path = os.path.join(project_dir, "training_image_features.csv")
    val_path = os.path.join(project_dir, "validation_image_features.csv")
    test_path = os.path.join(project_dir, "testing_image_features.csv")

    print("[*] Loading datasets...")
    print(f"    Train: {train_path}")
    print(f"    Validation: {val_path}")
    print(f"    Test: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"[+] Loaded shapes: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    return train_df, val_df, test_df


def section_header(title):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80 + "\n")


# ==============================================================================
# 1. POLYNOMIAL CURVE FITTING
# ==============================================================================
def run_polynomial_curve_fitting(train_df, val_df, test_df, output_dir):
    """
    Fits Polynomial Curves of degrees 1, 2, 3, 4, and 5 to analyze
    the relationship between an extracted image feature (texture_laplacian_var)
    and target continuous variable (Damage_Score). Evaluates underfitting vs overfitting.
    """
    section_header("1. Polynomial Curve Fitting Analysis")

    feature_col = "texture_laplacian_var"
    target_col = "Damage_Score"

    # Single feature curve fitting
    X_train = train_df[[feature_col]].values
    y_train = train_df[target_col].values
    X_val = val_df[[feature_col]].values
    y_val = val_df[target_col].values
    X_test = test_df[[feature_col]].values
    y_test = test_df[target_col].values

    # Degrees to evaluate
    degrees = [1, 2, 3, 4, 5]
    poly_results = []

    # Prepare dense points for smooth visualization
    x_min = min(X_train.min(), X_test.min())
    x_max = max(X_train.max(), X_test.max())
    X_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot on left
    axes[0].scatter(X_train[:400], y_train[:400], color='gray', alpha=0.35, s=20, label='Train Samples (subset)')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for deg, color in zip(degrees, colors):
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=deg), LinearRegression())
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        poly_results.append({
            "Degree": deg,
            "Train_RMSE": train_rmse,
            "Val_RMSE": val_rmse,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        })

        # Plot curve
        y_plot = model.predict(X_plot)
        axes[0].plot(X_plot, y_plot, color=color, linewidth=2.2, label=f'Degree {deg} (R²={test_r2:.3f})')

    axes[0].set_title(f"Polynomial Fits: {feature_col} vs {target_col}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(feature_col, fontsize=11)
    axes[0].set_ylabel(target_col, fontsize=11)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].legend(loc="upper right", frameon=True)

    # Train vs Val vs Test RMSE Comparison on right
    poly_df = pd.DataFrame(poly_results)
    print("Polynomial Curve Fitting Results:")
    print(poly_df.to_string(index=False))

    axes[1].plot(poly_df["Degree"], poly_df["Train_RMSE"], marker='o', linewidth=2, label="Train RMSE", color='#2b5c8f')
    axes[1].plot(poly_df["Degree"], poly_df["Val_RMSE"], marker='s', linewidth=2, linestyle='--', label="Val RMSE", color='#e67e22')
    axes[1].plot(poly_df["Degree"], poly_df["Test_RMSE"], marker='^', linewidth=2, linestyle=':', label="Test RMSE", color='#27ae60')
    axes[1].set_title("Bias-Variance / Overfitting Analysis by Polynomial Degree", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Polynomial Degree", fontsize=11)
    axes[1].set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    axes[1].set_xticks(degrees)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "1_polynomial_curve_fitting.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved visualization: {plot_path}")

    return poly_df


# ==============================================================================
# 2. REGRESSION ALGORITHMS
# ==============================================================================
def run_regression_algorithms(train_df, val_df, test_df, feature_cols, target_col, output_dir):
    """
    Evaluates multiple regression algorithms predicting Damage_Score from image features:
      1. Linear Regression
      2. Ridge Regression (L2 regularization)
      3. Lasso Regression (L1 regularization)
      4. Polynomial Regression (Degree 2)
      5. Decision Tree Regressor
      6. Random Forest Regressor
    """
    section_header("2. Regression Algorithms Evaluation")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Pre-scale for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    regressors = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Lasso Regression": (Lasso(alpha=0.01), True),
        "Polynomial Reg (Deg 2)": (make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10.0)), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=6, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1), False)
    }

    results = []
    predictions = {}

    for name, (model, requires_scale) in regressors.items():
        X_tr = X_train_scaled if requires_scale else X_train
        X_te = X_test_scaled if requires_scale else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Algorithm": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2
        })

    reg_results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    print("Regression Performance Comparison:")
    print(reg_results_df.to_string(index=False))

    # Visualization: Actual vs Predicted for top models
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (name, _) in enumerate(regressors.items()):
        ax = axes[idx]
        y_pred = predictions[name]
        ax.scatter(y_test[:350], y_pred[:350], alpha=0.4, color='#34495e', edgecolors='none', s=25)
        # Identity line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=1.8, label='Ideal Fit (y = x)')
        
        row = reg_results_df[reg_results_df["Algorithm"] == name].iloc[0]
        ax.set_title(f"{name}\nRMSE: {row['RMSE']:.3f} | R²: {row['R2_Score']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Actual Damage Score", fontsize=10)
        ax.set_ylabel("Predicted Damage Score", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "2_regression_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved regression predictions plot: {plot_path}")

    return reg_results_df


# ==============================================================================
# 3. CLASSIFICATION ALGORITHMS
# ==============================================================================
def run_classification_algorithms(train_df, val_df, test_df, feature_cols, target_col, output_dir):
    """
    Evaluates multiple classification algorithms predicting Condition_Class from image features:
      1. Logistic Regression
      2. Decision Tree Classifier
      3. Random Forest Classifier
      4. Support Vector Machine (RBF Kernel)
      5. K-Nearest Neighbors (KNN)
    """
    section_header("3. Classification Algorithms Evaluation")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])
    class_names = le.classes_

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1), False),
        "Support Vector Machine": (SVC(kernel='rbf', C=1.0, probability=True, random_state=42), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7), True)
    }

    results = []
    confusion_matrices = {}

    for name, (clf, requires_scale) in classifiers.items():
        X_tr = X_train_scaled if requires_scale else X_train
        X_te = X_test_scaled if requires_scale else X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    clf_results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
    print("Classification Performance Comparison:")
    print(clf_results_df.to_string(index=False))

    # Visualization: Confusion Matrices for all classifiers
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (name, _) in enumerate(classifiers.items()):
        ax = axes[idx]
        cm = confusion_matrices[name]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        
        row = clf_results_df[clf_results_df["Algorithm"] == name].iloc[0]
        ax.set_title(f"{name}\nAcc: {row['Accuracy']*100:.1f}% | F1: {row['F1_Score']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Blank out the 6th unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "3_classification_confusion_matrices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices plot: {plot_path}")

    return clf_results_df


# ==============================================================================
# 4. COMPARATIVE PERFORMANCE ANALYSIS
# ==============================================================================
def run_comparative_performance_analysis(reg_results_df, clf_results_df, output_dir):
    """
    Generates side-by-side benchmark comparison bar charts for regression and classification.
    """
    section_header("4. Comparative Performance Analysis")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Regression Comparison: RMSE and R²
    x_indices = np.arange(len(reg_results_df))
    width = 0.35

    axes[0].bar(x_indices - width/2, reg_results_df["RMSE"], width=width, label='RMSE (Lower is Better)', color='#e74c3c')
    axes[0].bar(x_indices + width/2, reg_results_df["R2_Score"], width=width, label='R² Score (Higher is Better)', color='#2980b9')
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(reg_results_df["Algorithm"], rotation=30, ha='right', fontsize=9)
    axes[0].set_title("Regression Benchmark Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Metric Value", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)

    # Classification Comparison: Accuracy and F1-Score
    x_indices_clf = np.arange(len(clf_results_df))
    axes[1].bar(x_indices_clf - width/2, clf_results_df["Accuracy"] * 100, width=width, label='Accuracy %', color='#27ae60')
    axes[1].bar(x_indices_clf + width/2, clf_results_df["F1_Score"] * 100, width=width, label='Weighted F1 %', color='#8e44ad')
    axes[1].set_xticks(x_indices_clf)
    axes[1].set_xticklabels(clf_results_df["Algorithm"], rotation=30, ha='right', fontsize=9)
    axes[1].set_title("Classification Benchmark Comparison", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "4_comparative_performance_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparative benchmark chart: {plot_path}")


# ==============================================================================
# 5. ALGORITHM EXPLANATION GENERATOR
# ==============================================================================
def generate_algorithm_explanations(output_dir):
    """
    Creates a comprehensive Markdown explanation document detailing all algorithms used,
    their mathematical formulations, trade-offs, and practical demolition use-case fit.
    """
    section_header("5. Generating Detailed Algorithm Explanation")

    explanation_content = """# Machine Learning Algorithm Explanations & Performance Report
**Project:** Demolition Material Image Features Analysis (PBL 1)  
**Dataset:** Training, Validation, and Testing Computer Vision Extracted Feature Datasets  

---

## 1. Polynomial Curve Fitting

### Mathematical Formulation
Polynomial curve fitting models a non-linear relationship between an independent variable $x$ and a dependent variable $y$ by an $n$-th degree polynomial:
$$\\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \\dots + w_d x^d = \\sum_{i=0}^{d} w_i x^i$$

### Underfitting vs. Overfitting (Bias-Variance Tradeoff)
1. **Degree 1 (Linear - High Bias / Underfitting):**
   - Assumes a strictly linear relationship. Fails to capture localized saturation in texture variance and non-linear transitions.
2. **Degrees 2 & 3 (Optimal Balance):**
   - Introduces parabolic and cubic curvatures that capture rapid degradation followed by plateauing damage metrics, maintaining generalizability on unseen test data.
3. **Degree 4 & 5 (High Variance / Overfitting):**
   - Oscillates between peripheral data points. May improve training error marginally while destabilizing predictions on test extremes (Runge's phenomenon).

---

## 2. Regression Algorithms

### 2.1 Linear Regression (Ordinary Least Squares)
- **Concept:** Minimizes the residual sum of squares:
  $$\\min_{w} \\|y - Xw\\|_2^2$$
- **Strengths:** Highly interpretable coefficients, zero hyperparameter tuning, fast computation.
- **Limitations:** Susceptible to multicollinearity among correlated visual features (e.g., width, perimeter, area).

### 2.2 Ridge Regression (L2 Regularization)
- **Concept:** Penalizes large weight values using the Euclidean norm:
  $$\\min_{w} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_2^2$$
- **Strengths:** Mitigates multicollinearity and dampens variance across correlated image shape and color features.

### 2.3 Lasso Regression (L1 Regularization)
- **Concept:** Penalizes absolute parameter magnitudes:
  $$\\min_{w} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_1$$
- **Strengths:** Acts as an automatic feature selector by driving uninformative feature weights to absolute zero.

### 2.4 Polynomial Regression (Degree 2 with L2 Regularization)
- **Concept:** Expands the feature space to include all pairwise interaction terms ($x_i x_j$) and quadratic powers ($x_i^2$).
- **Strengths:** Captures synergistic visual effects (e.g., interaction between edge density and Laplacian texture variance).

### 2.5 Decision Tree Regressor
- **Concept:** Recursively partitions the feature space into orthogonal axis-aligned rectangles by greedily minimizing variance (mean squared error).
- **Strengths:** Non-parametric, requires no feature scaling, captures non-linear step thresholds.
- **Limitations:** High variance; prone to over-segmenting continuous target values.

### 2.6 Random Forest Regressor
- **Concept:** An ensemble of $B$ bootstrap-aggregated (bagged) de-correlated decision trees with random feature subsampling:
  $$\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} T_b(X)$$
- **Strengths:** Drastically lowers variance without increasing bias, resilient to noise and outliers, provides robust feature importance scores.

---

## 3. Classification Algorithms

### 3.1 Logistic Regression
- **Concept:** Maps a linear combination of inputs through the sigmoid/softmax function:
  $$P(Y=k|X) = \\frac{e^{w_k^T X}}{\\sum_j e^{w_j^T X}}$$
- **Strengths:** Outputs well-calibrated class probabilities, fast training, interpretable odds ratios.
- **Limitations:** Linear decision boundaries struggle when classes have complex multi-modal overlap.

### 3.2 Decision Tree Classifier
- **Concept:** Partitions data using Gini impurity or Information Entropy:
  $$I_G(p) = 1 - \\sum_{k=1}^K p_k^2$$
- **Strengths:** Generates white-box if-then classification rules matching building inspection workflows.

### 3.3 Random Forest Classifier
- **Concept:** Aggregates majority votes from hundreds of randomly perturbed decision trees.
- **Strengths:** Excels on tabular vision features, highly robust to class imbalances and noisy labels.

### 3.4 Support Vector Machine (SVM with RBF Kernel)
- **Concept:** Maps samples into an infinite-dimensional reproducing kernel Hilbert space using the radial basis function:
  $$K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$$
  Maximizes the geometric margin between support vectors of different classes.
- **Strengths:** Highly effective in high-dimensional non-linear feature spaces.

### 3.5 K-Nearest Neighbors (KNN)
- **Concept:** Instance-based non-parametric classifier that assigns the majority label among the $k$ closest training points in Euclidean feature space.
- **Strengths:** Simple, naturally handles multi-class problems.
- **Limitations:** Sensitive to the curse of dimensionality and irrelevant visual features.

---

## 4. Key Engineering Insights
1. **Normalization is Mandatory:** Distance-based algorithms (SVM, KNN, Ridge, Lasso) require `StandardScaler` to prevent high-magnitude features (like `contour_area`) from dominating smaller scale features (like `relative_area`).
2. **Ensemble Superiority:** Tree ensembles (Random Forest) demonstrate the most balanced trade-off between bias and variance on both regression and multi-class classification tasks.
"""
    doc_path = os.path.join(output_dir, "ALGORITHM_EXPLANATION.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(explanation_content)
    print(f"[+] Generated markdown documentation: {doc_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    # Identify directories
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_script_dir)
    output_dir = current_script_dir

    print(f"[*] PBL 1 Execution initiated.")
    print(f"    Project Directory: {project_dir}")
    print(f"    Output Directory:  {output_dir}")

    # Load datasets
    train_df, val_df, test_df = load_data(project_dir)

    # Selected extracted visual features
    feature_cols = [
        'pixel_obj_width', 'pixel_obj_height', 'relative_area', 'edge_density',
        'fill_ratio', 'color_std', 'mean_brightness', 'std_brightness', 'contrast',
        'texture_laplacian_var', 'dark_pixel_ratio', 'bright_pixel_ratio', 'mean_hue',
        'mean_saturation', 'saturation_std', 'contour_area', 'contour_perimeter',
        'circularity', 'solidity', 'extent'
    ]

    # Task 1: Polynomial Curve Fitting
    poly_df = run_polynomial_curve_fitting(train_df, val_df, test_df, output_dir)

    # Task 2: Regression Algorithms (Predicting Damage_Score)
    reg_results_df = run_regression_algorithms(train_df, val_df, test_df, feature_cols, "Damage_Score", output_dir)

    # Task 3: Classification Algorithms (Predicting Condition_Class)
    clf_results_df = run_classification_algorithms(train_df, val_df, test_df, feature_cols, "Condition_Class", output_dir)

    # Task 4: Comparative Performance Analysis
    run_comparative_performance_analysis(reg_results_df, clf_results_df, output_dir)

    # Task 5: Algorithm Explanations
    generate_algorithm_explanations(output_dir)

    # Save summary tables as CSV
    poly_df.to_csv(os.path.join(output_dir, "summary_polynomial_fitting.csv"), index=False)
    reg_results_df.to_csv(os.path.join(output_dir, "summary_regression_metrics.csv"), index=False)
    clf_results_df.to_csv(os.path.join(output_dir, "summary_classification_metrics.csv"), index=False)

    print("\n[SUCCESS] All PBL 1 tasks completed successfully! All charts and reports generated in pbl1/ folder.\n")


if __name__ == "__main__":
    main()
