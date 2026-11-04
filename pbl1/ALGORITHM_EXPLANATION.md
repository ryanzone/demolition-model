# Machine Learning Algorithm Explanations & Performance Report
**Project:** Demolition Material Image Features Analysis (PBL 1)  
**Dataset:** Training, Validation, and Testing Computer Vision Extracted Feature Datasets  

---

## 1. Polynomial Curve Fitting

### Mathematical Formulation
Polynomial curve fitting models a non-linear relationship between an independent variable $x$ and a dependent variable $y$ by an $n$-th degree polynomial:
$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_d x^d = \sum_{i=0}^{d} w_i x^i$$

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
  $$\min_{w} \|y - Xw\|_2^2$$
- **Strengths:** Highly interpretable coefficients, zero hyperparameter tuning, fast computation.
- **Limitations:** Susceptible to multicollinearity among correlated visual features (e.g., width, perimeter, area).

### 2.2 Ridge Regression (L2 Regularization)
- **Concept:** Penalizes large weight values using the Euclidean norm:
  $$\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$$
- **Strengths:** Mitigates multicollinearity and dampens variance across correlated image shape and color features.

### 2.3 Lasso Regression (L1 Regularization)
- **Concept:** Penalizes absolute parameter magnitudes:
  $$\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_1$$
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
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)$$
- **Strengths:** Drastically lowers variance without increasing bias, resilient to noise and outliers, provides robust feature importance scores.

---

## 3. Classification Algorithms

### 3.1 Logistic Regression
- **Concept:** Maps a linear combination of inputs through the sigmoid/softmax function:
  $$P(Y=k|X) = \frac{e^{w_k^T X}}{\sum_j e^{w_j^T X}}$$
- **Strengths:** Outputs well-calibrated class probabilities, fast training, interpretable odds ratios.
- **Limitations:** Linear decision boundaries struggle when classes have complex multi-modal overlap.

### 3.2 Decision Tree Classifier
- **Concept:** Partitions data using Gini impurity or Information Entropy:
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
- **Strengths:** Generates white-box if-then classification rules matching building inspection workflows.

### 3.3 Random Forest Classifier
- **Concept:** Aggregates majority votes from hundreds of randomly perturbed decision trees.
- **Strengths:** Excels on tabular vision features, highly robust to class imbalances and noisy labels.

### 3.4 Support Vector Machine (SVM with RBF Kernel)
- **Concept:** Maps samples into an infinite-dimensional reproducing kernel Hilbert space using the radial basis function:
  $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
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
