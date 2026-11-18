"""Chapter 3 graphs for PBL2 (Units 3 and 4), saved as PNG in pbl2/graphs/.

Uses the same data, features and settings as the PBL2 notebooks
(grouped splits, 26 numeric features, k = 4, GMM 4 diagonal components,
Ward hierarchical clustering, HMM/Viterbi on binned Damage_Score).

  fig_3_1_kmeans_elbow_silhouette.png   elbow + silhouette for k = 2..10
  fig_3_1_kmeans_clusters.png           K-means clusters in PCA space + condition mix per cluster
  fig_3_1_kmeans_workflow.png           K-means workflow diagram
  fig_3_2_gmm_membership.png            GMM membership probability histogram
  fig_3_2_gmm_aic_bic.png               AIC / BIC for 1..10 components
  fig_3_3_dendrogram.png                Ward dendrogram
  fig_3_4_pca_variance.png              PCA scree + cumulative explained variance
  fig_3_5_hmm_viterbi_workflow.png      HMM / Viterbi workflow diagram
  fig_3_6_hmm_viterbi_confusion.png     HMM/Viterbi confusion matrix
  fig_3_6_method_comparison.png         accuracy / macro F1: all Unit 3 and 4 methods

Run:  python make_graphs.py   (from the pbl2 folder)
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             pairwise_distances_argmin, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from pbl2_common import ORIGINAL_FEATURES, STATES, TARGET, load_train_test

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "graphs"
OUT.mkdir(exist_ok=True)
SEED = 42
K = 4
CLASS_COLORS = ["#7b3fa0", "#2a6fb0", "#3a9d5d", "#d1495b"]   # Excellent, Good, Fair, Poor
BLUE = "#2a6fb0"
ORANGE = "#e07b00"

train, test = load_train_test()
y_train, y_test = train[TARGET].astype(str), test[TARGET].astype(str)

# ---- Preprocessing (fit on train only), as in the notebooks
imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(train[ORIGINAL_FEATURES].replace([np.inf, -np.inf], np.nan)))
X_test = scaler.transform(imputer.transform(test[ORIGINAL_FEATURES].replace([np.inf, -np.inf], np.nan)))
sample_idx = np.random.RandomState(SEED).choice(len(X_train), 3000, replace=False)


def save(fig, name):
    """Save a figure at slide/report resolution and close it."""
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def map_to_class(train_clusters, test_clusters):
    """Map clusters to their majority training class and return test predictions."""
    mapping = {c: y_train[train_clusters == c].mode().iloc[0] if (train_clusters == c).any()
               else y_train.mode().iloc[0] for c in range(K)}
    return np.array([mapping[c] for c in test_clusters])


# ======================================================= Figure 3.1  K-means
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_train)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_train, km.labels_, sample_size=3000, random_state=SEED))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(ks), inertia, "o-", color=BLUE)
axes[0].axvline(K, color=ORANGE, ls="--", label=f"k = {K} (used)")
axes[0].set_title("Elbow method (inertia)")
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(ks), sil, "o-", color=BLUE)
axes[1].axvline(K, color=ORANGE, ls="--", label=f"k = {K} (used)")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend()
save(fig, "fig_3_1_kmeans_elbow_silhouette.png")

kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X_train)
train_km, test_km = kmeans.labels_, kmeans.predict(X_test)
pca2 = PCA(n_components=2).fit(X_train)
Z = pca2.transform(X_train[sample_idx])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=train_km[sample_idx], cmap="viridis", s=10, alpha=0.7)
axes[0].set_title(f"K-means clusters (k = {K}) in PCA space")
axes[0].set_xlabel(f"PC1 ({100 * pca2.explained_variance_ratio_[0]:.1f}% variance)")
axes[0].set_ylabel(f"PC2 ({100 * pca2.explained_variance_ratio_[1]:.1f}% variance)")
fig.colorbar(sc, ax=axes[0], label="Cluster")
mix = pd.crosstab(pd.Series(train_km, name="Cluster"), y_train).reindex(columns=STATES, fill_value=0)
(mix.div(mix.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=axes[1], color=CLASS_COLORS, width=0.7)
axes[1].set_title("Condition mix inside each cluster (%)")
axes[1].set_ylabel("Share of items (%)")
axes[1].set_xlabel("Cluster")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "fig_3_1_kmeans_clusters.png")

# ========================================================= Figure 3.2  GMM
gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=SEED,
                      n_init=1, max_iter=100, reg_covar=1e-5).fit(X_train[sample_idx])
max_prob = gmm.predict_proba(X_train[sample_idx]).max(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(max_prob, bins=20, color=BLUE, edgecolor="white")
ax.set_title("GMM: membership probability of the most likely component")
ax.set_xlabel("Maximum membership probability")
ax.set_ylabel("Number of samples")
ax.grid(axis="y", alpha=0.3)
save(fig, "fig_3_2_gmm_membership.png")

aic, bic = [], []
for n in range(1, 11):
    g = GaussianMixture(n_components=n, covariance_type="diag", random_state=SEED,
                        n_init=1, max_iter=100, reg_covar=1e-5).fit(X_train)
    aic.append(g.aic(X_train))
    bic.append(g.bic(X_train))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), aic, "o-", color=BLUE, label="AIC")
ax.plot(range(1, 11), bic, "s-", color=ORANGE, label="BIC")
ax.axvline(K, color="gray", ls="--", label=f"{K} components (used)")
ax.set_title("GMM model selection: AIC and BIC")
ax.set_xlabel("Number of components")
ax.set_ylabel("Information criterion (lower is better)")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "fig_3_2_gmm_aic_bic.png")

# =============================================== Figure 3.3  Hierarchical
link = sch.linkage(X_train[:150], method="ward")
fig, ax = plt.subplots(figsize=(13, 6))
sch.dendrogram(link, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9,
               show_contracted=True, ax=ax, color_threshold=0.6 * link[:, 2].max())
ax.set_title("Hierarchical clustering dendrogram (Ward linkage, 150-sample subset)")
ax.set_xlabel("Cluster / sample")
ax.set_ylabel("Euclidean distance")
save(fig, "fig_3_3_dendrogram.png")

# ================================================================ Figure 3.4
pca_full = PCA().fit(X_train)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)
n95 = int(np.argmax(cum >= 0.95) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar(range(1, len(ratio) + 1), ratio, color=BLUE)
axes[0].set_title("Scree plot: variance explained per component")
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance ratio")
axes[1].plot(range(1, len(cum) + 1), cum, "o-", color=BLUE)
axes[1].axhline(0.95, color=ORANGE, ls="--", label="95% threshold")
axes[1].axvline(n95, color="gray", ls=":", label=f"{n95} components")
axes[1].set_title("Cumulative explained variance")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
save(fig, "fig_3_4_pca_variance.png")

# =================================================== Figure 3.6  HMM / Viterbi
def sequences(df):
    """Yield each image's objects ordered by sample number (as in the notebooks)."""
    num = pd.to_numeric(df["Sample_ID"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce")
    d = df.assign(_n=num.values)
    for _, g in d.groupby("Image_ID", sort=True):
        yield g.sort_values("_n")


dmg = pd.to_numeric(train["Damage_Score"], errors="coerce")
med = dmg.median()
cuts = np.unique(np.quantile(dmg.fillna(med), [0.25, 0.5, 0.75]))
n_sym = len(cuts) + 1
obs_of = lambda g: np.digitize(pd.to_numeric(g["Damage_Score"], errors="coerce").fillna(med), cuts)
idx = {s: i for i, s in enumerate(STATES)}
S, alpha = len(STATES), 0.5
start, trans, emit = np.full(S, alpha), np.full((S, S), alpha), np.full((S, n_sym), alpha)
for g in sequences(train):
    st, ob = [idx[s] for s in g[TARGET].astype(str)], obs_of(g)
    start[st[0]] += 1
    for a, b in zip(st[:-1], st[1:]):
        trans[a, b] += 1
    for s_, o_ in zip(st, ob):
        emit[s_, o_] += 1
start /= start.sum()
trans /= trans.sum(axis=1, keepdims=True)
emit /= emit.sum(axis=1, keepdims=True)


def viterbi(obs):
    """Most likely state path for one observation sequence (log space)."""
    ls, lt, le = np.log(start), np.log(trans), np.log(emit)
    n = len(obs)
    dp, back = np.full((n, S), -np.inf), np.zeros((n, S), int)
    dp[0] = ls + le[:, obs[0]]
    for t in range(1, n):
        sc_ = dp[t - 1][:, None] + lt
        back[t] = sc_.argmax(axis=0)
        dp[t] = sc_.max(axis=0) + le[:, obs[t]]
    path = np.zeros(n, int)
    path[-1] = dp[-1].argmax()
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return np.array(STATES)[path]


y_true, y_pred = [], []
for g in sequences(test):
    y_true += g[TARGET].astype(str).tolist()
    y_pred += viterbi(obs_of(g)).tolist()
hmm_acc = accuracy_score(y_true, y_pred)
hmm_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"HMM/Viterbi accuracy {hmm_acc:.4f}, macro F1 {hmm_f1:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=STATES)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
            cbar=False, xticklabels=STATES, yticklabels=STATES, ax=ax, linewidths=0.5, linecolor="white")
ax.set_title(f"HMM / Viterbi - test confusion matrix (accuracy {hmm_acc:.3f})\n"
             "colour = share of each actual class, numbers = items")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
save(fig, "fig_3_6_hmm_viterbi_confusion.png")

# ---- comparison of all Unit 3 + 4 methods (clusters mapped to majority class)
rng = np.random.RandomState(SEED)
sub = rng.choice(len(X_train), 2500, replace=False)
agg = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X_train[sub])
cent = np.vstack([X_train[sub][agg == c].mean(axis=0) if (agg == c).any() else X_train[sub].mean(axis=0)
                  for c in range(K)])
gm = GaussianMixture(n_components=K, covariance_type="diag", random_state=SEED, n_init=3,
                     max_iter=200, reg_covar=1e-6).fit(X_train)
pca95 = PCA(n_components=0.95).fit(X_train)
km_p = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(pca95.transform(X_train))
methods = {
    "K-means": map_to_class(train_km, test_km),
    "GMM": map_to_class(gm.predict(X_train), gm.predict(X_test)),
    "Hierarchical": map_to_class(pairwise_distances_argmin(X_train, cent), pairwise_distances_argmin(X_test, cent)),
    "PCA + K-means": map_to_class(km_p.labels_, km_p.predict(pca95.transform(X_test))),
    "HMM / Viterbi": np.array(y_pred),
}
truth = {m: (np.array(y_true) if m == "HMM / Viterbi" else y_test.values) for m in methods}
names = list(methods)
accs = [accuracy_score(truth[m], methods[m]) for m in names]
f1s = [f1_score(truth[m], methods[m], average="macro", zero_division=0) for m in names]
baseline = y_test.value_counts(normalize=True).max()
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(names))
b1 = ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color=BLUE)
b2 = ax.bar(x + 0.2, f1s, 0.4, label="Macro F1", color=ORANGE)
ax.axhline(baseline, color="gray", ls="--", label=f"Majority-class baseline ({baseline:.3f})")
for r in list(b1) + list(b2):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.08)
ax.set_title("Unit 3 and 4 methods on the test split")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc="upper left")
save(fig, "fig_3_6_method_comparison.png")


# ========================================================= workflow diagrams
def workflow(title, steps, filename, colors=None):
    """Draw a left-to-right boxed workflow diagram."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(2.3 * n, 3.2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, text in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.08, 0.55), 0.84, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#e8f0fa", ec=BLUE, lw=1.6))
        ax.text(i + 0.5, 1.0, text, ha="center", va="center", fontsize=9.5, wrap=True)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((i + 0.93, 1.0), (i + 1.07, 1.0), arrowstyle="-|>",
                                         mutation_scale=16, color="#444"))
    ax.set_title(title, fontsize=13, pad=4)
    save(fig, filename)


workflow("K-means clustering workflow",
         ["Image-derived\nfeature table", "Impute +\nstandardise\n(train only)", "Choose k\n(elbow,\nsilhouette)",
          "Fit K-means\n(k = 4)", "Assign test\nitems to\nnearest centroid", "Interpret clusters\nvs condition\nlabels"],
         "fig_3_1_kmeans_workflow.png")
workflow("HMM and Viterbi workflow",
         ["Items grouped by\nImage_ID, ordered\nby sample no.", "Observation:\nbinned\nDamage_Score",
          "Estimate start,\ntransition,\nemission probs", "Forward /\nbackward\nlikelihood",
          "Viterbi\ndecoding", "Evaluate state\nsequence (confusion\nmatrix, F1)"],
         "fig_3_5_hmm_viterbi_workflow.png")
print("done ->", OUT)
