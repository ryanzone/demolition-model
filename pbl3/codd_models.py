"""Importable model classes for CODD PBL3.

MixtureOfExperts lives in a module (not inside a notebook) so that joblib
files saved from notebooks can be loaded from any script, including the
stacking classifier.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class MixtureOfExperts(ClassifierMixin, BaseEstimator):
    """Soft-gated mixture of per-material Random Forest experts.

    Same architecture as the original notebook version: a logistic-regression
    gate predicts the material from the remaining features, one Random Forest
    expert is trained per material, and expert probabilities are combined with
    the gate probabilities. X is the full 30-feature frame. The gate column is
    used as the gate label in fit() and is removed from the features the gate
    and experts see.
    """

    def __init__(self, preprocessor, gate_col="Material_Type",
                 min_samples=30, random_state=42):
        self.preprocessor = preprocessor
        self.gate_col = gate_col
        self.min_samples = min_samples
        self.random_state = random_state

    def _features(self, X):
        return X.drop(columns=[self.gate_col])

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        material = X[self.gate_col].fillna("Unknown").astype(str)
        feats = self._features(X)

        self.classes_ = np.unique(y)
        self.gate_ = Pipeline([
            ("prep", clone(self.preprocessor)),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]).fit(feats, material)

        self.experts_ = {}
        for material_type in material.unique():
            mask = (material == material_type).to_numpy()
            if mask.sum() < self.min_samples or y[mask].nunique() < 2:
                continue
            expert = Pipeline([
                ("prep", clone(self.preprocessor)),
                ("model", RandomForestClassifier(
                    n_estimators=150, max_depth=15, min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=self.random_state, n_jobs=-1)),
            ])
            expert.fit(feats[mask], y[mask])
            self.experts_[material_type] = expert

        if not self.experts_:
            raise ValueError("No expert models were trained. Check material groups and min_samples.")

        counts = y.value_counts()
        prior = np.array([counts.get(c, 0) for c in self.classes_], dtype=float)
        self.fallback_ = prior / prior.sum()
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        feats = self._features(X)
        gate_probs = self.gate_.predict_proba(feats)
        gate_classes = self.gate_.named_steps["model"].classes_

        probs = np.zeros((len(feats), len(self.classes_)))
        for j, material_type in enumerate(gate_classes):
            expert = self.experts_.get(material_type)
            if expert is None:
                continue
            expert_probs = expert.predict_proba(feats)
            for k, label in enumerate(expert.named_steps["model"].classes_):
                idx = np.where(self.classes_ == label)[0][0]
                probs[:, idx] += gate_probs[:, j] * expert_probs[:, k]

        sums = probs.sum(axis=1, keepdims=True)
        probs = np.divide(probs, sums, out=np.zeros_like(probs), where=sums != 0)
        missing = sums.ravel() == 0
        if missing.any():
            probs[missing] = self.fallback_
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
