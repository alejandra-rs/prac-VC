from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class ThresholdRules(ClassifierMixin, BaseEstimator):
    def __init__(self, candidates=("perclos", "blink_mean", "long_blinks"), quantiles=20):
        self.candidates = candidates
        self.quantiles = quantiles

    def classify(self, values, thresholds):
        return self.classes_[np.searchsorted(thresholds, values, side="right")]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        best = -1
        for feature in self.candidates:
            values = X[feature].to_numpy()
            cuts = np.unique(np.quantile(values, np.linspace(0, 1, self.quantiles + 1)[1:-1]))
            for thresholds in combinations(cuts, len(self.classes_) - 1):
                score = f1_score(y, self.classify(values, thresholds), average="macro")
                if score > best:
                    best, self.feature_, self.thresholds_ = score, feature, np.array(thresholds)
        return self

    def predict(self, X):
        return self.classify(X[self.feature_].to_numpy(), self.thresholds_)


def classical_models(cfg, seed):
    return {
        "Reglas": (ThresholdRules(quantiles=cfg.rule_quantiles), {}),
        "SVM": (make_pipeline(MinMaxScaler(), SVC(class_weight="balanced")), vars(cfg.svm)),
        "Random Forest": (RandomForestClassifier(300, class_weight="balanced", random_state=seed),
                          vars(cfg.random_forest)),
        "Boosting": (HistGradientBoostingClassifier(class_weight="balanced", random_state=seed), vars(cfg.boosting)),
    }


def fit_tuned(model, grid, X, y, groups):
    if not grid:
        return clone(model).fit(X, y)
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=GroupKFold(len(np.unique(groups))), n_jobs=-1)
    return search.fit(X, y, groups=groups).best_estimator_
