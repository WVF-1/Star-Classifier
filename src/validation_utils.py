# src/validation_utils.py

import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score


def run_loocv(model, X, y, scoring="accuracy"):
    """
    Run Leave-One-Out Cross Validation.
    """
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)
    return scores.mean(), scores.std()


def run_kfold(model, X, y, k=5, scoring="accuracy"):
    """
    Run k-Fold Cross Validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    return scores.mean(), scores.std()
