# src/learning_curve_utils.py

import numpy as np
from sklearn.model_selection import learning_curve, KFold, LeaveOneOut


def compute_learning_curve(
    model,
    X,
    y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv_type="kfold",
    k=5,
    scoring="accuracy"
):
    """
    Compute learning curves for a given model.

    Parameters
    ----------
    model : sklearn estimator
    X : array-like
    y : array-like
    train_sizes : array-like
        Fractions or absolute numbers of training samples.
    cv_type : str
        'kfold' or 'loocv'
    k : int
        Number of folds (used if cv_type='kfold')
    scoring : str
        Scoring metric

    Returns
    -------
    train_sizes_abs : np.ndarray
    train_scores_mean : np.ndarray
    val_scores_mean : np.ndarray
    """

    if cv_type == "kfold":
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_type == "loocv":
        cv = LeaveOneOut()
    else:
        raise ValueError("cv_type must be 'kfold' or 'loocv'")

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return (
        train_sizes_abs,
        train_scores.mean(axis=1),
        val_scores.mean(axis=1)
    )
