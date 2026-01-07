# src/feature_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# Ordinal mappings (domain-aware)
# ---------------------------------------

STAR_COLOR_ORDER = {
    "blue": 0,
    "blue white": 1,
    "whitish": 2,
    "white": 3,
    "white-yellow": 4,
    "yellowish white": 5,
    "yellow-white": 6,
    "yellowish": 7,
    "pale yellow orange": 8,
    "orange": 9,
    "orange-red": 10,
    "red": 11
}

SPECTRAL_CLASS_ORDER = {
    "O": 0,
    "B": 1,
    "A": 2,
    "F": 3,
    "G": 4,
    "K": 5,
    "M": 6
}


# ---------------------------------------
# Feature engineering helpers
# ---------------------------------------

def log_transform(df, columns):
    """
    Apply log10 transform to selected columns.
    """
    df = df.copy()
    for col in columns:
        df[col] = np.log10(df[col] + 1e-9)
    return df


def normalize_star_color(value: str) -> str:
    """
    Normalize star color strings to canonical form.
    """
    value = value.strip().lower()
    value = value.replace("-", " ")
    value = " ".join(value.split())

    CANONICAL_MAP = {
        "blue": "blue",
        "blue white": "blue white",
        "bluewhite": "blue white",
        "white": "white",
        "whitish": "whitish",
        "white yellow": "white-yellow",
        "yellow white": "yellow-white",
        "yellowish white": "yellowish white",
        "yellowish": "yellowish",
        "pale yellow orange": "pale yellow orange",
        "orange": "orange",
        "orange red": "orange-red",
        "red": "red"
    }

    return CANONICAL_MAP.get(value, value)


def ordinal_encode(df):
    """
    Ordinal encode star color and spectral class.
    """
    df = df.copy()

    df["Star color"] = (
        df["Star color"]
        .astype(str)
        .apply(normalize_star_color)
        .map(STAR_COLOR_ORDER)
    )

    df["Spectral Class"] = df["Spectral Class"].map(SPECTRAL_CLASS_ORDER)

    if df[["Star color", "Spectral Class"]].isnull().any().any():
        raise ValueError(
            "Ordinal encoding failed: unexpected category encountered."
        )

    return df


def scale_features(X):
    """
    Standardize features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
