# src/data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Standard train-validation split.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
