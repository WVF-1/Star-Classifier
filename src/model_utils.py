# src/model_utils.py

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_models(random_state=42):
    """
    Return dictionary of baseline models.
    """
    return {
        "logistic_regression": LogisticRegression(
            multi_class="multinomial", max_iter=1000
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
        "svm": SVC(kernel="rbf", probability=True)
    }
