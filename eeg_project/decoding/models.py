from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..results import EvalResult


def classical_models(random_state: int) -> dict[str, Pipeline]:
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    return {
        "logistic_regression": Pipeline(
            [
                ("csp", csp),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "svm": Pipeline(
            [
                ("csp", CSP(n_components=8, reg=None, log=True, norm_trace=False)),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("csp", CSP(n_components=8, reg=None, log=True, norm_trace=False)),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ]
        ),
    }


def split_subject(X: np.ndarray, y: np.ndarray, random_state: int, test_size: float) -> tuple[np.ndarray, ...]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate_classical_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_names: list[str],
    random_state: int = 42,
    test_size: float = 0.25,
) -> list[EvalResult]:
    available = classical_models(random_state)
    results: list[EvalResult] = []
    for subject in sorted(set(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        X_train, X_test, y_train, y_test = split_subject(X[mask], y[mask], random_state, test_size)
        for model_name in model_names:
            model = available[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(
                EvalResult(
                    model=model_name,
                    subject=subject,
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    macro_f1=float(f1_score(y_test, y_pred, average="macro")),
                    confusion=confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]),
                )
            )
    return results

