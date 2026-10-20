"""SVM training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class TrainingResult:
    """Everything the UI needs from one reproducible SVM training run."""

    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray
    training_ms: float
    feature_names: list[str]
    config: dict[str, Any]

    @property
    def svc(self) -> SVC:
        return self.pipeline.named_steps["svc"]

    @property
    def support_vector_count(self) -> int:
        return int(self.svc.n_support_.sum())


def train_svm(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    test_size: float,
    seed: int,
    config: dict[str, Any],
) -> TrainingResult:
    """Split data, fit an SVC pipeline, and return transparent evaluation data."""
    selected = X.loc[:, feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        selected, y, test_size=test_size, random_state=seed, stratify=y
    )
    svc_args: dict[str, Any] = {
        "C": config["C"],
        "kernel": config["kernel"],
        "gamma": config["gamma"],
        "degree": config["degree"],
        "coef0": config["coef0"],
        "class_weight": config["class_weight"],
    }
    steps: list[tuple[str, Any]] = []
    if config["scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svc", SVC(**svc_args)))
    pipeline = Pipeline(steps)
    started = perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = (perf_counter() - started) * 1000
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    return TrainingResult(
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        confusion=confusion_matrix(y_test, predictions, labels=[0, 1]),
        training_ms=elapsed,
        feature_names=feature_names,
        config=config.copy(),
    )


def raw_linear_parameters(result: TrainingResult) -> tuple[np.ndarray, float]:
    """Return linear coefficients in the original (unscaled) feature units."""
    if result.svc.kernel != "linear":
        raise ValueError("Raw parameters only exist for a linear SVM.")
    weights = result.svc.coef_[0].astype(float).copy()
    intercept = float(result.svc.intercept_[0])
    scaler = result.pipeline.named_steps.get("scaler")
    if scaler is not None:
        intercept -= float(np.dot(weights, scaler.mean_ / scaler.scale_))
        weights /= scaler.scale_
    return weights, intercept
