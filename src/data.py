"""Reproducible datasets used by the teaching app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


FRUIT_FEATURES = ["weight_g", "redness", "height_mm", "diameter_mm"]
FRUIT_LABELS = {
    "weight_g": "Weight (g)",
    "redness": "Redness (0–1)",
    "height_mm": "Height (mm)",
    "diameter_mm": "Middle diameter (mm)",
}


@dataclass(frozen=True)
class Dataset:
    """A labelled binary dataset together with presentation metadata."""

    X: pd.DataFrame
    y: pd.Series
    class_names: dict[int, str]
    display_names: dict[str, str]


def make_fruit_dataset(n_samples: int, overlap: float, seed: int) -> Dataset:
    """Create plausible, slightly overlapping apple and pear measurements.

    ``overlap`` increases within-class variation and gently pulls means together,
    making soft-margin behaviour easy to observe without implausible fruit values.
    """
    rng = np.random.default_rng(seed)
    n_apple = n_samples // 2
    n_pear = n_samples - n_apple
    # weight, redness, height, diameter
    apple_mean = np.array([182.0, 0.72, 76.0, 79.0])
    pear_mean = np.array([154.0, 0.47, 106.0, 57.0])
    midpoint = (apple_mean + pear_mean) / 2
    mean_mix = min(max(overlap, 0.0), 1.0) * 0.22
    apple_mean = apple_mean * (1 - mean_mix) + midpoint * mean_mix
    pear_mean = pear_mean * (1 - mean_mix) + midpoint * mean_mix
    base_std = np.array([13.0, 0.075, 5.5, 4.5])
    std = base_std * (1 + 2.8 * overlap)

    apples = rng.normal(apple_mean, std, size=(n_apple, 4))
    pears = rng.normal(pear_mean, std, size=(n_pear, 4))
    X = pd.DataFrame(np.vstack([apples, pears]), columns=FRUIT_FEATURES)
    X["weight_g"] = X["weight_g"].clip(70, 300)
    X["redness"] = X["redness"].clip(0, 1)
    X["height_mm"] = X["height_mm"].clip(35, 150)
    X["diameter_mm"] = X["diameter_mm"].clip(25, 130)
    y = pd.Series([0] * n_apple + [1] * n_pear, name="class")
    return Dataset(X, y, {0: "Apple", 1: "Pear"}, FRUIT_LABELS)


def make_iris_binary_dataset(n_samples: int, seed: int) -> Dataset:
    """Return a reproducibly sampled Setosa-versus-Versicolor Iris dataset."""
    iris = load_iris(as_frame=True)
    mask = iris.target < 2
    X = iris.data.loc[mask].copy()
    y = iris.target.loc[mask].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(X), size=n_samples, replace=False)
        X, y = X.iloc[selected].reset_index(drop=True), y.iloc[selected].reset_index(drop=True)
    names = {name: name.replace(" (cm)", " (cm)").title() for name in X.columns}
    return Dataset(X, y, {0: "Setosa", 1: "Versicolor"}, names)
