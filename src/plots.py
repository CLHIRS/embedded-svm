"""Plotly charts for data, geometry, and saved experiment trade-offs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import Dataset
from .model import TrainingResult


CLASS_COLORS = {0: "#d95f02", 1: "#1b9e77"}


def data_scatter(dataset: Dataset, x_feature: str, y_feature: str) -> go.Figure:
    """Render the raw two-dimensional teaching view."""
    frame = dataset.X.copy()
    frame["Class"] = dataset.y.map(dataset.class_names)
    figure = px.scatter(
        frame, x=x_feature, y=y_feature, color="Class",
        color_discrete_sequence=["#d95f02", "#1b9e77"],
        labels={x_feature: dataset.display_names[x_feature], y_feature: dataset.display_names[y_feature]},
        title="Where would you draw the separating line?",
    )
    figure.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "white"}})
    figure.update_layout(legend_title_text="Fruit", margin=dict(l=10, r=10, t=45, b=10))
    return figure


def svm_geometry_plot(
    dataset: Dataset, result: TrainingResult, x_feature: str, y_feature: str
) -> go.Figure:
    """Plot decision regions, margins, support vectors, and test mistakes in 2D."""
    figure = data_scatter(dataset, x_feature, y_feature)
    x_all = dataset.X[x_feature]
    y_all = dataset.X[y_feature]
    pad_x, pad_y = max(x_all.max() - x_all.min(), 1) * 0.12, max(y_all.max() - y_all.min(), 1) * 0.12
    xs = np.linspace(x_all.min() - pad_x, x_all.max() + pad_x, 160)
    ys = np.linspace(y_all.min() - pad_y, y_all.max() + pad_y, 160)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = pd.DataFrame({x_feature: grid_x.ravel(), y_feature: grid_y.ravel()})
    scores = result.pipeline.decision_function(grid).reshape(grid_x.shape)
    figure.add_trace(go.Contour(
        x=xs, y=ys, z=scores, contours={"start": 0, "end": 0, "size": 1},
        line={"color": "#252525", "width": 3}, showscale=False, hoverinfo="skip", name="Decision boundary",
    ))
    if result.svc.kernel == "linear":
        figure.add_trace(go.Contour(
            x=xs, y=ys, z=scores, contours={"start": -1, "end": 1, "size": 2},
            line={"color": "#666666", "width": 1.5}, showscale=False, hoverinfo="skip", name="Margins",
        ))
    scaler = result.pipeline.named_steps.get("scaler")
    support = result.svc.support_vectors_.copy()
    if scaler is not None:
        support = scaler.inverse_transform(support)
    figure.add_trace(go.Scatter(
        x=support[:, 0], y=support[:, 1], mode="markers", name="Support vectors",
        marker={"symbol": "circle-open", "size": 15, "line": {"color": "#111", "width": 2}},
    ))
    wrong = result.y_test.to_numpy() != result.predictions
    if wrong.any():
        figure.add_trace(go.Scatter(
            x=result.X_test.iloc[wrong, 0], y=result.X_test.iloc[wrong, 1], mode="markers", name="Test mistakes",
            marker={"symbol": "x", "size": 14, "color": "#b2182b", "line": {"width": 2}},
        ))
    figure.update_layout(title="Trained SVM: boundary, support vectors, and test mistakes")
    return figure


def comparison_scatter(frame: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
    """Show an accuracy-versus-inference-cost view for saved experiments."""
    return px.scatter(
        frame, x=x_column, y=y_column, color="kernel", hover_name="name", size="support_vectors",
        labels={"model_bytes": "Estimated model bytes", "estimated_ops": "Estimated operations", "n_features": "Features"},
        title="Accuracy versus estimated inference cost",
    )
