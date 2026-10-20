"""Transparent, analytic embedded-inference estimates and C export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import TrainingResult, raw_linear_parameters


@dataclass(frozen=True)
class CostEstimate:
    """Deterministic model storage and operation estimates; not timing data."""

    parameter_bytes: int
    multiplies: int
    adds: int
    support_vectors: int
    kernel_evaluations: int = 0
    exponentials: int = 0


def estimate_cost(result: TrainingResult, dtype: str) -> CostEstimate:
    """Estimate inference memory/arithmetic directly from fitted model dimensions."""
    bytes_per_value = 4 if dtype == "float32" else 8
    n_features = len(result.feature_names)
    svc = result.svc
    if svc.kernel == "linear":
        return CostEstimate((n_features + 1) * bytes_per_value, n_features, n_features, 0)

    n_sv = result.support_vector_count
    # Store each support vector, one dual coefficient per vector, and one intercept.
    parameter_bytes = (n_sv * (n_features + 1) + 1) * bytes_per_value
    if svc.kernel == "rbf":
        # Per support vector: feature differences/squares/sum + multiply by gamma;
        # one exponential and the weighted accumulation are recorded separately.
        return CostEstimate(parameter_bytes, n_sv * (n_features + 2), n_sv * (n_features + 1), n_sv, n_sv, n_sv)
    return CostEstimate(parameter_bytes, n_sv * (n_features + 1), n_sv * n_features, n_sv, n_sv)


def linear_formula(result: TrainingResult, precision: int = 5) -> str:
    """Make the raw-unit decision score readable for a lecture slide."""
    weights, intercept = raw_linear_parameters(result)
    terms = [f"({weight:.{precision}g}) · {name}" for weight, name in zip(weights, result.feature_names)]
    return "score = " + " + ".join(terms) + f" + ({intercept:.{precision}g})"


def generate_c_code(result: TrainingResult, dtype: str) -> str:
    """Generate generic, dependency-free C code for a trained linear SVM."""
    weights, intercept = raw_linear_parameters(result)
    c_type = "float" if dtype == "float32" else "double"
    suffix = "f" if dtype == "float32" else ""
    weight_values = ", ".join(f"{value:.9g}{suffix}" for value in weights)
    return f"""/* Generic linear SVM inference. Inputs use the original feature units. */
#include <stddef.h>

static const {c_type} SVM_WEIGHTS[] = {{ {weight_values} }};
static const {c_type} SVM_BIAS = {intercept:.9g}{suffix};

/* Returns 1 for the positive class and 0 for the negative class. */
int svm_predict(const {c_type} features[], size_t count) {{
    {c_type} score = SVM_BIAS;
    for (size_t i = 0; i < count; ++i) {{
        score += SVM_WEIGHTS[i] * features[i];
    }}
    return score >= 0 ? 1 : 0;
}}
"""


def linear_score(result: TrainingResult, values: list[float]) -> float:
    """Compute the exported raw-unit arithmetic for one observation."""
    weights, intercept = raw_linear_parameters(result)
    return float(np.dot(weights, values) + intercept)
