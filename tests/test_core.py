from src.data import make_fruit_dataset
from src.embedded import estimate_cost
from src.model import train_svm


def test_fruit_data_is_reproducible():
    first = make_fruit_dataset(80, 0.3, 42)
    second = make_fruit_dataset(80, 0.3, 42)
    assert first.X.equals(second.X)
    assert list(first.X.columns) == ["weight_g", "redness", "height_mm", "diameter_mm"]


def test_linear_model_has_small_deterministic_cost():
    data = make_fruit_dataset(80, 0.3, 42)
    result = train_svm(data.X, data.y, ["weight_g", "redness"], 0.25, 42, {
        "C": 1.0, "kernel": "linear", "gamma": "scale", "degree": 3,
        "coef0": 0.0, "class_weight": None, "scale": True,
    })
    estimate = estimate_cost(result, "float32")
    assert estimate.parameter_bytes == 12
    assert estimate.multiplies == 2
