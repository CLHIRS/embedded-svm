"""Streamlit entry point for the SVM Embedded Playground."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from src.data import Dataset, make_fruit_dataset, make_iris_binary_dataset
from src.embedded import estimate_cost, generate_c_code, linear_formula, linear_score
from src.model import TrainingResult, train_svm
from src.plots import comparison_scatter, data_scatter, svm_geometry_plot


st.set_page_config(page_title="SVM Embedded Playground", page_icon="🍎", layout="wide")


def initialise_state() -> None:
    """Create persistent state only once per browser session."""
    st.session_state.setdefault("trained_result", None)
    st.session_state.setdefault("trained_dataset_key", None)
    st.session_state.setdefault("experiments", [])


def current_dataset(kind: str, samples: int, overlap: float, seed: int) -> Dataset:
    """Create the selected lecture dataset."""
    if kind == "Apple vs Pear (synthetic)":
        return make_fruit_dataset(samples, overlap, seed)
    return make_iris_binary_dataset(samples, seed)


def configuration_controls() -> dict[str, Any]:
    """Render progressive-disclosure SVM controls and return sklearn settings."""
    st.caption("Start with C. Open the next level only when you need another modelling decision.")
    level = st.radio("Lesson level", ["Basic", "Intermediate", "Advanced"], horizontal=True)
    C = st.slider(
        "C — penalty for training mistakes", 0.01, 30.0, 1.0, 0.01,
        help="Higher C pushes the SVM to reduce training mistakes, often producing a narrower, less forgiving margin. Lower C permits more mistakes for a wider margin.",
    )
    config: dict[str, Any] = {
        "C": C, "kernel": "linear", "gamma": "scale", "degree": 3,
        "coef0": 0.0, "class_weight": None, "scale": True,
    }
    if level in ("Intermediate", "Advanced"):
        config["kernel"] = st.selectbox(
            "Kernel", ["linear", "rbf"],
            help="Linear draws one flat separating plane. RBF can bend around more complex patterns but requires support-vector kernel evaluations at inference.",
        )
        if config["kernel"] == "rbf":
            config["gamma"] = st.selectbox(
                "Gamma", ["scale", "auto", 0.01, 0.1, 1.0, 10.0], index=0,
                help="For RBF, gamma controls how locally each training point influences the boundary. Larger values make more detailed, potentially less general boundaries.",
            )
    if level == "Advanced":
        st.markdown("**Advanced model and preprocessing settings**")
        config["kernel"] = st.selectbox(
            "Kernel (advanced)", ["linear", "rbf", "poly"], index=["linear", "rbf", "poly"].index(config["kernel"]),
            help="Polynomial is included for exploration. Its degree and offset only affect this kernel.",
        )
        if config["kernel"] in ("rbf", "poly"):
            config["gamma"] = st.selectbox(
                "Gamma (advanced)", ["scale", "auto", 0.01, 0.1, 1.0, 10.0], index=0,
                help="Kernel coefficient. Its effect depends on the chosen kernel and feature scaling.",
            )
        if config["kernel"] == "poly":
            config["degree"] = st.slider("Polynomial degree", 2, 6, 3, help="The highest power used by the polynomial kernel. Higher degrees add flexibility and cost.")
            config["coef0"] = st.number_input("Polynomial coef0", -5.0, 5.0, 0.0, 0.1, help="A constant offset used by the polynomial kernel.")
        class_weight_name = st.selectbox(
            "Class weights", ["None", "balanced"],
            help="Balanced automatically upweights the rarer class. It is most useful with imbalanced data.",
        )
        config["class_weight"] = None if class_weight_name == "None" else "balanced"
        config["scale"] = st.checkbox(
            "Standardize selected features", value=True,
            help="Subtract each training-feature mean and divide by its standard deviation. Usually important when feature units differ; exported linear coefficients are converted back to raw units.",
        )

    st.latex(r"\min_{w,b,\xi}\;\frac{1}{2}\lVert w\rVert^2 + C\sum_i\xi_i \quad \text{subject to}\quad y_i(w^T x_i+b)\geq 1-\xi_i,\;\xi_i\geq0")
    st.markdown("**Formula legend**")
    st.markdown(
        "| Symbol | Meaning |\n"
        "| --- | --- |\n"
        "| $x_i$ | Feature vector for observation $i$ (for example, weight and redness) |\n"
        "| $y_i$ | Its true class label: Apple or Pear |\n"
        "| $w$ | Learned feature weights; their direction sets the separating boundary |\n"
        "| $b$ | Learned offset (intercept) that shifts the boundary |\n"
        "| $\\xi_i$ | Slack: how much observation $i$ is allowed to enter or cross the margin |\n"
        "| $C$ | Penalty for slack; the accuracy-versus-margin trade-off control |"
    )
    st.info("**Think about C:** What changed when C increased—training mistakes, margin width, or the number of support vectors?")
    return config


def save_experiment(result: TrainingResult, name: str, dtype: str) -> None:
    """Store a compact comparison snapshot rather than another copy of the model."""
    estimate = estimate_cost(result, dtype)
    signature = hashlib.sha1(f"{name}{len(st.session_state.experiments)}".encode()).hexdigest()[:6]
    st.session_state.experiments.append({
        "name": name.strip() or f"Experiment {len(st.session_state.experiments) + 1}",
        "id": signature,
        "kernel": result.svc.kernel,
        "C": result.config["C"],
        "gamma": result.config["gamma"],
        "scaling": "yes" if result.config["scale"] else "no",
        "class_weight": result.config["class_weight"] or "none",
        "features": ", ".join(result.feature_names),
        "n_features": len(result.feature_names),
        "accuracy": result.metrics["accuracy"],
        "precision": result.metrics["precision"],
        "recall": result.metrics["recall"],
        "f1": result.metrics["f1"],
        "support_vectors": result.support_vector_count,
        "model_bytes": estimate.parameter_bytes,
        "estimated_ops": estimate.multiplies + estimate.adds,
        "training_ms": result.training_ms,
    })


def show_evaluation(result: TrainingResult, dataset: Dataset, x_axis: str, y_axis: str) -> None:
    """Display model quality and, when valid, the full 2D SVM geometry."""
    st.subheader("Training results")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{result.metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{result.metrics['precision']:.1%}")
    m3.metric("Recall", f"{result.metrics['recall']:.1%}")
    m4.metric("F1", f"{result.metrics['f1']:.1%}")
    m5.metric("Training time", f"{result.training_ms:.1f} ms")
    m6.metric("Support vectors", result.support_vector_count)
    left, right = st.columns([1.65, 1])
    with left:
        if len(result.feature_names) == 2 and result.feature_names == [x_axis, y_axis]:
            st.plotly_chart(svm_geometry_plot(dataset, result, x_axis, y_axis), width="stretch", key="trained_geometry")
        else:
            st.plotly_chart(data_scatter(dataset, x_axis, y_axis), width="stretch", key="evaluation_data_fallback")
            st.caption("The decision geometry is only overlaid when the SVM uses exactly the two displayed features. A 3+ dimensional model cannot be shown faithfully as one 2D boundary.")
    with right:
        cm = pd.DataFrame(
            result.confusion,
            index=[f"Actual {dataset.class_names[0]}", f"Actual {dataset.class_names[1]}"],
            columns=[f"Predicted {dataset.class_names[0]}", f"Predicted {dataset.class_names[1]}"],
        )
        st.markdown("**Confusion matrix (test set)**")
        st.dataframe(cm, width="stretch")
        st.caption("Training time is a measurement from this browser session; it is not embedded-device timing.")


def show_compile_export(result: TrainingResult, dtype: str) -> None:
    """Show a deliberately honest, target-independent deployment view."""
    architecture, numeric_type, optimisation = st.columns(3)
    with architecture:
        st.selectbox("Architecture target", ["Generic C", "Cortex-M4", "Cortex-M7"], help="Placeholder target selection for later toolchain integration.")
    with numeric_type:
        st.metric("Numeric type", dtype, help="This selection controls the deterministic storage estimate and C type.")
    with optimisation:
        st.selectbox("Compiler optimization", ["-O0", "-O1", "-O2", "-O3", "-Os"], index=4, help="A planning setting only; this MVP does not run a target compiler.")
    estimate = estimate_cost(result, dtype)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated parameter bytes", estimate.parameter_bytes, help="Analytic estimate: learned numeric parameters × selected numeric type size. It excludes code, stack, and runtime overhead.")
    c2.metric("Estimated multiplies / adds", f"{estimate.multiplies} / {estimate.adds}", help="Analytic inference arithmetic count; it is not a cycle or latency measurement.")
    c3.metric("Kernel evaluations", estimate.kernel_evaluations, help="One evaluation per support vector for nonlinear SVM inference.")
    st.caption("**Estimates, not measurements.** No MCU latency is reported because it requires a specific board, compiler, clock configuration, and benchmark method.")
    if result.svc.kernel == "linear":
        st.markdown("**Learned decision formula in original feature units**")
        st.code(linear_formula(result), language="text")
        st.markdown("**Generic C inference code**")
        st.code(generate_c_code(result, dtype), language="c")
        st.caption("The exported score is ≥ 0 for the positive class. It expects features in the order shown in the formula.")
    else:
        extra = f", plus {estimate.exponentials} exponential operations" if estimate.exponentials else ""
        st.warning(
            f"A {result.svc.kernel.upper()} SVM cannot be reduced to one fixed weight vector. This model stores {estimate.support_vectors} support vectors and performs {estimate.kernel_evaluations} kernel evaluations per prediction{extra}. Generic linear C would be misleading, so it is intentionally not generated."
        )


def show_inference(result: TrainingResult, dataset: Dataset) -> None:
    """Let students run a single prediction and inspect the actual score."""
    st.caption("Training learns parameters from many labelled examples. Embedded inference applies the frozen parameters to one new measurement.")
    defaults = result.X_train.median()
    values: list[float] = []
    columns = st.columns(min(4, len(result.feature_names)))
    for idx, feature in enumerate(result.feature_names):
        with columns[idx % len(columns)]:
            values.append(st.number_input(
                dataset.display_names[feature], value=float(defaults[feature]), format="%.3f",
                key=f"infer_{feature}", help="A single new measurement to pass through the trained model.",
            ))
    if st.button("Classify this fruit", type="secondary"):
        observation = pd.DataFrame([values], columns=result.feature_names)
        prediction = int(result.pipeline.predict(observation)[0])
        score = float(result.pipeline.decision_function(observation)[0])
        st.success(f"Predicted class: **{dataset.class_names[prediction]}**")
        st.write(f"Decision score: `{score:.5f}` (positive means {dataset.class_names[1]}).")
        if result.svc.kernel == "linear":
            raw_score = linear_score(result, values)
            st.code(f"{linear_formula(result)}\nscore = {raw_score:.5f}", language="text")
        else:
            st.caption("This score is produced from support-vector kernel evaluations, not one w·x + b arithmetic expression.")


def show_comparison() -> None:
    """Compare compact experiment snapshots kept in session state."""
    experiments = st.session_state.experiments
    if not experiments:
        st.info("Save two or more trained experiments to compare accuracy against estimated inference cost here.")
        return
    frame = pd.DataFrame(experiments)
    table = frame.copy()
    for column in ["accuracy", "precision", "recall", "f1"]:
        table[column] = table[column].map(lambda value: f"{value:.1%}")
    table["training_ms"] = table["training_ms"].map(lambda value: f"{value:.1f}")
    st.dataframe(table.drop(columns=["id"]), width="stretch", hide_index=True)
    if len(frame) > 1:
        x_labels = {"C": "C", "Features": "n_features", "Support vectors": "support_vectors", "Model bytes": "model_bytes", "Estimated ops": "estimated_ops"}
        y_labels = {"Accuracy": "accuracy", "F1": "f1", "Recall": "recall"}
        left, right = st.columns(2)
        with left:
            x_choice = st.selectbox("Comparison X axis", list(x_labels), index=3)
        with right:
            y_choice = st.selectbox("Comparison Y axis", list(y_labels), index=0)
        st.plotly_chart(comparison_scatter(frame, x_labels[x_choice], y_labels[y_choice]), width="stretch", key="experiment_comparison")
        st.info("**Design question:** Does adding features or support vectors improve accuracy enough to justify the added embedded model storage and arithmetic?")


def main() -> None:
    initialise_state()
    st.title("🍎 SVM Embedded Playground 🍐")
    st.caption("An interactive lesson on SVM geometry, hyperparameters, and embedded deployment trade-offs.")

    with st.expander("0. Problem", expanded=True):
        st.write("A sensor measures fruit properties and must classify each item as an apple or pear. We want good predictions, but an embedded device also has limited memory and compute for each inference.")
        st.markdown("`Sensor features  →  SVM training  →  decision boundary  →  embedded inference`")

    with st.expander("1. Data", expanded=True):
        dataset_kind = st.selectbox("Dataset", ["Apple vs Pear (synthetic)", "Iris (Setosa vs Versicolor)"], help="The fruit data is purpose-built for this lesson. Iris is a familiar real measurement dataset with two classes selected.")
        samples = st.slider("Number of observations", 40, 200, 120, 10, help="Total labelled observations. The Iris option contains up to 100 binary-class observations.")
        overlap = st.slider("Class overlap / noise", 0.0, 1.0, 0.32, 0.01, help="Higher values make class measurements more variable and closer together. This makes the soft-margin trade-off visible.", disabled=dataset_kind != "Apple vs Pear (synthetic)")
        seed = st.number_input("Random seed", 0, 99_999, 42, 1, help="The seed makes generated data and the train/test split reproducible.")
        test_split = st.slider("Test split", 0.15, 0.45, 0.25, 0.05, help="Fraction held out from training for an honest evaluation.")
        if dataset_kind != "Apple vs Pear (synthetic)" and samples > 100:
            st.caption("Iris has 100 observations for this binary selection; the app will use all 100.")
    dataset = current_dataset(dataset_kind, samples, overlap, int(seed))

    with st.expander("2. Feature selection & visualization", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            selected_features = st.multiselect(
                "Features used by the SVM", list(dataset.X.columns), default=list(dataset.X.columns[:2]),
                help="The classifier can use more than two features. More information can help accuracy, but can also increase model storage and inference work.",
            )
        with f2:
            x_axis = st.selectbox("X-axis for the plot", list(dataset.X.columns), index=0, format_func=lambda name: dataset.display_names[name], help="Display choice only; it does not change which features train the model.")
            y_options = [feature for feature in dataset.X.columns if feature != x_axis]
            y_axis = st.selectbox("Y-axis for the plot", y_options, index=0, format_func=lambda name: dataset.display_names[name], help="Display choice only. A 2D chart cannot fully show a model trained on three or four features.")
        st.plotly_chart(data_scatter(dataset, x_axis, y_axis), width="stretch", key="data_preview")
        st.caption("The selected features train the SVM; the X/Y axes only choose a two-dimensional window into the data.")

    with st.expander("3. SVM configuration", expanded=True):
        config = configuration_controls()
        st.markdown("**Embedded estimate setting**")
        dtype = st.selectbox(
            "Numeric type for estimates and C export", ["float32", "float64"],
            help="Float32 uses four bytes per learned parameter; float64 uses eight. This is a planning assumption, not a measured device configuration.",
        )

    dataset_key = (dataset_kind, len(dataset.X), overlap, int(seed))
    result: TrainingResult | None = st.session_state.trained_result
    with st.expander("4. Training & evaluation", expanded=True):
        if not selected_features:
            st.warning("Select at least one feature before training.")
        elif st.button("Train SVM", type="primary"):
            st.session_state.trained_result = train_svm(dataset.X, dataset.y, selected_features, test_split, int(seed), config)
            st.session_state.trained_dataset_key = dataset_key
            result = st.session_state.trained_result
            st.success("Training complete. Explore the evaluation and deployment sections below.")

        if result is not None and st.session_state.trained_dataset_key != dataset_key:
            st.warning("The dataset has changed since the last training run. Train again before interpreting results or exporting a model.")
            result = None
        if result is not None:
            show_evaluation(result, dataset, x_axis, y_axis)
            save_col, name_col = st.columns([1, 3])
            with name_col:
                experiment_name = st.text_input("Experiment name", value="", placeholder="e.g., Linear C=1, weight + redness")
            with save_col:
                st.write("")
                if st.button("Save experiment"):
                    save_experiment(result, experiment_name, dtype)
                    st.success("Saved for comparison in this browser session.")
        else:
            st.info("Set up the first experiment, then choose **Train SVM** to reveal evaluation, compile, and inference views.")

    if result is not None:
        with st.expander("5. Compile & export", expanded=True):
            show_compile_export(result, dtype)
        with st.expander("6. Interactive inference", expanded=True):
            show_inference(result, dataset)
    with st.expander("7. Experiment comparison", expanded=True):
        show_comparison()


if __name__ == "__main__":
    main()
