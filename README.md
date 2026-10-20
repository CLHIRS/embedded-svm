# SVM Embedded Playground

An interactive Streamlit teaching app for explaining SVM geometry, model choices, and embedded inference trade-offs with a fruit-classification example.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (normally `http://localhost:8501`).

## Architecture

- `app.py` assembles the student-first lesson flow and Streamlit state.
- `src/data.py` creates reproducible Apple/Pear data and an optional binary Iris dataset.
- `src/model.py` trains and evaluates an `sklearn` SVC pipeline.
- `src/plots.py` keeps interactive Plotly plots separate from application logic.
- `src/embedded.py` converts linear models to raw-unit formulas / generic C and provides deterministic storage and operation estimates.

The compile panel intentionally reports no MCU latency: without a target board, toolchain, clock rate, and measurement procedure, timing would be misleading. Its byte and arithmetic figures are labeled estimates derived from the fitted model shape.
