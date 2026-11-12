"""A small page so someone can type four iris measurements and see a prediction."""

import pandas as pd
import streamlit as st

from save_models import train

EXAMPLES = {
    "Setosa": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    },
    "Versicolor": {
        "sepal_length": 6.0,
        "sepal_width": 2.8,
        "petal_length": 4.5,
        "petal_width": 1.4,
    },
    "Virginica": {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.6,
        "petal_width": 2.1,
    },
}

FIELDS = [
    ("sepal_length", "Sepal length (cm)", 4.3, 7.9),
    ("sepal_width", "Sepal width (cm)", 2.0, 4.4),
    ("petal_length", "Petal length (cm)", 1.0, 6.9),
    ("petal_width", "Petal width (cm)", 0.1, 2.5),
]


@st.cache_resource
def load_models():
    bundle, log_acc, tree_acc = train()
    return bundle, log_acc, tree_acc


def use_example(values):
    for key, value in values.items():
        st.session_state[key] = value


st.set_page_config(page_title="Iris species predictor", layout="centered")
st.title("Iris species predictor")
st.write(
    "Enter the four measurements from one flower, in centimeters. "
    "Both models were trained on 120 flowers from iris.csv. "
    "A new flower is scaled with the training mean and training standard deviation "
    "before either model sees it."
)

bundle, log_acc, tree_acc = load_models()

if "sepal_length" not in st.session_state:
    use_example(EXAMPLES["Setosa"])

st.write("Or start from a real example:")
example_cols = st.columns(3)
for col, (name, values) in zip(example_cols, EXAMPLES.items()):
    with col:
        st.button(f"Try {name.lower()}", on_click=use_example, args=(values,))

input_cols = st.columns(2)
measurements = {}
for i, (key, label, low, high) in enumerate(FIELDS):
    with input_cols[i % 2]:
        measurements[key] = st.number_input(
            label,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            key=key,
            help=f"In the training data this measurement runs from {low} to {high} cm.",
        )

row = pd.DataFrame(
    [[measurements[key] for key, *_rest in FIELDS]],
    columns=bundle["feature_cols"],
)
scaled = bundle["scaler"].transform(row)
log_pred = bundle["log_reg"].predict(scaled)[0]
tree_pred = bundle["tree"].predict(scaled)[0]
log_proba = bundle["log_reg"].predict_proba(scaled)[0]
tree_proba = bundle["tree"].predict_proba(scaled)[0]

st.subheader("Prediction")
result_cols = st.columns(2)
with result_cols[0]:
    st.metric("Logistic Regression", log_pred)
    st.bar_chart(
        pd.DataFrame({"probability": log_proba}, index=bundle["log_reg"].classes_)
    )
with result_cols[1]:
    st.metric("Decision Tree", tree_pred)
    st.bar_chart(
        pd.DataFrame({"probability": tree_proba}, index=bundle["tree"].classes_)
    )

if log_pred == tree_pred:
    st.success(f"Both models call this flower {log_pred}.")
else:
    st.warning(
        f"The models disagree. Logistic Regression says {log_pred}. "
        f"The Decision Tree says {tree_pred}."
    )

with st.expander("What the models actually see"):
    st.write(
        "Each scaled value is (measurement - training mean) / training standard deviation. "
        "Zero means this flower matches the training average for that measurement."
    )
    scaled_table = pd.DataFrame(
        {
            "centimeters": [measurements[col] for col in bundle["feature_cols"]],
            "scaled": scaled.ravel(),
        },
        index=bundle["feature_cols"],
    )
    st.dataframe(scaled_table.round(3))

st.caption(
    f"On the 30 test flowers, Logistic Regression scored {log_acc:.3f} "
    f"and the Decision Tree scored {tree_acc:.3f}."
)
