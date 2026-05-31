"""
Solar Power Generation Prediction MVP
ENGG2112 Machine Learning Project

Run:
    python -m streamlit run app.py

This version uses the built-in sample_data.csv automatically.
No CSV upload is required for the final presentation demo.
"""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Solar Power Prediction MVP",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Power Generation Prediction MVP")
st.caption(
    "A fixed-input Streamlit dashboard that predicts generated solar power from weather and solar-related features."
)

# -----------------------------
# Constants
# -----------------------------
TARGET_COL = "generated_power_kw"
DATA_FILE = "sample_data.csv"
ANGLE_COLS = [
    "wind_direction_10_m_above_gnd",
    "wind_direction_80_m_above_gnd",
    "wind_direction_900_mb",
    "azimuth",
]

SCALED_MODELS = ["Linear Regression", "SVR", "MLP"]
TREE_MODELS = ["Random Forest"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
ALL_MODELS = ["Linear Regression", "Random Forest", "SVR", "MLP"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])

# -----------------------------
# Helper functions
# -----------------------------
def load_builtin_data() -> pd.DataFrame:
    """Load the built-in dataset from the same folder as app.py."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Cannot find {DATA_FILE}. Please keep sample_data.csv in the same folder as app.py.")
        st.stop()
    return pd.read_csv(DATA_FILE)


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing as the project code:
    1. remove duplicate rows
    2. convert direction/angle columns into sine and cosine components
    3. remove original angle columns
    4. keep numeric data only
    """
    df = raw_df.copy().drop_duplicates()

    missing_angles = [col for col in ANGLE_COLS if col not in df.columns]
    if missing_angles:
        raise ValueError("Missing angle columns: " + ", ".join(missing_angles))

    for col in ANGLE_COLS:
        radians = np.deg2rad(pd.to_numeric(df[col], errors="coerce"))
        df[col + "_sin"] = np.sin(radians)
        df[col + "_cos"] = np.cos(radians)

    df = df.drop(columns=ANGLE_COLS)

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        df = df.drop(columns=non_numeric_cols)

    df = df.dropna()
    return df


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


@st.cache_data(show_spinner=False)
def train_models(raw_df: pd.DataFrame, test_size: float, random_state: int):
    """Preprocess the built-in data, train all models, and return metrics and predictions."""
    df = preprocess_data(raw_df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in the dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVR": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    trained_models = {}
    predictions = {}
    metrics = []

    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        mae, rmse, r2 = evaluate_model(y_test, pred)
        metrics.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})
        predictions[name] = pred
        trained_models[name] = model

    results_df = pd.DataFrame(metrics).sort_values(by="R²", ascending=False).reset_index(drop=True)

    return {
        "processed_df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "models": trained_models,
        "predictions": predictions,
        "results_df": results_df,
    }


def predict_one_row(model_name: str, model, scaler, row_df: pd.DataFrame) -> float:
    """Predict one selected sample row."""
    if model_name in SCALED_MODELS:
        row_scaled = scaler.transform(row_df)
        return float(model.predict(row_scaled)[0])
    return float(model.predict(row_df)[0])


def plot_actual_vs_predicted(y_test, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.55)
    low = min(y_test.min(), np.min(y_pred))
    high = max(y_test.max(), np.max(y_pred))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=2)
    ax.set_xlabel("Actual Power (kW)")
    ax.set_ylabel("Predicted Power (kW)")
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.grid(True, alpha=0.3)
    return fig


def plot_residuals(y_test, y_pred, model_name: str):
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, alpha=0.55)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Power (kW)")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{model_name}: Residual Plot")
    ax.grid(True, alpha=0.3)
    return fig


def plot_feature_importance(model_name: str, model, feature_columns):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        title = f"{model_name}: Feature Importance"
        x_label = "Importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
        title = "Linear Regression: Absolute Coefficient Size"
        x_label = "Absolute Coefficient"
    else:
        return None

    importance_df = pd.DataFrame({"Feature": feature_columns, "Value": values})
    importance_df = importance_df.sort_values(by="Value", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Value"])
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("MVP Controls")
st.sidebar.success("Using built-in sample_data.csv. No upload is required.")
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Model limitations")
st.sidebar.write(
    "Predictions are based on historical data patterns. This MVP should not be used as a guaranteed forecast for new locations, extreme weather, or safety-critical decisions."
)

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost is not installed. The app will still run with the other four models.")

# -----------------------------
# 1. Built-in input data
# -----------------------------
st.subheader("1. Built-in Solar Dataset")
raw_df = load_builtin_data()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{raw_df.shape[0]}")
c2.metric("Columns", f"{raw_df.shape[1]}")
c3.metric("Target", TARGET_COL)

with st.expander("Preview the fixed input data", expanded=False):
    st.dataframe(raw_df.head(10), use_container_width=True)

st.info(
    "For the presentation demo, the app uses selected samples from this built-in dataset as the input. "
    "This avoids file-upload errors during the live MVP demonstration."
)

# -----------------------------
# 2. Train and compare models
# -----------------------------
st.subheader("2. Train and Compare Models")

try:
    with st.spinner("Preprocessing built-in data and training models..."):
        artifacts = train_models(raw_df, float(test_size), int(random_state))
except Exception as exc:
    st.error("The MVP could not process the built-in dataset.")
    st.exception(exc)
    st.stop()

results_df = artifacts["results_df"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]
feature_columns = artifacts["X"].columns.tolist()
best_model_name = results_df.iloc[0]["Model"]

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", best_model_name)
col2.metric("Best R²", f"{results_df.iloc[0]['R²']:.4f}")
col3.metric("Best RMSE", f"{results_df.iloc[0]['RMSE']:.4f}")

st.dataframe(
    results_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
    use_container_width=True,
)

st.markdown(
    "**Interpretation:** Lower MAE/RMSE means smaller prediction error. Higher R² means the model explains more variation in generated solar power."
)

# -----------------------------
# 3. Prediction demo using selected fixed sample
# -----------------------------
st.subheader("3. Fixed Sample Prediction Demo")

selected_model = st.selectbox("Choose a model", results_df["Model"].tolist())
selected_model_obj = artifacts["models"][selected_model]
selected_pred_array = artifacts["predictions"][selected_model]

max_index = min(20, len(X_test) - 1)
sample_number = st.slider(
    "Choose one built-in test sample as the demo input",
    min_value=0,
    max_value=max_index,
    value=0,
    step=1,
)

row_df = X_test.iloc[[sample_number]]
actual_value = float(y_test.iloc[sample_number])
predicted_value = predict_one_row(
    selected_model,
    selected_model_obj,
    artifacts["scaler"],
    row_df,
)
absolute_error = abs(actual_value - predicted_value)

p1, p2, p3 = st.columns(3)
p1.metric("Predicted Power", f"{predicted_value:.2f} kW")
p2.metric("Actual Power", f"{actual_value:.2f} kW")
p3.metric("Absolute Error", f"{absolute_error:.2f} kW")

with st.expander("Show this sample's input features", expanded=False):
    st.dataframe(row_df.T.rename(columns={row_df.index[0]: "Input Value"}), use_container_width=True)

st.write(
    "This section is the main MVP demo: a user selects one prepared weather/solar condition sample, "
    "and the system outputs the predicted generated power in kW."
)

# -----------------------------
# 4. Visualise model output
# -----------------------------
st.subheader("4. Model Output Visualisation")

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.pyplot(plot_actual_vs_predicted(y_test, selected_pred_array, selected_model), use_container_width=True)
with viz_col2:
    st.pyplot(plot_residuals(y_test, selected_pred_array, selected_model), use_container_width=True)

importance_fig = plot_feature_importance(selected_model, selected_model_obj, feature_columns)
if importance_fig is not None:
    st.pyplot(importance_fig, use_container_width=True)
    st.caption(
        "This chart shows which features contribute most strongly to the model prediction. It is model explanation, not proof of direct causation."
    )
else:
    st.info("Feature importance is not directly available for this model. Use Random Forest, XGBoost, or Linear Regression for an explanation chart.")

# -----------------------------
# 5. Stakeholder summary
# -----------------------------
st.subheader("5. Stakeholder Summary")
st.markdown(
    f"""
    This MVP demonstrates a decision-support workflow for solar power planning.  
    The current best-performing model is **{best_model_name}** based on the fixed built-in dataset split.  
    Users can compare models, inspect prediction errors, select prepared input samples, and review feature-based explanations.

    **Main limitation:** the model learns from historical data. Its reliability may reduce when applied to a new location, new sensor setup, missing features, or unusual weather conditions.
    """
)
