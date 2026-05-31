# %% Data preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

# Read data
df = pd.read_csv("spg.csv")

# Delete duplicate rows
df = df.drop_duplicates()

# Convert angle features into sine and cosine components
angle_cols = [
    "wind_direction_10_m_above_gnd",
    "wind_direction_80_m_above_gnd",
    "wind_direction_900_mb",
    "azimuth"
]

for col in angle_cols:
    radians = np.deg2rad(df[col])
    df[col + "_sin"] = np.sin(radians)
    df[col + "_cos"] = np.cos(radians)

# Drop the original angle columns
df = df.drop(columns=angle_cols)

# Define features and target variable
X = df.drop(columns=["generated_power_kw"])
y = df["generated_power_kw"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature standardization
# Linear Regression, SVR and MLP are sensitive to feature scale.
# Random Forest and XGBoost can use the original unscaled data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preparation completed.")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# %% Correlation heatmap

corr_matrix = df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Variables Correlation Heatmap")
plt.show()


# %% Helper function for evaluation

def evaluate_model(model_name, y_true, y_pred):
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_value = r2_score(y_true, y_pred)

    print(f"{model_name} Results")
    print(f"MAE  = {mae_value:.4f}")
    print(f"RMSE = {rmse_value:.4f}")
    print(f"R²   = {r2_value:.4f}")
    print()

    return mae_value, rmse_value, r2_value


# %% Linear Regression model

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

mae, rmse, r2 = evaluate_model("Linear Regression", y_test, y_pred)


# %% Linear Regression plots

# True vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("True Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("Linear Regression: True vs Predicted Solar Power")
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Power (kW)")
plt.ylabel("Residuals")
plt.title("Linear Regression Residual Plot")
plt.grid(True)
plt.show()

# Coefficient plot
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Linear Regression Coefficients")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# %% Random Forest model

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf, rmse_rf, r2_rf = evaluate_model("Random Forest", y_test, y_pred_rf)


# %% Random Forest plots

# True vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("True Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("Random Forest: True vs Predicted Solar Power")
plt.grid(True)
plt.show()

# Residual plot
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Power (kW)")
plt.ylabel("Residuals")
plt.title("Random Forest Residual Plot")
plt.grid(True)
plt.show()

# Feature importance plot
rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(rf_importance["Feature"], rf_importance["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# %% SVR model

svr = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

mae_svr, rmse_svr, r2_svr = evaluate_model("SVR", y_test, y_pred_svr)


# %% SVR plots

# True vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_svr, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("True Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("SVR: True vs Predicted Solar Power")
plt.grid(True)
plt.show()

# Residual plot
residuals_svr = y_test - y_pred_svr

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_svr, residuals_svr, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Power (kW)")
plt.ylabel("Residuals")
plt.title("SVR Residual Plot")
plt.grid(True)
plt.show()


# %% MLP model

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)

mae_mlp, rmse_mlp, r2_mlp = evaluate_model("MLP", y_test, y_pred_mlp)


# %% MLP plots

# True vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_mlp, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("True Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("MLP: True vs Predicted Solar Power")
plt.grid(True)
plt.show()

# Residual plot
residuals_mlp = y_test - y_pred_mlp

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_mlp, residuals_mlp, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Power (kW)")
plt.ylabel("Residuals")
plt.title("MLP Residual Plot")
plt.grid(True)
plt.show()

# Training loss curve
plt.figure(figsize=(7, 5))
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.grid(True)
plt.show()


# %% XGBoost model

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

mae_xgb, rmse_xgb, r2_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb)


# %% XGBoost plots

# True vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)
plt.xlabel("True Power (kW)")
plt.ylabel("Predicted Power (kW)")
plt.title("XGBoost: True vs Predicted Solar Power")
plt.grid(True)
plt.show()

# Residual plot
residuals_xgb = y_test - y_pred_xgb

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_xgb, residuals_xgb, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Power (kW)")
plt.ylabel("Residuals")
plt.title("XGBoost Residual Plot")
plt.grid(True)
plt.show()

# Feature importance plot
xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(xgb_importance["Feature"], xgb_importance["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# %% Model comparison table

results_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "SVR",
        "MLP",
        "XGBoost"
    ],
    "MAE": [
        mae,
        mae_rf,
        mae_svr,
        mae_mlp,
        mae_xgb
    ],
    "RMSE": [
        rmse,
        rmse_rf,
        rmse_svr,
        rmse_mlp,
        rmse_xgb
    ],
    "R2": [
        r2,
        r2_rf,
        r2_svr,
        r2_mlp,
        r2_xgb
    ]
})

print("Model Comparison")
print(results_df)


# %% Model comparison plots

# R2 comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R² Score")
plt.title("Model Comparison Based on R² Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()

# MAE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.ylabel("MAE")
plt.title("Model Comparison Based on MAE")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()

# RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.ylabel("RMSE")
plt.title("Model Comparison Based on RMSE")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()


# %% Actual vs Predicted line plot

n = 100

plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:n], label="Actual", linewidth=2)
plt.plot(y_pred[:n], label="Linear Regression", alpha=0.8)
plt.plot(y_pred_rf[:n], label="Random Forest", alpha=0.8)
plt.plot(y_pred_svr[:n], label="SVR", alpha=0.8)
plt.plot(y_pred_mlp[:n], label="MLP", alpha=0.8)
plt.plot(y_pred_xgb[:n], label="XGBoost", alpha=0.8)

plt.xlabel("Test Sample Index")
plt.ylabel("Generated Power (kW)")
plt.title("Actual vs Predicted Power for First 100 Test Samples")
plt.legend()
plt.grid(True)
plt.show()


# %% Residual distribution comparison

plt.figure(figsize=(9, 5))
plt.hist(residuals, bins=30, alpha=0.5, label="Linear Regression")
plt.hist(residuals_rf, bins=30, alpha=0.5, label="Random Forest")
plt.hist(residuals_svr, bins=30, alpha=0.5, label="SVR")
plt.hist(residuals_mlp, bins=30, alpha=0.5, label="MLP")
plt.hist(residuals_xgb, bins=30, alpha=0.5, label="XGBoost")
plt.axvline(0, color='r', linestyle='--')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution Comparison")
plt.legend()
plt.grid(True)
plt.show()


# %%
