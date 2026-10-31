# random_forest.py - random forest regressor model and plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

VIEWPOINT = "Non-linear ensemble: combines multiple deep decision trees to capture complex state-crop interactions."


def train_random_forest(x_train, y_train, x_test, y_test, seed=42):
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    mse = mean_squared_error(y_test, pred_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred_test),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 (test)": r2_score(y_test, pred_test),
        "R2 (train)": r2_score(y_train, pred_train),
        "pred": pred_test,
        "y_test": np.array(y_test),
    }
    return model, metrics


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted (Random Forest)"):
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    ax.scatter(y_true, y_pred, s=6, color="#0d9488", alpha=0.35)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xlabel("Actual log price", fontsize=8)
    ax.set_ylabel("Predicted log price", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_series):
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    importance_series.sort_values().plot.barh(ax=ax, color="#059669", alpha=0.9)
    ax.set_title("Aggregated Feature Importance", fontsize=8, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
