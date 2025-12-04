import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ----------------------------------------------------------------------
# 1. Forecast vs Actual
# ----------------------------------------------------------------------
def plot_forecast(y_true, y_pred, model_name, save_path=None):
    plt.figure(figsize=(14, 5))
    plt.plot(y_true.index, y_true, label="Actual", linewidth=2)
    plt.plot(y_true.index, y_pred, label="Predicted", alpha=0.7)
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# 2. Residual Plot
# ----------------------------------------------------------------------
def plot_residuals(y_true, y_pred, model_name, save_path=None):
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 4))
    plt.plot(y_true.index, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{model_name}: Residuals")
    plt.ylabel("Residual")
    plt.xlabel("Time")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# 3. Feature Importance
# ----------------------------------------------------------------------
def plot_feature_importance(model, feature_names, model_name, save_path=None):
    if not hasattr(model, "feature_importances_"):
        print(f"Model {model_name} has no feature_importances_. Skipping.")
        return

    importance = model.feature_importances_
    idx = np.argsort(importance)

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx], importance[idx])
    plt.title(f"{model_name}: Feature Importance")
    plt.xlabel("Importance")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# 4. Hidden State Sequence Plot
# ----------------------------------------------------------------------
def plot_hidden_states(df, save_path=None):
    plt.figure(figsize=(15, 4))
    df['hidden_state'].plot()
    plt.title("Hidden States Over Time")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# 5. Hidden State Means (heatmap)
# ----------------------------------------------------------------------
def plot_state_means(means_df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.heatmap(means_df, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Mean Feature Values by Hidden State")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
