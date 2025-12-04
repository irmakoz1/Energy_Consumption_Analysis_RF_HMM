import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd



def evaluate_model(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return {
        "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
        "RMSE": mean_squared_error(y_true[mask], y_pred[mask], squared=False),
        "MAPE": np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100,
        "R2": r2_score(y_true[mask], y_pred[mask]),
    }