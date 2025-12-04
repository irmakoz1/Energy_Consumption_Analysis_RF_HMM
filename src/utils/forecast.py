import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor



def forecast_hmm_ar(hmm_model, hmm_ar_models, X_scaled, scaler):
    hidden_states = hmm_model.predict(X_scaled)
    preds = []

    for i, st in enumerate(hidden_states):
        if st in hmm_ar_models:
            try:
                preds.append(hmm_ar_models[st].forecast(steps=1)[0])
            except:
                preds.append(np.nan)
        else:
            preds.append(np.nan)

    return np.array(preds)


def forecast_model(model, X):
    return model.predict(X)


def forecast_model_hmm(model, X, hidden_states):
    Xh = np.column_stack([X, hidden_states])
    return model.predict(Xh)