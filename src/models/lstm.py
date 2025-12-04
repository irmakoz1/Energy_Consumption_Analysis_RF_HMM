import numpy as np
import src.utils.preprocessing as preprocessing
import src.models.hmm_model as hmm_model
import src.models.LR_model as LR
import src.models.random_forest as rf
import src.models.xg_boost_model as train_xgboost
import src.models.hmm_augmented as hmm_augmented
import src.utils.forecast as forecast_models
import src.utils.evaluation as evaluate
import pandas as pd
import src.models.PCA_model as pca_model
import src.models.random_forest as random_forest
import src.models.xg_boost_model as xg_boost_model
import src.models.LR_model as lr_model
import src.models.hmm_augmented as hmm_augmented_model
import src.models.hmm_model as hmm_model

# Import your LSTM module
from src.models.LSTM_module import train_lstm_model, forecast_lstm

def pipeline(df_train, target="WBE", seq_len=24):
    feature_cols = ["TEMP", "HUMID", "SOLAR", "WIND", "WBCW", "WBHW", "hourOfDay"]

    # ----- Preprocess
    df_train, X, X_scaled, scaler = preprocessing.preprocess_data(df_train, feature_cols)
    y = df_train[target].values

    # ----- Find optimal HMM states
    optimal_n = hmm_model.find_optimal_hmm_states(X_scaled, 2, 10)
    print("Optimal HMM States:", optimal_n)

    # ----- Train HMM
    hmm_model_f = train_hmm(X_scaled, optimal_n)
    df_train = add_hidden_states(df_train, hmm_model_f, X_scaled)
    hidden_states = df_train["hidden_state"].values

    # ======================================================
    # BASELINE MODELS
    # ======================================================
    lr = train_linear_regression(X, y)
    rf_m = train_random_forest(X, y)
    xgbm = train_xgboost(X, y)

    # ======================================================
    # HMM-AUGMENTED MODELS
    # ======================================================
    hmm_ar_models = train_hmm_ar(df_train, target)
    hmm_rf_m = train_hmm_rf(X, hidden_states, y)
    hmm_xgbm_m = train_hmm_xgb(X, hidden_states, y)

    # ======================================================
    # LSTM MODEL
    # ======================================================
    lstm_model, lstm_seq_len = train_lstm_model(X_scaled, y, seq_len=seq_len)
    pred_lstm = forecast_lstm(lstm_model, X_scaled, lstm_seq_len)

    # Pad predictions to match original length
    pred_lstm_full = np.concatenate([np.full(seq_len, np.nan), pred_lstm])

    # ======================================================
    # FORECAST
    # ======================================================
    pred_lr = forecast_model(lr, X)
    pred_rf = forecast_model(rf_m, X)
    pred_xgb = forecast_model(xgbm, X)

    pred_hmm_ar = forecast_hmm_ar(hmm_model_f, hmm_ar_models, X_scaled, scaler)
    pred_hmm_rf = forecast_model_hmm(hmm_rf_m, X, hidden_states)
    pred_hmm_xgb = forecast_model_hmm(hmm_xgbm_m, X, hidden_states)

    # ======================================================
    # Evaluation
    # ======================================================
    results = {
        "Linear Regression": evaluate_model(y, pred_lr),
        "Random Forest": evaluate_model(y, pred_rf),
        "XGBoost": evaluate_model(y, pred_xgb),
        "HMM + AR": evaluate_model(y, pred_hmm_ar),
        "HMM + RF": evaluate_model(y, pred_hmm_rf),
        "HMM + XGBoost": evaluate_model(y, pred_hmm_xgb),
        "LSTM": evaluate_model(y, pred_lstm_full),
    }

    # ------------------------------------------------------
    # RETURN ARTIFACTS FOR PLOTTING
    # ------------------------------------------------------
    artifacts = {
        "y": y,
        "X": X,
        "X_scaled": X_scaled,
        "hidden_states": hidden_states,
        "scaler": scaler,
        "models": {
            "Linear Regression": lr,
            "Random Forest": rf_m,
            "XGBoost": xgbm,
            "HMM + AR": hmm_ar_models,
            "HMM + RF": hmm_rf_m,
            "HMM + XGBoost": hmm_xgbm_m,
            "LSTM": lstm_model,
        },
        "predictions": {
            "Linear Regression": pred_lr,
            "Random Forest": pred_rf,
            "XGBoost": pred_xgb,
            "HMM + AR": pred_hmm_ar,
            "HMM + RF": pred_hmm_rf,
            "HMM + XGBoost": pred_hmm_xgb,
            "LSTM": pred_lstm_full,
        },
    }

    return df_train, results, artifacts
