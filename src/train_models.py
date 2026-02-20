import numpy as np
import pandas as pd

import src.utils.preprocessing as preprocessing
import src.models.hmm_model as hmm_model
import src.models.LR_model as lr_model
import src.models.random_forest as rf_model
import src.models.xg_boost_model as xgb_model
import src.models.hmm_augmented as hmm_augmented_model
import src.utils.forecast as forecast_models
import src.utils.evaluation as evaluate

def pipeline(df, target="WBE", feature_cols=None, val_ratio=1/12):
    """
    Run the full pipeline:
    - train/validation split (time series)
    - feature scaling
    - HMM + baseline models
    - HMM-augmented models
    - Forecast & evaluation
    """

    if feature_cols is None:
        feature_cols = ["TEMP", "HUMID", "SOLAR", "WIND", "WBCW", "WBHW", "hourOfDay"]

    #Preprocess & split
    df_train, df_val, X_train, X_val, X_train_scaled, X_val_scaled, y_train, y_val, scaler = \
        preprocessing.preprocess_data(df, feature_cols, val_ratio=val_ratio, target=target)

    # Find optimal HMM states using training data only
    optimal_n = hmm_model.find_optimal_hmm_states(X_train_scaled, 2, 10)
    print("Optimal HMM States:", optimal_n)

    # Train HMM
    hmm_model_f = hmm_model.train_hmm(X_train_scaled, optimal_n)
    df_train = hmm_model.add_hidden_states(df_train, hmm_model_f, X_train_scaled)
    hidden_states_train = df_train["hidden_state"].values


    lr = lr_model.train_linear_regression(X_train, y_train)
    rf_m = rf_model.train_random_forest(X_train, y_train)
    xgbm = xgb_model.train_xgboost(X_train, y_train)


    hmm_ar_models = hmm_augmented_model.train_hmm_ar(df_train, target)
    hmm_rf_m = hmm_augmented_model.train_hmm_rf(X_train, hidden_states_train, y_train)
    hmm_xgbm_m = hmm_augmented_model.train_hmm_xgb(X_train, hidden_states_train, y_train)

    pred_lr = forecast_models.forecast_model(lr, X_val)
    pred_rf = forecast_models.forecast_model(rf_m, X_val)
    pred_xgb = forecast_models.forecast_model(xgbm, X_val)

    pred_hmm_ar = forecast_models.forecast_hmm_ar(hmm_model_f, hmm_ar_models, X_val_scaled, scaler)
    pred_hmm_rf = forecast_models.forecast_model_hmm(hmm_rf_m, X_val,
                                                     hmm_model.predict_hidden_states(X_val_scaled))
    pred_hmm_xgb = forecast_models.forecast_model_hmm(hmm_xgbm_m, X_val,
                                                      hmm_model.predict_hidden_states(X_val_scaled))


    results = {
        "Linear Regression": evaluate.evaluate_model(y_val, pred_lr),
        "Random Forest": evaluate.evaluate_model(y_val, pred_rf),
        "XGBoost": evaluate.evaluate_model(y_val, pred_xgb),
        "HMM + AR": evaluate.evaluate_model(y_val, pred_hmm_ar),
        "HMM + RF": evaluate.evaluate_model(y_val, pred_hmm_rf),
        "HMM + XGBoost": evaluate.evaluate_model(y_val, pred_hmm_xgb),
    }

    # RETURN ARTIFACTS FOR PLOTTING
    artifacts = {
        "y_val": y_val,
        "X_val": X_val,
        "X_val_scaled": X_val_scaled,
        "hidden_states_val": hmm_model.predict_hidden_states(X_val_scaled),
        "scaler": scaler,
        "models": {
            "Linear Regression": lr,
            "Random Forest": rf_m,
            "XGBoost": xgbm,
            "HMM + AR": hmm_ar_models,
            "HMM + RF": hmm_rf_m,
            "HMM + XGBoost": hmm_xgbm_m,
        },
        "predictions": {
            "Linear Regression": pred_lr,
            "Random Forest": pred_rf,
            "XGBoost": pred_xgb,
            "HMM + AR": pred_hmm_ar,
            "HMM + RF": pred_hmm_rf,
            "HMM + XGBoost": pred_hmm_xgb,
        },
    }

    return df_train, df_val, results, artifacts

