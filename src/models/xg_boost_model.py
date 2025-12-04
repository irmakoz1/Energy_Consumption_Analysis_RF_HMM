import joblib
import numpy as np
from xgboost import XGBRegressor
import pandas as pd


def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model
