import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd

import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def train_hmm_ar(df, target_col="WBE"):
    """
    Simple AR on each hidden state.
    """
    models = {}
    for state in df["hidden_state"].unique():
        state_series = df[df["hidden_state"] == state][target_col]
        if len(state_series) > 10:
            models[state] = ARIMA(state_series, order=(2, 0, 2)).fit()
    return models


def train_hmm_rf(X, hidden_states, y):
    X_hmm = np.column_stack([X, hidden_states])
    model = RandomForestRegressor(
        n_estimators=300, random_state=42
    )
    model.fit(X_hmm, y)
    return model


def train_hmm_xgb(X, hidden_states, y):
    X_hmm = np.column_stack([X, hidden_states])
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5
    )
    model.fit(X_hmm, y)
    return model

