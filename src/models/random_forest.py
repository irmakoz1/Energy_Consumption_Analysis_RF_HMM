
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300, random_state=42
    )
    model.fit(X_train, y_train)
    return model
