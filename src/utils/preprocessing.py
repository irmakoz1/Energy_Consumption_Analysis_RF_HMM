import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class EnergyDataLoader:
    """Class to handle loading and preprocessing of ASHRAE datasets."""

    def __init__(self, root_dir):
        self.root = Path(root_dir)

    def load_A_data(self, file_name):
        df = pd.read_csv(self.root / file_name, delim_whitespace=True)
        df['Datetime'] = pd.to_datetime({
            'year': df['YEAR'] + 1900,
            'month': df['MONTH'],
            'day': df['DAY'],
            'hour': (df['HOUR'] // 100).astype(int),
            'minute': 0
        })
        df.set_index('Datetime', inplace=True)
        df.drop(['YEAR', 'MONTH', 'DAY', 'HOUR'], axis=1, inplace=True)
        df['hourOfDay'] = df.index.hour
        df['dayOfWeek'] = df.index.dayofweek
        df['month_of_day'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        return df

    def load_B_data(self, file_name, is_train=True):
        cols = ['Date', 'HorizRad', 'SE_Rad', 'S_Rad', 'SW_Rad']
        if is_train:
            cols.append('Beam_Rad')
        df = pd.read_csv(self.root / file_name, header=None, delim_whitespace=True)
        df.columns = cols
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df['Day'] = np.floor(df['Date'])
        df['Time'] = df['Date'] - df['Day']
        df.drop('Date', axis=1, inplace=True)
        return df

    def preprocess_data(self, df, feature_cols, val_ratio=0.2, target="WBE"):
        """
        Preprocess dataframe for model training and create train/validation split.

        Args:
        df (pd.DataFrame): Input dataframe (time series must be sorted by date)
        feature_cols (list of str): Columns to use as features
        val_ratio (float): Fraction of data to reserve for validation / forward forecast
        target (str): Name of the target column

        Returns:
        df_train, df_val (pd.DataFrame): Train/validation dataframes
        X_train, X_val (np.ndarray): Original features
        X_train_scaled, X_val_scaled (np.ndarray): Scaled features
        y_train, y_val (np.ndarray): Target values
        scaler (StandardScaler): Fitted scaler on training data
        """
        df = df.copy()

    # Feature matrix and target
        X = df[feature_cols].values
        y = df[target].values

    # Train/validation split (time-based)
        split_idx = int(len(df) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

    # Scale features: fit scaler on training only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        return df_train, df_val, X_train, X_val, X_train_scaled, X_val_scaled, y_train, y_val, scaler

