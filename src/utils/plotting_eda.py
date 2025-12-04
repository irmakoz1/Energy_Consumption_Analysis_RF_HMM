
import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


class EnergyVisualizer:
    """Class to handle plotting of energy time series and correlation matrices."""

    @staticmethod
    def plot_time_series(df, target_col, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[target_col], label=target_col)
        plt.xlabel('Datetime')
        plt.ylabel(target_col)
        plt.title(f'{target_col} over Time')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df, save_path=None):
        corr_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.title('Correlation Matrix')
        plt.show()
