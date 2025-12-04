
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PCAAnalyzer:
    """Class to perform PCA analysis and visualize feature importance."""

    def __init__(self, df, target_cols):
        self.df = df
        self.target_cols = target_cols
        self.scaler = StandardScaler()
        self.loadings = None

    def run_pca(self, n_components=None):
        features = [col for col in self.df.columns if col not in self.target_cols]
        X_scaled = self.scaler.fit_transform(self.df[features])
        self.features = features

        pca = PCA(n_components=n_components)
        self.pca = pca.fit(X_scaled)
        self.X_scaled = X_scaled

        # Create loadings DataFrame
        self.loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=features
        )
        return self.loadings

    def plot_variable_importance(self):
        explained_var = self.pca.explained_variance_ratio_
        importance = np.sum((self.loadings**2).values * explained_var, axis=1)
        importance_df = pd.DataFrame({'Variable': self.features, 'Importance': importance})
        importance_df.sort_values('Importance', ascending=False, inplace=True)
        importance_df['Cumulative'] = importance_df['Importance'].cumsum()

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Variable'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Contribution to Total Variance')
        plt.title('Variable Importance in PCA (Weighted by Explained Variance)')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return importance_df

    def plot_explained_variance(self):
        cum_var = self.pca.explained_variance_ratio_.cumsum()
        n_components_95 = (cum_var >= 0.95).argmax() + 1

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.show()
        return n_components_95

    def plot_loadings_heatmap(self, n_components=5):
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.loadings.iloc[:, :n_components], cmap='coolwarm', center=0, annot=True, fmt=".2f")
        plt.title(f"PCA Loadings: First {n_components} Principal Components")
        plt.show()

    def top_related_variables(self, target):
        target_vector = self.loadings.loc[target].values.reshape(1, -1)
        similarities = cosine_similarity(self.loadings.values, target_vector).flatten()
        sim_df = pd.DataFrame({'Variable': self.loadings.index, 'Cosine Similarity': similarities})
        sim_df = sim_df.sort_values(by='Cosine Similarity', ascending=False)
        return sim_df.iloc[1:11]  # Skip the target itself