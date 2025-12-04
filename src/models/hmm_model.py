import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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




class EnergyHMM:
    """
    Wrapper for Gaussian HMM, supporting:
    - feature scaling
    - BIC-based model selection
    - hidden state prediction
    - visualization of states & transitions
    """

    def __init__(self, df, feature_cols, random_state=42):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.X = df[feature_cols].values
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.model = None
        self.hidden_states = None
        self.state_means_df = None

    # -----------------------------
    def _check_fitted(self):
        if self.model is None:
            raise ValueError("HMM model is not fitted. Call .fit(n_states) first.")

    # -----------------------------
    def select_optimal_states(self, min_states=2, max_states=10):
        """Compute BIC for different state counts and return optimum."""
        bic_scores = []

        for n in range(min_states, max_states + 1):
            model = GaussianHMM(
                n_components=n,
                covariance_type='full',
                n_iter=500,
                random_state=self.random_state
            )

            model.fit(self.X_scaled)
            logL = model.score(self.X_scaled)

            n_features = self.X_scaled.shape[1]
            n_params = n * n + 2 * n * n_features - 1

            bic = -2 * logL + n_params * np.log(len(self.X_scaled))
            bic_scores.append((n, bic))

        best_n = min(bic_scores, key=lambda x: x[1])[0]
        return best_n, bic_scores

    # -----------------------------
    def fit(self, n_states):
        """Fit a Gaussian HMM with chosen number of states."""
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=500,
            random_state=self.random_state,
        )

        self.model.fit(self.X_scaled)
        self.hidden_states = self.model.predict(self.X_scaled)
        self.df['hidden_state'] = self.hidden_states

    # -----------------------------
    def predict_hidden_states(self, new_df):
        """Predict hidden states for new unseen data."""
        self._check_fitted()
        X_new = new_df[self.feature_cols].values
        X_new_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_new_scaled)

    # -----------------------------
    def add_hidden_states_to_df(self, df):
        """Return df with hidden_state column added."""
        df = df.copy()
        df["hidden_state"] = self.predict_hidden_states(df)
        return df

    # -----------------------------
    def compute_state_means(self):
        """Inverse-transform state means to original units."""
        self._check_fitted()
        means_original = self.scaler.inverse_transform(self.model.means_)
        self.state_means_df = pd.DataFrame(
            means_original,
            columns=self.feature_cols
        )
        self.state_means_df.index.name = "Hidden State"
        return self.state_means_df

    # -----------------------------
    def get_artifacts(self):
        self._check_fitted()
        return {
            "model": self.model,
            "scaler": self.scaler,
            "X": self.X,
            "X_scaled": self.X_scaled,
            "hidden_states": self.hidden_states,
            "state_means": self.state_means_df
        }

    # ----------------------------- Plotting -----------------------------
    def plot_hidden_states(self):
        self._check_fitted()
        plt.figure(figsize=(15, 4))
        self.df['hidden_state'].plot()
        plt.title("Hidden States Over Time")
        plt.grid(True)
        plt.show()

    def plot_state_means_heatmap(self):
        self._check_fitted()
        if self.state_means_df is None:
            self.compute_state_means()

        plt.figure(figsize=(12, 6))
        sns.heatmap(self.state_means_df.round(2), annot=True, cmap="coolwarm")
        plt.title("Mean Feature Values by Hidden State")
        plt.show()

    def plot_bic(self, bic_scores, best_n):
        n_states = [n for n, _ in bic_scores]
        bics = [b for _, b in bic_scores]

        plt.figure(figsize=(8, 5))
        plt.plot(n_states, bics, marker='o')
        plt.axvline(best_n, color='red', linestyle='--', label=f'Best: {best_n}')
        plt.xlabel("Number of Hidden States")
        plt.ylabel("BIC Score")
        plt.title("BIC Scores for Different Hidden States")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_transition_diagram(self, threshold=0.05):
        self._check_fitted()
        transmat = self.model.transmat_
        n = transmat.shape[0]
        G = nx.DiGraph()

        for i in range(n):
            for j in range(n):
                w = transmat[i, j]
                if w > threshold:
                    G.add_edge(f"State {i}", f"State {j}", weight=round(w, 2))

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=self.random_state)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("HMM State Transition Diagram")
        plt.show()
