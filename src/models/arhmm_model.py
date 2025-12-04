# src/models/arhmm.py
import numpy as np

class ARHMM:
    """
    Autoregressive HMM with Gaussian emissions that depend on AR lags + covariates.
    Simplified EM implementation for moderate T and small number of states.
    """

    def __init__(self, n_states, ar_order, n_covariates, seed=None):
        self.n_states = int(n_states)
        self.ar_order = int(ar_order)
        self.n_covariates = int(n_covariates)
        rng = np.random.RandomState(seed)

        # Transition matrix: rows sum to 1
        A = rng.rand(self.n_states, self.n_states)
        self.trans_mat = A / A.sum(axis=1, keepdims=True)

        # Parameters: for each state, an intercept, AR coefs, covariate coefs
        self.means = rng.randn(self.n_states) * 0.1
        self.ar_coefs = rng.randn(self.n_states, self.ar_order) * 0.1
        self.cov_coefs = rng.randn(self.n_states, self.n_covariates) * 0.1
        self.vars = np.ones(self.n_states) * 1.0  # state noise variances

        # training traces
        self.log_likelihoods_ = []

    def _emission_logprob(self, y_t, y_lags, x_t):
        """
        Return log p(y_t | state s) for every state s.
        """
        # shape: (n_states,)
        means = self.means + np.einsum('sd,d->s', self.ar_coefs, y_lags) + np.einsum('sd,d->s', self.cov_coefs, x_t)
        var = self.vars
        # Gaussian log-likelihood per state
        lp = -0.5 * (np.log(2 * np.pi * var) + (y_t - means) ** 2 / var)
        return lp, means

    def _compute_emission_matrix(self, Y, X):
        T = len(Y)
        p = self.ar_order
        emit_log = np.full((T, self.n_states), -1e10)
        for t in range(p, T):
            y_lags = Y[t - p:t][::-1]  # most recent first
            lp, _ = self._emission_logprob(Y[t], y_lags, X[t])
            emit_log[t] = lp
        return emit_log

    def _forward_backward(self, emit_log):
        """
        Returns gamma (T x n_states), xi (T-1 x n_states x n_states), and log-likelihood
        using log-space forward-backward with scaling for stability.
        """
        T = emit_log.shape[0]
        K = self.n_states

        # log pi: uniform initial prior in log-space
        log_pi = -np.log(K) * np.ones(K)

        # forward
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_pi + emit_log[0]
        for t in range(1, T):
            # log-sum-exp over previous states
            prev = log_alpha[t - 1][:, None] + np.log(self.trans_mat)  # (K, K)
            log_alpha[t] = emit_log[t] + logsumexp(prev, axis=0)

        # backward
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            elem = np.log(self.trans_mat) + emit_log[t + 1] + log_beta[t + 1]
            log_beta[t] = logsumexp(elem, axis=1)

        # log-likelihood from final alpha using log-sum-exp
        loglik = logsumexp(log_alpha[-1])

        # gamma: posterior state probs
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # xi: t from 0..T-2
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            # compute unnormalized log xi
            log_xi_t = (
                log_alpha[t][:, None] +
                np.log(self.trans_mat) +
                emit_log[t + 1][None, :] +
                log_beta[t + 1][None, :]
            )
            log_xi_t -= logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)
        return gamma, xi, loglik

    def fit(self, Y, X, n_iter=50, tol=1e-4, verbose=False):
        """
        EM training. Y: 1D array length T. X: (T, n_covariates).
        """
        Y = np.asarray(Y)
        X = np.asarray(X)
        T = len(Y)
        p = self.ar_order
        assert X.shape[0] == T

        prev_ll = -np.inf
        self.log_likelihoods_ = []

        for it in range(n_iter):
            emit_log = self._compute_emission_matrix(Y, X)
            gamma, xi, loglik = self._forward_backward(emit_log)

            # M-step
            # Update transition matrix
            sum_xi = xi.sum(axis=0)  # (K, K)
            self.trans_mat = sum_xi / sum_xi.sum(axis=1, keepdims=True)

            # Update regression parameters per state via weighted least squares
            for s in range(self.n_states):
                weights = gamma[p:, s]  # length T-p
                # Build design matrix: intercept + AR lags + covariates for t=p..T-1
                rows = []
                ys = []
                for t in range(p, T):
                    y_lags = Y[t - p:t][::-1]
                    rows.append(np.concatenate(([1.0], y_lags, X[t])))
                    ys.append(Y[t])
                X_reg = np.vstack(rows)  # shape (T-p, 1+p+n_covariates)
                y_reg = np.array(ys)

                W = np.diag(weights + 1e-8)  # avoid zeros
                XtWX = X_reg.T @ W @ X_reg
                XtWy = X_reg.T @ W @ y_reg
                # Solve safely with pseudo-inverse fallback
                try:
                    beta = np.linalg.solve(XtWX, XtWy)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(XtWX) @ XtWy

                self.means[s] = beta[0]
                self.ar_coefs[s] = beta[1:1 + p]
                self.cov_coefs[s] = beta[1 + p:]

                # update variance
                resid = y_reg - X_reg @ beta
                self.vars[s] = np.sum(weights * resid ** 2) / (weights.sum() + 1e-8)
                self.vars[s] = max(self.vars[s], 1e-6)  # keep positive

            self.log_likelihoods_.append(loglik)
            if verbose:
                print(f"EM iter {it+1}, loglik = {loglik:.6f}")

            if it > 0 and abs(loglik - prev_ll) < tol:
                if verbose:
                    print("Converged.")
                break
            prev_ll = loglik
        return self

    def predict_states(self, Y, X):
        """
        Return most likely state (Viterbi) sequence for data Y, X.
        """
        # Simple Viterbi in logspace using emission log-probs
        Y = np.asarray(Y)
        X = np.asarray(X)
        T = len(Y)
        p = self.ar_order
        emit_log = self._compute_emission_matrix(Y, X)
        K = self.n_states

        delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)

        delta[0] = -np.log(K) + emit_log[0]
        for t in range(1, T):
            for j in range(K):
                choices = delta[t - 1] + np.log(self.trans_mat[:, j])
                psi[t, j] = np.argmax(choices)
                delta[t, j] = choices[psi[t, j]] + emit_log[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def forecast(self, Y_init, X_future, n_steps):
        """
        Forecast mean predictions for n_steps using last Y_init (length ar_order) and future X.
        Returns predicted values and predicted most-likely state sequence.
        """
        Y_init = list(Y_init[-self.ar_order:])
        preds = []
        states = []
        for t in range(n_steps):
            x_t = X_future[t]
            y_lags = np.array(Y_init[-self.ar_order:][::-1])
            means = self.means + np.einsum('sd,d->s', self.ar_coefs, y_lags) + np.einsum('sd,d->s', self.cov_coefs, x_t)
            s = np.argmax(means)
            y_hat = means[s]
            preds.append(y_hat)
            states.append(s)
            Y_init.append(y_hat)
        return np.array(preds), np.array(states)


# helper: stable log-sum-exp
def logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    res = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        res = np.squeeze(res, axis=axis)
    return res
