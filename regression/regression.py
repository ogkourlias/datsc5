#!/usr/bin/env python3

"""
    usage:
        python3 ./regr.py
"""

# METADATA VARIABLES
__author__ = "Orfeas Gkourlias"
__status__ = "WIP"
__version__ = "0.2"

# IMPORTS
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin, BaseEstimator


# CLASSES
class Regr(BaseEstimator, RegressorMixin):
    """Regression class"""
    def __init__(self, method="linear", regular="ridge", lambda_=0.5,
                 alpha=0.01, num_iters=500, tol=1e-8,
                 threshold=0.5, scale_X=False):
        self.method = method
        self.regular = regular
        self.lambda_ = lambda_
        self.alpha = alpha
        self.num_iters = num_iters
        self.tol = tol
        self.threshold = threshold
        self.scale_X = scale_X

    def get_params(self, deep=True):
        """
        Return estimator parameters for cloning and grid search compatibility.
        """
        return {
            "method": self.method,
            "regular": self.regular,
            "lambda_": self.lambda_,
            "alpha": self.alpha,
            "num_iters": self.num_iters,
            "tol": self.tol,
            "threshold": self.threshold,
            "scale_X": self.scale_X,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator. Returns self.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                # still set it to allow new params if user provides them
                setattr(self, key, value)
            else:
                setattr(self, key, value)
        return self

    ### General ###
    def _sigmoid(self, z):
        """
        Return sigmoid of Z
        """
        sig = 1 / (1 + np.exp(-z))
        return sig

    def _compute_cost(self, X, y, y_hat):
        """
        Return MSE cost for linear regression with ridge regularization.
        Or return log cost for logistic regression with ridge regularization.
        """
        m = len(y)
        reg = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        if self.method == "linear":
            residuals = y_hat - y
            mse = np.sum(residuals ** 2) / (2 * m)
            cost = mse + reg
            return cost
        elif self.method == "logistic":
            eps = 1e-8
            log = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            cost = log + reg
            return cost

    def _compute_gradient(self, X, y, y_hat):
        """
        Compute gradient for linear or logistic regression with optional ridge regularization.
        """
        m = len(y)
        loss = y_hat - y
        grad = (X.T @ loss) / m
        grad[1:] += (self.lambda_ / m) * self.theta[1:]
        return grad

    def _predict_raw(self, X):
        """
        Predict Ys using parameters (x matrix) and theta (coefficients)
        """
        y_hat = X @ self.theta
        if self.method == "logistic":
            y_hat = self._sigmoid(y_hat)
        return y_hat

    def probs_to_class(self, y_hat):
        """
        Converts logistic regression probabilities to actual binary classifiers
        """
        return np.array([1.0 if h >= self.threshold else 0.0 for h in y_hat])

    def fit(self, X, y):
        """
        Fitting model with or without regularization.
        L1 = Lasso (not implemented).
        L2 = Ridge (default).

        Important: Be conscious over whether the input x and theta should be scaled.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.scale_X:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        X_b = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        self.theta = np.zeros(X_b.shape[1])
        self.cost_history = np.zeros(self.num_iters)

        if self.regular == "ridge":
            for i in range(self.num_iters):
                y_hat = self._predict_raw(X_b)
                grad = self._compute_gradient(X_b, y, y_hat)
                self.theta = self.theta - self.alpha * grad
                new_y_hat = self._predict_raw(X_b)
                self.cost_history[i] = self._compute_cost(X_b, y, new_y_hat)
                if i > 0 and abs(self.cost_history[i] - self.cost_history[i - 1]) < self.tol:
                    self.cost_history = self.cost_history[:i + 1]
                    break

        if self.scale_X:
            self.reverse_theta()

        return self

    def predict(self, X, prob=False):
        """
        Predict Ys using parameters (x matrix) and theta (coefficients)
        """
        X = np.asarray(X, dtype=float)
        if self.scale_X and getattr(self, "scaler", None) is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        X_b = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        y_hat = self._predict_raw(X_b)
        if self.method == "logistic":
            if prob:
                return y_hat
            return self.probs_to_class(y_hat)
        return y_hat

    def score(self, X, y):
        """
        Return R^2 score for linear regression,
        or accuracy for logistic regression.
        """
        y_pred = self.predict(X)
        if self.method == "linear":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        elif self.method == "logistic":
            return np.mean(y_pred == y)

    ### Scaling ###
    def reverse_theta(self):
        """
        Transforms the theta parameters obtained from a regression model on scaled data
        back to the original (non-scaled) feature space.
        """
        theta_original = np.zeros_like(self.theta)
        theta_original[0] = self.theta[0] - np.sum(
            (self.theta[1:] * self.scaler.mean_) / self.scaler.scale_
        )
        theta_original[1:] = self.theta[1:] / self.scaler.scale_
        self.theta = theta_original


# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
