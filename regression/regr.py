#!/usr/bin/env python3

"""
    usage:
        python3 ./regr.py
"""

# METADATA VARIABLES
__author__ = "Orfeas Gkourlias"
__status__ = "WIP"
__version__ = "0.1"

# IMPORTS
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler


# CLASSES
class Regr:
    """Regression class"""

    def __init__(self) -> None:
        pass

    ### General ###
    def prep_data(self, df, y_col):
        """
        Prepare data for regression, assuming the following:
        1. All features in provided dataframe are used in model.
        2. Features are not co-linear.
        3. Y is known and provided (provide column name as string)
        """

        # Initialize theta, theta[0] = intercept theta.
        y = np.array(df[y_col])
        df = df.drop(columns=[y_col])
        x = np.concatenate([np.ones((df.shape[0], 1)), df.to_numpy()], axis=1)
        theta = np.zeros(x.shape[1])
        return x, y, theta

    def get_cost(self, x, y, theta):
        """
        Return mean-squared error (MSE) of model predictions vs. actual Ys
        """
        mse_vec = sum((x @ theta - y) ** 2) / (2 * len(y))
        return mse_vec

    ### Scaling ###
    def scale_x(self, x):
        """
        Scale and return Xs
        """
        scaler = StandardScaler()
        m, n = x.shape

        theta = np.zeros(n)

        # Skip intersect here
        x_features_scaled = scaler.fit_transform(x[:, 1:])
        x_scaled = np.c_[np.ones(m), x_features_scaled]
        return x_scaled, theta, scaler

    def reverse_theta(self, theta, scaler):
        """
        Transforms the theta parameters obtained from a regression model on scaled data
        back to the original (non-scaled) feature space.
        """
        # initialize
        theta_original = np.zeros_like(theta)
        # tranform back intercept
        theta_original[0] = theta[0] - np.sum(
            (theta[1:] * scaler.mean_) / scaler.scale_
        )
        # transform back coefficients
        theta_original[1:] = theta[1:] / scaler.scale_

        return theta_original

    ### Linear Regression Gradient Descent
    def gradient_descent_linear(
        self, x, y, theta, alpha=0.001, num_iters=100, regular="ridge", lambda_=0.5
    ):
        """
        Gradient descent with or without regularization.
        L1 = Lasso.
        L2 = Ridge.
        Otherwise, no regularization.
        """
        # initialize list of costs
        m = len(y)
        cost_history = np.zeros(num_iters)
        if regular == "ridge":
            for i in range(num_iters):
                grad = (x.T @ ((x @ theta) - y)) / m
                reg = (lambda_ / m) * theta
                reg[0] = 0
                theta = theta - alpha * (grad + reg)
                cost_history[i] = (
                    np.sum((x @ theta - y) ** 2) + lambda_ * np.sum(theta[1:] ** 2)
                ) / (2 * m)
        else:
            for i in range(0, num_iters):
                theta = theta - alpha / y.size * (x.T @ ((x @ theta) - y))
                cost_history[i] = sum((x @ theta - y) ** 2) / (2 * len(y))

        return theta, cost_history

    def sigmoid(self, z):
        """
        Return sigmoid of Z
        """
        sig = 1 / (1 + np.exp(-z))
        return sig

    def compute_cost_logistic(self, x, y, theta, lambda_):
        """
        Compute cost for logistic regression with regularization.

        Parameters:
        X:  Input feature matrix (m x n)
        y: True labels vector (m,)
        theta: Parameters vector (n,)
        lambda_: Regularization parameter

        Return:
        J:Cost value
        """
        m = len(y)
        h = self.sigmoid(x.dot(theta))
        reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        cost = -1 / m * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
        return cost + reg

    def gradient_descent_logistic(self, x, y, theta, alpha, num_iters, lambda_):
        """
        Perform gradient descent to find optimal theta.

        Parameters:
        X: Input feature matrix (m x n)
        y: True labels vector (m,)
        theta: Initial parameters vector (n,)
        alpha: Learning rate
        num_iters: Number of iterations
        lambda_:  Regularization parameter

        Return:
        theta:  Updated parameters vector
        J_history:  History of cost values
        """

        m = len(y)
        cost_history = np.zeros(num_iters)
        for i in range(num_iters):
            grad = (x.T @ ((self.sigmoid(x @ theta)) - y)) / m
            reg = (lambda_ / m) * theta
            reg[0] = 0
            theta = theta - alpha * (grad + reg)
            cost_history[i] = self.compute_cost_logistic(x, y, theta, lambda_)
        return theta, cost_history

    def predict_logistic(self, x, theta, threshold=0.5):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters theta.

        Parameters:
        X: Input feature matrix (m x n)
        theta: Parameters vector (n,) (computed by compute cost and gradient descentl)
        threshold: Threshold for prediction

        Return:
        p: Predicted labels vector (m,)
        """
        pred = self.sigmoid(x.dot(theta))
        p = [1.0 if h >= threshold else 0.0 for h in pred]
        return p


# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
