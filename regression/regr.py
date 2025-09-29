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

    def compute_loss(self, y_hat, y):
        """
        Return array of residuals between Y predictions and actual Y's
        """
        loss = y_hat - y
        return loss
    
    def compute_cost(self, y_hat, y, method, theta, lambda_=0.5):
        """
        Return MSE cost for linear regression with ridge regularization.
        Or return log cost for logistic regression with ridge regularization.
        """
        m = len(y)
        reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        if method == "linear":
            residuals = y_hat - y
            mse = np.sum(residuals ** 2) / (2 * m)
            cost = mse + reg
            return cost
        elif method == "logistic":
            log = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            cost = log + reg
            return cost
    
    def predict(self, x, theta, method = "linear"):
        """
        Predict Ys using parameters (x matrix) and theta (coefficients)
        """
        y_hat = x @ theta
        if method == "logistic":
            # IMPORTANT: These are probabilities, NOT the class labels itself.
            y_hat = self.sigmoid(y_hat)
        return y_hat
    
    def odds_to_class(self, y_hat, threshold = 0.5):
        """
        Converts logistic regression odds to actual binary classifiers
        """
        return np.array([1.0 if h >= threshold else 0.0 for h in y_hat])
    
    def compute_gradient(self, x, y, y_hat, theta, lambda_=0.5):
        """
        Compute gradient for linear or logistic regression with optional ridge regularization.
        """
        m = len(y)
        loss = y_hat - y  # shape (m, )

        # Base gradient
        grad = (x.T @ loss) / m

        # Ridge regularization (exclude bias)
        grad[1:] += (lambda_ / m) * theta[1:]
        return grad
    
    
    def fit(self, x, y, theta, alpha=0.01, num_iters=500, method = "linear", regular="ridge", lambda_ = 0.5):
        """
        Fitting model with or without regularization.
        L1 = Lasso (not implemented).
        L2 = Ridge (default).

        Important: Be conscious over whether the input x and theta should be scaled.
        """
        cost_history = np.zeros(num_iters)

        if regular == "ridge":
            for i in range(num_iters):
                y_hat = self.predict(x, theta, method)
                grad = self.compute_gradient(x, y, y_hat, theta, lambda_)
                theta = theta - alpha * grad
                new_y_hat = self.predict(x, theta, method)
                cost_history[i] = self.compute_cost(new_y_hat, y, method, theta, lambda_)
        return theta, cost_history
    

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
    
    # Logistic Specific
    def sigmoid(self, z):
        """
        Return sigmoid of Z
        """
        sig = 1 / (1 + np.exp(-z))
        return sig

# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
