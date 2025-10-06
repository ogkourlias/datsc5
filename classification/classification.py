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
from sklearn.model_selection import train_test_split
from collections import Counter


# CLASSES
class Classifier:
    """Classification class"""

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
        return x, y
    
    def train_test_split(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) #reserve 30% of the data for testing
        return x_train, x_test, y_train, y_test

    def get_likelihoods(self, x_train, y_train):
        labels = np.unique(y_train)
        xs_by_lab = {lab: x_train[y_train == lab] for lab in labels}
        mu = np.array([np.mean(xs_by_lab[label], axis=0) for label in labels])
        sd = np.array([np.std(xs_by_lab[label], axis=0) for label in labels])
        return mu, sd

    def classify(self, x_test, labels, priors, mu, sd):
        y_pred = np.zeros(len(x_test))
        for i, x in enumerate(x_test):
            posteriors = []
            for j, label in enumerate(labels):
                sq_diff = (x - mu[j, :]) ** 2
                norm_fac = 1 / np.sqrt(2 * np.pi * sd[j, :])
                likelihoods = norm_fac * np.exp(-sq_diff / (2 * sd[j, :])**2)
                prod = np.prod(likelihoods) * priors[j]
                posteriors.append(prod)

            index_max = np.argmax(posteriors)
            y_pred[i] = labels[index_max]
        return y_pred



    def get_priors(self, y_train):
        labels = np.unique(y_train)
        m = len(y_train)
        priors_dict = Counter(y_train)
        priors = np.array([priors_dict[label] / m for label in labels])
        return labels, priors

# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
