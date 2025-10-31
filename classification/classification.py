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
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy.typing as npt
import pandas as pd

# CLASSES
class NaiveBayes:
    """Guassian NaiveBayes class"""

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
    
class ID3:
    """ID3 Decision tree class"""
    def __init__(self) -> None:
        pass

    def entropy(self, targets: npt.NDArray):
        """
        Calculate entropy for the target array, where target consists of class labels.
        """
        m = len(targets)
        values, counts = np.unique(targets, return_counts=True)
        probs = counts / m
        entr = -np.sum(probs * np.log2(probs))
        return entr
    

    def information_gain_single(self, feature: npt.NDArray, targets: npt.NDArray):
        """
        Returns information gain for certain feature split
        """
        original_entropy = self.entropy(targets)
        feature_array = np.array(feature)
        target_array = np.array(targets)
        m = len(target_array)
        indices = [(feature_array == val).nonzero()[0] for val in np.unique(feature_array)]
        entropies = [self.entropy(target_array[idx]) for idx in indices]
        dv_over_d = np.array([len(idx) for idx in indices]) / m
        weighted = np.sum(dv_over_d * entropies)
        return original_entropy - weighted

    def best_split_numeric(self, data: pd.DataFrame, feature: str, target: str):
        """
        Finds highest possible information gain splitting point for numeric feature
        """
        values = sorted(data[feature].unique())
        best_gain = -1
        best_threshold = None
        base_entropy = self.entropy(np.array(data[target]))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            left = data[data[feature] < threshold]
            right = data[data[feature] >= threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            # weighted entropy
            weighted_entropy = (len(left) / len(data)) * self.entropy(left[target]) + \
                            (len(right) / len(data)) * self.entropy(right[target])
            gain = base_entropy - weighted_entropy

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_gain, best_threshold

    def information_gain(self, df: pd.DataFrame, features: list[str], target: str):
        """
        Calculate information gain for a target given a list of feature strings
        """
        original_entropy = self.entropy(np.array(df[target]))
        gains = {}

        for feature in features:
            # Get all entropies for all features
            entropies = df.groupby(feature)[target].apply(self.entropy)

            # Get weights
            weights = df[feature].value_counts(normalize=True)

            # Align against mismatching
            weights, entropies = weights.align(entropies, join='inner')

            # Get weighted avg
            weighted_entropy = np.sum(weights * entropies)

            # Information gain
            gains[feature] = original_entropy - weighted_entropy
        return gains

    def train_id3(self, data: pd.DataFrame, features: list[str], target: str, feature_dict=None, depth=0, max_depth=4):
        """
        Train id3 given base list of features you want to train on
        """
        if feature_dict is None:
            feature_dict = {}

        # base case
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]
        if len(features) == 0 or depth >= max_depth:
            return data[target].mode()[0]

        gains = {}
        thresholds = {}

        # Get information gain for each feature
        for f in features:
            # Check if the feature is numeric
            if all(isinstance(x, (int, float)) for x in data[f] if x is not None):
                gains[f], thresholds[f] = self.best_split_numeric(data, f, target)
            else:
                gain_val = self.information_gain(data, [f], target)
                if isinstance(gain_val, dict):
                    gain_val = list(gain_val.values())[0]
                gains[f] = gain_val
                thresholds[f] = None

        max_gain_feature = max(gains, key=gains.get)
        max_threshold = thresholds[max_gain_feature]

        # New node
        feature_dict[max_gain_feature] = {}

        if max_threshold is not None:
            # If it's a numeric feature
            left_data = data[data[max_gain_feature] < max_threshold]
            right_data = data[data[max_gain_feature] >= max_threshold]

            if left_data.empty or right_data.empty:
                # No split, go to base
                return data[target].mode()[0]

            feature_dict[max_gain_feature][f"< {max_threshold}"] = self.train_id3(
                left_data,
                [f for f in features if f != max_gain_feature],
                target,
                depth=depth + 1,
                max_depth=max_depth
            )

            feature_dict[max_gain_feature][f">= {max_threshold}"] = self.train_id3(
                right_data,
                [f for f in features if f != max_gain_feature],
                target,
                depth=depth + 1,
                max_depth=max_depth
            )

        else:
            # Catagorical
            for val in data[max_gain_feature].unique():
                subset = data[data[max_gain_feature] == val]
                if subset.empty:
                    feature_dict[max_gain_feature][val] = data[target].mode()[0]
                else:
                    feature_dict[max_gain_feature][val] = self.train_id3(
                        subset,
                        [f for f in features if f != max_gain_feature],
                        target,
                        depth=depth + 1,
                        max_depth=max_depth
                    )

        return feature_dict


# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
