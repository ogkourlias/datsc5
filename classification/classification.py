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
from sklearn.base import ClassifierMixin, BaseEstimator

# CLASSES
class NaiveBayes(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes classifier"""

    def __init__(self):
        pass

    def get_params(self, deep=True):
        """
        """
        return {
        }
    
    # def set_params(self, **params):
    #     """
    #     Set the parameters of this estimator. Returns self.
    #     """
    #     for key, value in params.items():
    #         if not hasattr(self, key):
    #             # still set it to allow new params if user provides them
    #             setattr(self, key, value)
    #         else:
    #             setattr(self, key, value)
    #     return self

    def fit(self, X, y):
        """Fit the Gaussian Naive Bayes model"""
        X = np.array(X)
        y = np.array(y)
        self.labels_ = np.unique(y)
        xs_by_lab = {lab: X[y == lab] for lab in self.labels_}
        self.mu_ = np.array([np.mean(xs_by_lab[label], axis=0) for label in self.labels_])
        self.sd_ = np.array([np.std(xs_by_lab[label], axis=0) for label in self.labels_])
        priors_dict = Counter(y)
        m = len(y)
        self.priors_ = np.array([priors_dict[label] / m for label in self.labels_])
        return self

    def predict(self, X):
        """Predict class labels for samples in X"""
        X = np.array(X)
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            posteriors = []
            for j, label in enumerate(self.labels_):
                sq_diff = (x - self.mu_[j, :]) ** 2
                norm_fac = 1 / np.sqrt(2 * np.pi * self.sd_[j, :])
                likelihoods = norm_fac * np.exp(-sq_diff / (2 * self.sd_[j, :])**2)
                prod = np.prod(likelihoods) * self.priors_[j]
                posteriors.append(prod)
            y_pred[i] = self.labels_[np.argmax(posteriors)]
        return y_pred

    def score(self, X, y):
        """Return accuracy of the classifier."""
        return np.mean(self.predict(X) == y)

    
class ID3(ClassifierMixin, BaseEstimator):
    """ID3 Decision tree class"""
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.tree_ = None
        self.features_ = None # X series/array
        self.target_ = None # y series/array

    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

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
    
    def predict(self, sample: pd.Series, tree: dict):
        """Predict target for a SINGLE INSTANCE/ROW"""
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        branches = tree[feature]
        value = sample[feature]

        for cond, subtree in branches.items():
            if cond.startswith("<"):
                threshold = float(cond.split("<")[1])
                if value < threshold:
                    return self.predict(sample, subtree)
            elif cond.startswith(">="):
                threshold = float(cond.split(">=")[1])
                if value >= threshold:
                    return self.predict(sample, subtree)
            elif value == cond:
                return self.predict(sample, subtree)

        return None
    
    def predict_df(self, data: pd.DataFrame, tree: dict):
        data["pred"] = data.apply(self.predict, axis=1, tree=tree)
        return data
    
    def example_df_predict(self):
        data = pd.DataFrame({
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                    "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
        "Humidity": [85, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 91, 75, 91],
        "Windy": [False, True, False, False, False, True, True,
                False, False, False, True, True, False, True],
        "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    })
        train_df, test_df = train_test_split(data, test_size=0.3, random_state=42) #reserve 30% of the data for testing
        # train_df = x_train.insert(y_train)
        # test_df = x_test.insert(y_test)
        target = "Play"
        features = [col for col in data.columns if col != target]
        tree = self.train_id3(train_df, features, target)
        with_preds_df = self.predict_df(test_df, tree)
        print(with_preds_df)
    
    def fit(self, X, y):
        """Fit the decision tree classifier."""
        df = X.copy()
        df["target"] = y
        self.features_ = list(X.columns)
        self.target_ = "target"
        self.tree_ = self.train_id3(df, self.features_, self.target_, max_depth=self.max_depth)
        return self

    def predict(self, X):
        """Predict target values for given DataFrame X."""
        df = X.copy()
        df["pred"] = df.apply(self._predict_row, axis=1, tree=self.tree_)
        return df["pred"].values

    def _predict_row(self, sample, tree):
        """Recursive prediction for one sample."""
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        branches = tree[feature]
        value = sample[feature]
        for cond, subtree in branches.items():
            if cond.startswith("<"):
                threshold = float(cond.split("<")[1])
                if value < threshold:
                    return self._predict_row(sample, subtree)
            elif cond.startswith(">="):
                threshold = float(cond.split(">=")[1])
                if value >= threshold:
                    return self._predict_row(sample, subtree)
            elif value == cond:
                return self._predict_row(sample, subtree)
        return None

    def score(self, X, y):
        """Return accuracy."""
        return np.mean(self.predict(X) == y)

# MAIN
def main(args):
    """Main function"""
    # FINISH
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
