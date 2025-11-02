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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

# CLASSES
class VotingClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard'):
        """
        asdf
        """
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            cloned = clone(est)
            cloned.fit(X, y)
            self.fitted_estimators_.append((name, cloned))
        return self

    def predict(self, X):
        if self.voting == 'soft':
            avg_proba = np.mean(
                [est.predict_proba(X) for _, est in self.fitted_estimators_],
                axis=0
            )
            return np.argmax(avg_proba, axis=1)
        else:
            predictions = np.asarray([
                est.predict(X) for _, est in self.fitted_estimators_
            ])
            # majority vote along axis=0
            maj_vote = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
            )
            return maj_vote

    def predict_proba(self, X):
        if self.voting != 'soft':
            raise AttributeError("predict_proba is only available for soft voting.")
        return np.mean([est.predict_proba(X) for _, est in self.fitted_estimators_], axis=0)
