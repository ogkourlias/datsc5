#!/usr/bin/env python3

"""
    usage:
        python3 orfeas_gkourlias_boink.py
"""

# METADATA VARIABLES
__author__ = "Orfeas Gkourlias"
__status__ = "WIP"
__version__ = "0.2"

# IMPORTS
import sys
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

# CLASS
class DataProcessor(BaseEstimator, TransformerMixin):
    """Transformer"""
    
    def __init__(self, variance_threshold=0.0, minority_threshold=0.1,
                  nan_threshold=0.1, use_smote=True, target=None, 
                  pca=True, scale=True, impute=True):
        
        self.variance_threshold = variance_threshold
        self.minority_threshold = minority_threshold
        self.nan_threshold = nan_threshold
        self.use_smote = use_smote
        self.target = target
        self.pca = pca
        self.scale = scale
        self.impute = impute
        self.removed_columns_ = []  # underscore means "learned attribute"
    
    def fit(self, x, y=None):
        """Store constant columns."""
        x = x.copy()
        self.removed_columns_ = [col for col in x.columns if x[col].nunique() == 1]
        return self
    
    def transform(self, x, y=None):
        """Apply transformations."""
        x = x.copy()
        
        # Drop constant columns
        if self.removed_columns_:
            print(f"Dropping constant columns: {self.removed_columns_}")
            x = x.drop(columns=self.removed_columns_)
        
        self.numeric_cols = x.select_dtypes(include=["number","bool"]).columns.tolist()

        # Missing value check
        nan_prop = x.isna().sum() / x.shape[0]
        nan_fail = nan_prop[nan_prop > self.nan_threshold]
        if self.impute:
            if not nan_fail.empty: # If cols fail nan threshold then impute them
                fail_cols = nan_fail.index.to_list()
                fail_str_col = [col for col in fail_cols if col not in self.numeric_cols]
                fail_num_cols = [col for col in fail_cols if col in self.numeric_cols]

                if fail_num_cols:
                    x[fail_num_cols] = x[fail_num_cols].fillna(x[fail_num_cols].mean())

                # Fill in with most occuring catagorical value. 
                # BE VERY CAREFUL WITH THIS
                # It may introduce imbalances
                # Could use classification algorithm. Turn off impute flag and do it manually.
                if fail_str_col:
                    for col in fail_str_col:
                        x[col] = x[col].fillna(x[col].mode()[0])

            # Other cols have low missingness and we can drop the rows for those
            x = x.dropna()

        if self.scale:
            scaler = StandardScaler()
            x[self.numeric_cols] = scaler.fit_transform(x[self.numeric_cols])

        if self.pca:
            #TODO: Make this hahaxdlol
            ...
        
        # If target y is given and SMOTE is desired
        if y is not None and self.use_smote:
            class_dist = pd.Series(y).value_counts(normalize=True)
            low_rep = class_dist[class_dist < self.minority_threshold]
            
            if not low_rep.empty:
                print(f"Low representation for classes: {list(low_rep.index)}")
                sm = SMOTE(random_state=42)
                x_res, y_res = sm.fit_resample(x, y)
                return x_res, y_res
        
        return x, y


# MAIN
def main(args):
    """ Main function """
    # FINISH
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
