# Lets create a custom class in order to add custom function adder in our pipeline

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
fixed_acidity_ix, volatile_acidity_ix, pH_ix, density_ix = 0, 1, 8, 7


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_ratio_density=True):
        self.add_ratio_density = add_ratio_density

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fixed_volatile = X[:, fixed_acidity_ix] / X[:, volatile_acidity_ix]
        if self.add_ratio_density:
            pH_density = X[:, pH_ix] / X[:, density_ix]
            return np.c_[X, fixed_volatile, pH_density]

        else:
            return np.c_[X, fixed_volatile]
