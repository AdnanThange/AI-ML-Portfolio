import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

class SelectiveDataClarityMode:
    def __init__(self, task_type='classification', confidence_threshold=0.5):
        self.task_type = task_type
        self.confidence_threshold = confidence_threshold
        self.selected_features = []
        self.removed_outliers_idx = []

    def _encode_categorical(self, X):
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
        return X_encoded

    def _feature_importance(self, X, y):
        if self.task_type == 'classification':
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()
        model.fit(X, y)
        return model.feature_importances_

    def _remove_low_variance(self, X, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return X[X.columns[selector.get_support(indices=True)]]

    def _correlation_score(self, X, y):
        scores = []
        for col in X.columns:
            try:
                corr, _ = pearsonr(X[col], y)
                scores.append(abs(corr))
            except:
                scores.append(0)
        return np.array(scores)

    def _mutual_info(self, X, y):
        scores = []
        for col in X.columns:
            try:
                mi = mutual_info_score(X[col], y)
                scores.append(mi)
            except:
                scores.append(0)
        return np.array(scores)

    def _remove_outliers(self, X, y):
        Xy = pd.concat([X, y], axis=1)
        z_scores = ((Xy - Xy.mean()) / Xy.std()).abs()
        mask = (z_scores < 3).all(axis=1)
        self.removed_outliers_idx = X.index[~mask]
        return X[mask], y[mask]

    def fit_transform(self, X, y):
        X = self._encode_categorical(X)
        y = pd.Series(y)

        # Step 1: Remove low-variance features
        X = self._remove_low_variance(X)

        # Step 2: Outlier removal
        X, y = self._remove_outliers(X, y)

        # Step 3: Feature scoring
        if self.task_type == 'classification':
            scores = self._mutual_info(X, y)
        else:
            scores = self._correlation_score(X, y)

        # Step 4: Select features above confidence threshold
        max_score = np.max(scores)
        threshold_value = self.confidence_threshold * max_score
        self.selected_features = X.columns[scores >= threshold_value]

        return X[self.selected_features], y

    def get_removed_outliers(self):
        return self.removed_outliers_idx

    def get_selected_features(self):
        return self.selected_features


