import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.05, max_iterations=1000, random_state=5058, tol=1e-5):
        self.eta = eta
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.tol = tol

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        n_samples, n_features = X.shape
        rgen = np.random.default_rng(self.random_state)

        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        prev_cost = np.inf

        for i in range(self.max_iterations):
            net_input = np.dot(X, self.weights_) + self.bias_
            y_predicted = self._sigmoid(net_input)

            errors = (y_predicted - y)

            self.weights_ -= self.eta * (1 / n_samples) * X.T.dot(errors)
            self.bias_ -= self.eta * (1 / n_samples) * np.sum(errors)

            y_clipped = np.clip(y_predicted, 1e-15, 1 - 1e-15)
            cost = -np.mean(y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped))

            if abs(prev_cost - cost) < self.tol: break
            prev_cost = cost

        return self

    @staticmethod
    def _sigmoid(z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        net_input = np.dot(X, self.weights_) + self.bias_
        proba_1 = self._sigmoid(net_input)
        
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.where(probas[:, 1] >= 0.5, self.classes_[1], self.classes_[0])
