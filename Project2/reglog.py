import numpy as np


class LogisticRegression:
    
    def __init__(self, eta=0.05, max_iterations=1000, random_state=5058, tol=0.1e-4):
        self.rgen = np.random.default_rng(random_state)
        self.tol = tol
        self.eta = eta
        self.max_iterations = max_iterations
        self.prev_cost = np.inf 
    
    def fit(self, X, y): 
        n_samples, _ = X.shape
        
        self.weights = self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = np.float64(0.0)
        
        for i in range(self.max_iterations):
            y_predicted = self.predict_proba(X)
            
            errors = (y_predicted - y)
            norm_gradient = 1 / n_samples
            
            self.weights -= self.eta * norm_gradient * X.T.dot(errors)
            self.bias -= self.eta * norm_gradient * np.sum(errors)

            cost = (-y.dot(np.log(y_predicted)) - ((1 - y).dot(np.log(1 - y_predicted))))
            if i > 0 and abs(self.prev_cost - cost) < self.tol:
                break
                
            self.prev_cost = cost

        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def sigmoid_activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def predict_proba(self, X):
        return self.sigmoid_activation(self.net_input(X))