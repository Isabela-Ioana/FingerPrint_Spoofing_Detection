import numpy as np

class MultivariateGaussianClassifier:

    def __init__(self):
        self.classes = None
        self.prioris = {}
        self.means = {}
        self.covariances = {}


    def fit(self, X, y):
        self.classes = np.unique(y)
        N_total = X.shape[1]

        for c in self.classes:
            X_c = X[:, y == c]
            N_c = X_c.shape[1]
            
            # P(Class) = a priori probability
            self.prioris[c] = N_c / N_total
            
            # Maximum Likelihood estimation
            self.means[c] = np.mean(X_c, axis=1, keepdims=True)
            centered = X_c - self.means[c]
            self.covariances[c] = (centered @ centered.T) / N_c
        return self


    def _logpdf_mvg(self, X, mu, C):
        M = X.shape[0]
        _, log_det = np.linalg.slogdet(C)
        inv_C = np.linalg.inv(C)
        
        const = - (M / 2) * np.log(2 * np.pi) - 0.5 * log_det
        dist = X - mu

        term = -0.5 * np.sum(dist * (inv_C @ dist), axis=0)
        return const + term

    def predict(self, X):
        posteriors = []
        for c in self.classes:
            # log P(C|X) proportional with log P(X|C) + log P(C)
            log_posterior_c = self._logpdf_mvg(X, self.means[c], self.covariances[c]) + np.log(self.prioris[c])
            posteriors.append(log_posterior_c)
            
        posteriors_matrix = np.vstack(posteriors)

        # Choosing the class that maximizes the a posteriori probability
        idx_max = np.argmax(posteriors_matrix, axis=0)

        llr_scores = posteriors_matrix[1] - posteriors_matrix[0]

        return self.classes[idx_max], llr_scores
    