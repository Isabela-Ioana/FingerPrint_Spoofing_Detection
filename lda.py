import numpy as np
import scipy.linalg as scl
#m= nb of features
#n= nb of samples

class LDA:
    def __init__ (self, n_components=1):
        self.n_components= n_components
        self.W= None

    def fit(self,X,y):
        global_mean= X.mean(1).reshape(-1,1)
        N= X.shape[1]
        M= X.shape[0]

        SB= np.zeros((M,M))
        SW= np.zeros((M,M))

        for label in np.unique(y):
            X_c = X[:, y == label]
            n_c = X_c.shape[1]
            mean_c = np.mean(X_c, axis=1, keepdims=True)

            # Between-class covariance
            distance = mean_c - global_mean
            SB += n_c * (distance @ distance.T)
            
            # Within-class covariance
            centered_c = X_c - mean_c
            SW += (centered_c @ centered_c.T)

        SB /= N
        SW /= N

        _, U = scl.eigh(SB, SW)
        self.W = U[:, ::-1][:, :self.n_components]
        return self

    def transform(self,X):
        return self.W.T @ X
