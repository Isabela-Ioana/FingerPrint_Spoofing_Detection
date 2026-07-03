import numpy as np
import scipy.linalg as scl

class PCA:

    def __init__ (self,n_components=2):
        self.n_components= n_components
        self.mu= None
        self.P= None


    def fit(self,X):           #train
        self.mean= np.mean(X,axis=1, keepdims=True)      #axis=1 -> calc. means orizontally
        centered_X= X- self.mean

        C= (centered_X @ centered_X.T)/ float(X.shape[1])   #cov matrix
        U,_,_= np.linalg.svd(C)               #now U contains the eigenvectors automatically sorted 
        self.P= U[:,:self.n_components]
        return self
    
    def transform(self,X):
        center_data= X-self.mean
        return self.P.T @ center_data       #the projection of data
    
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)

