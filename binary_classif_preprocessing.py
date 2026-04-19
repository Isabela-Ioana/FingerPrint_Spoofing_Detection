import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
from pca import PCA
from lda import calculate_SB_SW,lda_directions,plot_lda_projection,plot_lda_comparison
from binary_classification import splitdata,load




if __name__ == '__main__':
    D,L= load("dataset.csv")

    #first we split in  training and valid
    (DTR, LTR), (DVAL, LVAL)= splitdata(D,L)

    #on training data we apply pca
    DP_training_PCA, P =PCA(DTR,LTR)
    DP_validation_PCA = np.dot(P.T, DVAL)

    print("\n DP_training_PCA \n",DP_training_PCA)



    #now, we apply LDA on DP_training_PCA instead of brute data
    SB,SW = calculate_SB_SW(DP_training_PCA, LTR)
    w_orth= lda_directions(SB,SW)
    DP_lda= np.dot(w_orth.T,DP_training_PCA)
    DP_validation_lda= np.dot(w_orth.T,DP_validation_PCA)
    plot_lda_comparison(DP_lda, LTR, DP_validation_lda, LVAL)

    threshold= (DP_lda[0, LTR==0].mean() + DP_lda[0, LTR==1].mean()) / 2.0 
    print("------------------ threshold ------------------ :\n", threshold)



    

