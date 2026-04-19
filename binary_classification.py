import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
from pca import PCA
from lda import calculate_SB_SW,lda_directions,plot_lda_projection,plot_lda_comparison

def load(name_file):
    attributes_list=[]
    fake_or_not=[]

    with open(name_file) as f:
        data= f.read().splitlines()
        
    
    
    for row in data:
        row1= row.split(",")
        wanted_attr= row1[0:6]
        fake_or_not_nb= row1[6]

        attributes_list.append(wanted_attr)
        fake_or_not.append(fake_or_not_nb)


    values= np.array(attributes_list, dtype=float)
    D= values.T

    names= np.array(fake_or_not, dtype= float)
    L= np.array(names)

    return D,L

def splitdata(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) 
    np.random.seed(seed) 
    idx = np.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain] 
    idxTest = idx[nTrain:] 
    DTR = D[:, idxTrain] 
    DVAL = D[:, idxTest] 
    LTR = L[idxTrain] 
    LVAL = L[idxTest] 
    return (DTR, LTR), (DVAL, LVAL)



if __name__ == '__main__':
    D,L= load("dataset.csv")
    # print("\n Subset of data: \n",D[:5,:5],"\n")
    # print("Subset of labels: \n", L[:5], "\n")


    # #call function
    # DP = PCA(D,L)
    # print(DP)


    # SB,SW= calculate_SB_SW(D,L)
    # w_orthogonal= lda_directions(SB,SW)
   

    # print("\n W orthogonal \n",w_orthogonal)

    # DP= np.dot(w_orthogonal.T,D)
    # print(DP)
    # plot_lda_projection(DP,L)



    (DTR, LTR), (DVAL, LVAL)= splitdata(D,L)
    SB,SW= calculate_SB_SW(DTR,LTR)
    w_orth=  lda_directions(SB,SW)
    DP= np.dot(w_orth.T,DTR)
    DP_validation= np.dot(w_orth.T, DVAL)
    plot_lda_comparison(DP,LTR,DP_validation,LVAL)

    
    threshold = (DP[0, LTR==0].mean() + DP[0, LTR==1].mean()) / 2.0 
    print("------------------ threshold ------------------ :\n", threshold)

                                            # m0 = DP[0, LTR == 0].mean()
                                            # m1 = DP[0, LTR == 1].mean()
                                            # print("m0", m0)
                                            # print("m1", m1)          m1>m0 => all good

    Predicted_labels_VAL = np.zeros(LVAL.shape, dtype=np.int32)
    Predicted_labels_VAL[DP_validation[0] >= threshold] = 1 
    Predicted_labels_VAL[DP_validation[0] < threshold] = 0 
    errors = np.sum(Predicted_labels_VAL != LVAL)
    error_rate = (errors / LVAL.size) * 100
    print(f"Number of validation labels: {LVAL.size}")
    print(f"Number of erors: {errors}")
    print(f"Error rate: {error_rate:.2f}%")

   


    #                                              binary classification scheme
    
    #   First, we split the data in training and validation (2/3 and 1/3 in this case). After that, we train the LDA model on the training
    # data, and finally we obtain the w_orthogonal which in the eigen vector. To obtain the final projection of data on the new direction,
    # we multiply w_orthogonal with initial data from training subset. =>   projection of training data
    # Normally, now the model knows the training data and we want to see how good is the model on unseen data. To test that, we have to use
    # the eigen vector from training data on validation data. After multiplying them, we'll obtain the projections of validation data based
    # on the training vector. => projection of validation data using training_data_eig_vector
    # More, these projections take the label 0 or 1 based on the following rule: < threshold=0 and >threshold=1
    # Now, to see how good the prediction was, we need to compare the initial labels for validation data and the predicted ones.


