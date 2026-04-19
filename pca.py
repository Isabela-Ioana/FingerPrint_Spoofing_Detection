
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl

def PCA(D,L,dimensions=2):
    #mean:
    dataset_mean= D.mean(1)
    dataset_mean= dataset_mean.reshape(dataset_mean.size,1)
    print("Dataset mean: \n ", dataset_mean)

    #centering the data:
    centered_data= D - dataset_mean
    #print(centered_data)

    #covariance matrix:
    C= (centered_data @ centered_data.T)/ float(D.shape[1])
    print("\n Covariance matrix: \n ", C)

    #we calculate eigenvalues and vectors
    U, s, Vh = np.linalg.svd(C)
    print("\n s: \n", s,"\n \n U: \n",U)
    P = U[:, 0:dimensions]
    print(f"\n P matrix for {dimensions} dimensions: \n ",P)



    DP = np.dot(P.T, D)


    #HISTOGRAMS
    plt.figure(figsize=(15, 10))
    labels = ['Spoofed', 'Authentic']
    
    for i in range(dimensions): 
        plt.subplot(2, 3, i + 1)
        
        
        for label_idx in [0, 1]:
            data_to_plot = DP[i, L == label_idx]
            plt.hist(data_to_plot, bins=30, alpha=0.5, label=labels[label_idx], edgecolor='k')
        
        plt.title(f"Principal Component {i+1}")
        plt.xlabel("Projected Value")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()


    #PCA
    plt.figure(figsize=(8, 6))
    
    # Define the labels and colors 
    labels = ['Authentic','Spoofed']
    colors = ['C0', 'C1'] 

    for label_idx in range(len(labels)):
        y_class = DP[:, L == label_idx]
        plt.scatter(y_class[0, :], y_class[1, :], label=labels[label_idx], edgecolors='k', alpha=0.8)

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.title("PCA: FINGERPRINT SPOOFING DETECTION (2D)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return DP, P
