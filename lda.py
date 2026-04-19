import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl


def calculate_SB_SW(D,L):
    global_mean= D.mean(1).reshape(-1,1)
    #print("Global mean: \n ",global_mean)
    N= D.shape[1]
    #print("\n Number of elements: \n ", N)
    SB=np.zeros((D.shape[0],D.shape[0]))
    SW=np.zeros((D.shape[0],D.shape[0]))


    for label in np.unique(L):
        D_c= D[:,L==label]            #D_c = just elements of a class
        n_c= D_c.shape[1]             # n_c= number of elements of a class
        mean_c= D_c.mean(1).reshape(-1,1)   #mean_c = mean of a class

        distance= mean_c - global_mean    
        SB= SB + n_c * (distance * distance.T)

        centered_data_c = D_c - mean_c
        SW+= (centered_data_c @ centered_data_c.T)

    SB= SB/ N
    SW= SW/ N
    return SB,SW

def lda_directions(SB,SW):
    s,U= scl.eigh(SB,SW)
    W= U[:,::-1][:,0:1]

    W=-W
    
    return W


def plot_lda_projection(D_proj, L):
   
    plt.figure(figsize=(7, 5))
    
    # Separate the 1D projected points by class
    Class0_pts = D_proj[0, L == 0]
    Class1_pts = D_proj[0, L == 1]
    
    # Create the histograms (using density=True to match your image)
    plt.hist(Class0_pts, bins=5, density=True, alpha=0.4, 
             color='sandybrown', label='Spoofed', edgecolor='darkorange')
    
    plt.hist(Class1_pts, bins=5, density=True, alpha=0.4, 
             color='forestgreen', label='Authentic', edgecolor='seagreen')

    plt.legend()
    plt.xlabel(r"Fingerprint spoofing", fontsize=12, labelpad=15)
    plt.tight_layout()
    plt.show()




def plot_lda_comparison(DP_train, LTR, DP_val, LVAL):
    # Creăm o singură figură cu 2 sub-grafice (1 rând, 2 coloane)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot pentru Training ---
    y_tr = DP_train.flatten()
    ax1.hist(y_tr[LTR == 0], bins=5, density=True, alpha=0.4, 
             color='sandybrown', label='Spoofed', edgecolor='sandybrown')
    ax1.hist(y_tr[LTR == 1], bins=5, density=True, alpha=0.4, 
             color='green', label='Authentic', edgecolor='green')
    ax1.set_title("(a) Model training set (DTR, LTR)")
    ax1.legend()

    # --- Plot pentru Validation ---
    y_val = DP_val.flatten()
    ax2.hist(y_val[LVAL == 0], bins=5, density=True, alpha=0.4, 
             color='sandybrown', label='Spoofed', edgecolor='sandybrown')
    ax2.hist(y_val[LVAL == 1], bins=5, density=True, alpha=0.4, 
             color='green', label='Authentic', edgecolor='green')
    ax2.set_title("(b) Model validation set (DVAL, LVAL)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

