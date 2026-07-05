import numpy as np


# 70% training           30% test
def train_test_split(X, y, test_size=0.3, seed=42):
    
    np.random.seed(seed)
    N = X.shape[1]
    
    shuffled_indices = np.random.permutation(N)
    
    test_set_size = int(N * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    X_train, X_test = X[:, train_indices], X[:, test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_true, y_pred):

    accuracy = np.mean(y_true == y_pred) * 100
    
    #Confusion matrix construction
    tp = np.sum((y_true == 1) & (y_pred == 1))        #true positive
    tn = np.sum((y_true == 0) & (y_pred == 0))        #true negative
    fp = np.sum((y_true == 0) & (y_pred == 1))        #false positive
    fn = np.sum((y_true == 1) & (y_pred == 0))        #false negative
    
    print(f"--> Accuracy: {accuracy:.2f}%")
    print("--> Confusion Matrix:")
    print(f"      Pred:0   Pred:1")
    print(f"True:0  [{tn:<5}] [{fp:<5}]  (Spoofed)")
    print(f"True:1  [{fn:<5}] [{tp:<5}]  (Authentic)")
    return accuracy


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :6].T  
    y = data[:, 6]     
    return X, y


def minDCF(llr_scores, labels, prior_class, cost_FN, cost_FP):
   

    sorted_indices= np.argsort(llr_scores)
    llr_scores= llr_scores[sorted_indices]
    labels= labels[sorted_indices]

    total_authentic = np.sum(labels == 1)
    total_spoofed = np.sum(labels == 0)

    p_fn_cum= np.cumsum(labels==1)
    p_fn_cum= np.concatenate([[0], p_fn_cum, [total_authentic]])

    p_fp_cum = total_spoofed - np.cumsum(labels == 0)
    p_fp_cum = np.concatenate([[total_spoofed], p_fp_cum, [0]])

    p_fn_rates = p_fn_cum / total_authentic
    p_fp_rates = p_fp_cum / total_spoofed

    dcf_values = cost_FN * p_fn_rates * prior_class + cost_FP * p_fp_rates * (1 - prior_class)

    dummy_dcf = min(cost_FN * prior_class, cost_FP * (1 - prior_class))

    min_dcf = np.min(dcf_values) / dummy_dcf
    return min_dcf

