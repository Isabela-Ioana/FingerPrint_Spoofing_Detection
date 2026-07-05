from pca import PCA
from lda import LDA
from mvg import MultivariateGaussianClassifier
from helping_functions import load_data, train_test_split, evaluate_predictions, minDCF



if __name__=='__main__':
    pca_model= PCA(n_components=2)
    lda_model= LDA(n_components=1)

    try:
        X, y= load_data('dataset.csv')
    
    except FileNotFoundError:
        print("Could not find file!")
    
    print(f"Dimensions: {X.shape[0]} features x {X.shape[1]} samples.\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
    

    prior_class = 0.5
    cost_FN = 1.0
    cost_FP = 1.0

    ######### MVG on un-processed data #########

    mvg_classifier= MultivariateGaussianClassifier()
    mvg_classifier.fit(X_train,y_train)

    predictions_raw, llr_raw = mvg_classifier.predict(X_test)
    min_dcf_raw = minDCF(llr_raw, y_test, prior_class, cost_FN, cost_FP)

    print("RAW MVG:")
    evaluate_predictions(y_test,predictions_raw)
    print(f"Raw MVG minDCF: {min_dcf_raw:.4f}\n")


    ######### UNSUPERVISED:  MVG on pre-processed data with PCA #########

    X_train_pca = pca_model.fit_transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    mvg_pca = MultivariateGaussianClassifier()
    mvg_pca.fit(X_train_pca, y_train)
    predictions_pca, llr_pca = mvg_pca.predict(X_test_pca)
    min_dcf_pca = minDCF(llr_pca, y_test, prior_class, cost_FN, cost_FP)


    print("PCA MVG:")
    evaluate_predictions(y_test, predictions_pca)
    print(f"PCA+MVG minDCF: {min_dcf_pca:.4f}\n")
    
    ######### SUPERVISED:  MVG on pre-processed data with LDA #########
    lda_model.fit(X_train, y_train)
    X_train_lda = lda_model.transform(X_train)
    X_test_lda = lda_model.transform(X_test)

    mvg_lda = MultivariateGaussianClassifier()
    mvg_lda.fit(X_train_lda, y_train)
    predictions_lda, llr_lda = mvg_lda.predict(X_test_lda)
    min_dcf_lda = minDCF(llr_lda, y_test, prior_class, cost_FN, cost_FP)


    print("LDA MVG:")
    evaluate_predictions(y_test, predictions_lda)
    print(f"LDA+MVG minDCF: {min_dcf_lda:.4f}\n")    


   

