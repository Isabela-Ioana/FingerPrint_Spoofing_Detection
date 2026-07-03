from pca import PCA
from lda import LDA
from mvg import MultivariateGaussianClassifier
from helping_functions import load_data, train_test_split, evaluate_predictions



if __name__=='__main__':
    pca_model= PCA(n_components=2)
    lda_model= LDA(n_components=1)

    try:
        X, y= load_data('dataset.csv')
    
    except FileNotFoundError:
        print("Could not find file!")
    
    print(f"Dimensions: {X.shape[0]} features x {X.shape[1]} samples.\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
    


    ######### MVG on un-processed data #########

    mvg_classifier= MultivariateGaussianClassifier()
    mvg_classifier.fit(X_train,y_train)

    predictions_raw= mvg_classifier.predict(X_test)

    print("RAW MVG:")
    evaluate_predictions(y_test,predictions_raw)


    ######### UNSUPERVISED:  MVG on pre-processed data with PCA #########

    X_train_pca = pca_model.fit_transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    mvg_pca = MultivariateGaussianClassifier()
    mvg_pca.fit(X_train_pca, y_train)
    predictions_pca = mvg_pca.predict(X_test_pca)

    print("PCA MVG:")
    evaluate_predictions(y_test, predictions_pca)
    
    ######### SUPERVISED:  MVG on pre-processed data with LDA #########
    lda_model.fit(X_train, y_train)
    X_train_lda = lda_model.transform(X_train)
    X_test_lda = lda_model.transform(X_test)

    mvg_lda = MultivariateGaussianClassifier()
    mvg_lda.fit(X_train_lda, y_train)
    predictions_lda= mvg_lda.predict(X_test_lda)

    print("LDA MVG:")
    evaluate_predictions(y_test, predictions_lda)
    
