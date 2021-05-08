from sklearn import model_selection
from tf_idf import return_vector
from sklearn.svm import LinearSVC

def SVM_accuracy():
    train_data, test_data, train_bow, test_bow, train_tfidf, test_tfidf = return_vector()
    svm = LinearSVC(random_state=1, dual=False, max_iter=10000)

    svmscores_bow = model_selection.cross_val_score(svm, train_bow, train_data["target"],cv=3, scoring="f1")

    svmscores_tfidf = model_selection.cross_val_score(svm, train_tfidf, train_data["target"], cv=3, scoring="f1")

    print("SVM BOW F1 scores:" + str(svmscores_bow))
    print("SVM BOW F1 scores mean: "+ str(svmscores_bow.mean()))

    print("SVM TFIDF scores: "+ str(svmscores_tfidf))
    print("SVM TFIDF scores mean: "+ str(svmscores_tfidf.mean()))


SVM_accuracy()