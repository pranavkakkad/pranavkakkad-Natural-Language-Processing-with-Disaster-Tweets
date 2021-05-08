from sklearn import model_selection
from tf_idf import return_vector
from sklearn.ensemble import RandomForestClassifier

def RandomForest_accuracy():
    rf = RandomForestClassifier()
    train_data, test_data, train_bow, test_bow, train_tfidf, test_tfidf = return_vector()
    rfscores_bow = model_selection.cross_val_score(rf, train_bow, train_data["target"], cv=3, scoring="f1")

    rfscores_tfidf = model_selection.cross_val_score(rf,train_bow,train_data["target"], cv=3, scoring="f1")

    print("RF BOW F1 scores: "+ str(rfscores_bow))
    print("RF BOW F1 scores mean: "+ str(rfscores_bow.mean()))

    print("RF TFIDF scores: " + str(rfscores_tfidf))
    print("RF TFIDF F1 scores mean: "+ str(rfscores_tfidf.mean()))

RandomForest_accuracy()