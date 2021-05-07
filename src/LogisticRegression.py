from tf_idf import return_vector
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

def LogisticRegression_accracy():
    train_data,test_data,train_bow, test_bow, train_tfidf, test_tfidf = return_vector()
    lr = LogisticRegression(C=1.0)
    lrscores_bow = model_selection.cross_val_score(lr, train_bow, train_data["target"],
                                                   cv=3, scoring="f1")
    lrscores_tfidf = model_selection.cross_val_score(lr, train_tfidf,
                                                     train_data["target"],cv=3,scoring="f1")

    print("LR bow F1 scores: " + str(lrscores_bow))
    print("LR bow F1 mean score: " + str(lrscores_bow.mean()))

    print("LR TFIDF F1 scores: "+ str(lrscores_tfidf))
    print("Lr TRIDF F1 mean score: "+ str(lrscores_tfidf.mean()))

LogisticRegression_accracy()


