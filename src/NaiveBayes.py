from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from tf_idf import return_vector

def NaiveBayes_accuracy():
    train_data, test_data, train_bow, test_bow, train_tfidf, test_tfidf = return_vector()
    nb = MultinomialNB()
    nbscore_bow = model_selection.cross_val_score(nb, train_bow, train_data["target"], cv=3, scoring="f1")

    nbscore_tfidf = model_selection.cross_val_score(nb, train_tfidf, train_data["target"], cv=3, scoring="f1")

    print("NB BOW F1 score: "+ str(nbscore_bow))
    print("NB BOW F1 score mean: "+ str(nbscore_bow.mean()))

    print("NB TFIDF F1 scores: " + str(nbscore_tfidf))
    print("NB TFIDF F1 scores mean: "+ str(nbscore_tfidf.mean()))


NaiveBayes_accuracy()