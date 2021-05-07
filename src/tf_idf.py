import numpy as np
import pandas as pd

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import spacy

nlp = spacy.load('en_core_web_lg')


tok = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

part1 = r'@[A-Za-z0-9./]+'
part2 = r'http?://[A-Za-z0-9./]+'
combined_part = r'|'.join((part1, part2))

# nltk.download('wordnet')
# nltk.download('stopwords')

def clean_data(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_part, "", souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    letters_only = re.sub("[^a-zA-Z]"," ",clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    lem_words =[]
    stop_words = set(stopwords.words('english'))
    # print(type(words))
    for i in words:
        if i not in stop_words:
            lem_words.append(lemmatizer.lemmatize(i))
    return (" ".join(lem_words)).strip()


#Test code
# print(clean_data("@PRanav http://www.google.ocom rocks is good,"))
#
def word_freq(text):
    #Return list with words along with the frequency
    text = text.split()
    unique_text = set(text)
    word_freq = []
    for words in unique_text:
        word_freq.append([words, text.count(words)])

    return word_freq

# print(word_freq("hi my name is pranav kakkad hi"))
def return_vector():
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")

    train_data["clean_text"] = train_data["text"].apply(clean_data)
    test_data["clean_text"] = test_data["text"].apply(clean_data)

    vectorizer = CountVectorizer()
    train_bow = vectorizer.fit_transform(train_data["clean_text"])
    test_bow = vectorizer.fit_transform(test_data["clean_text"])

    tfidf = TfidfVectorizer(min_df=2,max_df=0.5,ngram_range=(1,2))
    train_tfidf = tfidf.fit_transform(train_data["clean_text"])
    test_tfidf = tfidf.transform(test_data["clean_text"])

    return (train_data,test_data,train_bow, test_bow, train_tfidf, test_tfidf)