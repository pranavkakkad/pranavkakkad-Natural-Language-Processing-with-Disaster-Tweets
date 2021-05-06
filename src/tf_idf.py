import numpy as np
import pandas as pd

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

part1 = r'@[A-Za-z0-9./]+'
part2 = r'http?://[A-Za-z0-9./]+'
combined_part = r'|'.join((part1, part2))

nltk.download('wordnet')
nltk.download('stopwords')

def clean_data(text):

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
# print(clean_data("@PRanav http://www.google.ocom rocks is good"))

def word_freq(text):
    #Return list with words along with the frequency
    pass