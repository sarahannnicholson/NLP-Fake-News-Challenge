import csv
import re
import os
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

class FeatureData():
    def __init__(self, article_file_path, stances_file_path):
        self.number_of_classes = 4
        self.classes = ["Agrees", "Disagrees", "Discusses", "Unrelated"]
        self.articles = self._get_articles(article_file_path) # list of dictionaries
        self.stances = self._get_stances(stances_file_path)
        self.number_of_stances = len(self.stances)
        self.number_of_articles = len(self.articles)

    def _get_articles(self, path):
        # Body ID, articleBody
        articles = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                articles.append(row)
        return articles

    def _get_stances(self, path):
        # Headline, Body ID, Stance
        stances = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                stances.append(row)
        return stances

def normalize_word(w):
    wordnet_lemmatizer = WordNetLemmatizer()
    return wordnet_lemmatizer(w).lower()

def clean(str):
    return " ".join(re.findall(r'w+', string, flags=re.UNICODE).lower())

def get_tokenized_lemmas(str):
    return [normalize_word(token) for token in word_tokenize(str)]

def remove_stopwords(list_of_tokens):
    return [word for word in list_of_tokens if word not in feature_extraction.text.ENGLISH_STOP_WORDS]

if __name__ == '__main__':
    fd = FeatureData('./train_bodies.csv', './train_stances.csv')

