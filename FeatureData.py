import csv
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction


class FeatureData(object):
    def __init__(self, article_file_path, stances_file_path):
        self.number_of_classes = 4
        self.classes = ["Agrees", "Disagrees", "Discusses", "Unrelated"]
        self.articles = self._get_articles(article_file_path)  # list of dictionaries
        self.stances = self._get_stances(stances_file_path)
        self.number_of_stances = len(self.stances)
        self.number_of_articles = len(self.articles)

    def get_clean_articles(self):
        """Retrieves a list of dictionaries containing the fully cleaned articleBody and the Body ID of each article."""
        clean_articles = []

        for item in self.articles:
            cleaned_article = clean(item['articleBody'])
            tokens = tokenize_text(cleaned_article)
            no_stop_word_tokens = remove_stopwords(tokens)
            lemmatized_tokens = get_tokenized_lemmas(no_stop_word_tokens)
            clean_articles.append(' '.join(lemmatized_tokens))

        return clean_articles

    def get_clean_stances(self):
        """Retrieves a list of dictionaries containing the fully cleaned Headlines and the Body ID and Stance for
        each headline."""

        pass

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
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(w).lower()


def clean(text):
    return " ".join(re.findall(r'\w+', text.decode('utf-8'), flags=re.UNICODE)).lower()


def tokenize_text(text):
    return [token for token in word_tokenize(text)]


def remove_stopwords(list_of_tokens):
    return [word for word in list_of_tokens if word not in feature_extraction.text.ENGLISH_STOP_WORDS]


def get_tokenized_lemmas(tokens):
    return [normalize_word(token) for token in tokens]


if __name__ == '__main__':
    fd = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    clean_articles = fd.get_clean_articles()
    print clean_articles[0]
