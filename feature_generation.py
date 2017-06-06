"""Methods for generating each feature can be added to the FeatureGenerator class."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from FeatureData import FeatureData


class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, clean_articles, clean_stances):
        self.articles = clean_articles
        self.stances = clean_stances

    def get_features(self):
        """Retrieves the full set of features as a matrix (the X matrix for training.)"""

    def _get_ngrams(self, max_ngram_size):
        """Retrieves counts for ngrams of the article title in the article itself, from one up to size max_ngram_size.
        Returns a list containing the ngram counts."""

        for article, stance in zip(self.articles, self.stances):
            # Retrieves the vocabulary of ngrams for the title.
            stance_vectorizer = CountVectorizer(input=stance['Headline'], ngram_range=(1, max_ngram_size))
            stance_vectorizer.fit_transform([stance['Headline']]).toarray()

            # Search the article text and count headline ngrams.
            vocab = stance_vectorizer.get_feature_names()
            vectorizer = CountVectorizer(input=article['articleBody'], vocabulary=vocab, ngram_range=(1, max_ngram_size))
            ngram_counts = vectorizer.fit_transform([article['articleBody']]).toarray()
            features = vectorizer.get_feature_names()

            if np.count_nonzero(ngram_counts[0]) > 1:
                for x in np.nditer(np.nonzero(ngram_counts[0])):
                    print features[x]

                print 'Headline:'
                print stance['Headline']
                print 'Article:'
                print article['articleBody']
                print features
                print ngram_counts


if __name__ == '__main__':
    feature_data = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    feature_generator = FeatureGenerator(feature_data.get_clean_articles(), feature_data.get_clean_stances())
    feature_generator._get_ngrams(3)

