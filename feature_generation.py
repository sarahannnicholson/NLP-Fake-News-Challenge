"""Methods for generating each feature can be added to the FeatureGenerator class."""

import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tqdm

from FeatureData import FeatureData, tokenize_text


class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, clean_articles, clean_stances):
        self._articles = clean_articles  # dictionary {article ID: body}
        self._stances = clean_stances  # list of dictionaries
        self._max_ngram_size = 3
        self._refuting_words = ['fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt',
                                'doubts', 'bogus', 'debunk', 'pranks', 'retract']

    def get_features(self):
        """Retrieves the full set of features as a matrix (the X matrix for training.)"""
        feature_names = []
        features = []

        logging.debug('Retrieving headline ngrams...')
        ngrams = np.array(self._get_ngrams()).reshape(len(self._stances), self._max_ngram_size)
        features.append(ngrams)
        [feature_names.append('ngram_' + str(count)) for count in range(1, self._max_ngram_size + 1)]

        logging.debug('Retrieving refuting words...')
        refuting = np.array(self._get_refuting_words()).reshape((len(self._stances), len(self._refuting_words)))
        features.append(refuting)
        [feature_names.append(word + '_refuting') for word in self._refuting_words]

        logging.debug('Retrieving polarity...')
        polarity = np.array(self._polarity_feature()).reshape(len(self._stances), 2)
        features.append(polarity)
        feature_names.append('headline_polarity')
        feature_names.append('article_polarity')

        return {'feature_matrix': np.concatenate(features, axis=1), 'feature_names': feature_names}

    def _get_ngrams(self):
        """Retrieves counts for ngrams of the article title in the article itself, from one up to size max_ngram_size.
        Returns a list of lists, each containing the counts for a different size of ngram."""
        ngrams = [[] for _ in range(self._max_ngram_size)]

        for stance in tqdm.tqdm(self._stances):
            # Retrieves the vocabulary of ngrams for the headline.
            stance_vectorizer = CountVectorizer(input=stance['Headline'], ngram_range=(1, self._max_ngram_size))
            stance_vectorizer.fit_transform([stance['Headline']]).toarray()

            # Search the article text and count headline ngrams.
            vocab = stance_vectorizer.get_feature_names()
            vectorizer = CountVectorizer(input=self._articles[stance['Body ID']], vocabulary=vocab,
                                         ngram_range=(1, self._max_ngram_size))
            ngram_counts = vectorizer.fit_transform([self._articles[stance['Body ID']]]).toarray()
            features = vectorizer.get_feature_names()

            aggregated_counts = [0] * self._max_ngram_size

            # Create a list of the aggregated counts of each ngram size.
            for index in np.nditer(np.nonzero(ngram_counts[0]), ['zerosize_ok']):
                aggregated_counts[len(features[index].split()) - 1] += ngram_counts[0][index]

            for index, count in enumerate(aggregated_counts):
                ngrams[index].append(count)

        return ngrams

    def _get_refuting_words(self):
        """ Retrieves headlines of the articles and indicates a count of each of the refuting words in the headline.
        Returns a list containing the number of refuting words found (at lease once) in the headline. """

        features = []

        for stance in tqdm.tqdm(self._stances):
            # print "[DEBUG] stance ", stance
            count = [1 if refute_word in stance['Headline'] else 0 for refute_word in self._refuting_words]
            # print "[DEBUG] count ", count
            features.append(count)
        # print "[DEBUG] features", features
        return features

    def _polarity_feature(self):
        _refuting_words = ['fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not',
                           'despite', 'nope', 'nowhere', 'doubt', 'doubts', 'bogus', 'debunk', 'pranks',
                           'retract', 'nothing', 'never', 'none', 'budge']

        def determine_polarity(text):
            tokens = tokenize_text(text)
            return sum([token in _refuting_words for token in tokens]) % 2

        polarities = [[], []]
        for stance in tqdm.tqdm(self._stances):
            polarities[0].append(determine_polarity(stance['Headline']))
            polarities[1].append(determine_polarity(self._articles.get(stance['Body ID'])))

        return polarities


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    feature_data = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    feature_generator = FeatureGenerator(feature_data.get_clean_articles(), feature_data.get_clean_stances())
    features = feature_generator.get_features()
    feature_matrix = features['feature_matrix']
    feature_names = features['feature_names']

    # feature_generator._get_ngrams(3)
    # feature_generator._get_refuting_words()
    # X = feature_generator._polarity_feature()
