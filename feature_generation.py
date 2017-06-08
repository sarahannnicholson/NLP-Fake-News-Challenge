"""Methods for generating each feature can be added to the FeatureGenerator class."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from FeatureData import FeatureData, tokenize_text


class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, clean_articles, clean_stances):
        self.articles = clean_articles # dictionary {article ID: body}
        self.stances = clean_stances # list of dictionaries

    def get_features(self):
        """Retrieves the full set of features as a matrix (the X matrix for training.)"""
        pass

    def _get_ngrams(self, max_ngram_size):
        """Retrieves counts for ngrams of the article title in the article itself, from one up to size max_ngram_size.
        Returns a list of lists, each containing the counts for a different size of ngram."""
        ngrams = [[] for _ in range(max_ngram_size)]

        for stance in self.stances:
            # Retrieves the vocabulary of ngrams for the headline.
            stance_vectorizer = CountVectorizer(input=stance['Headline'], ngram_range=(1, max_ngram_size))
            stance_vectorizer.fit_transform([stance['Headline']]).toarray()

            # Search the article text and count headline ngrams.
            vocab = stance_vectorizer.get_feature_names()
            vectorizer = CountVectorizer(input=self.articles[stance['Body ID']], vocabulary=vocab,
                                         ngram_range=(1, max_ngram_size))
            ngram_counts = vectorizer.fit_transform([self.articles[stance['Body ID']]]).toarray()
            features = vectorizer.get_feature_names()

            aggregated_counts = [0] * max_ngram_size

            # Create a list of the aggregated counts of each ngram size.
            for index in np.nditer(np.nonzero(ngram_counts[0]), ['zerosize_ok']):
                aggregated_counts[len(features[index].split()) - 1] += ngram_counts[0][index]

            for index, count in enumerate(aggregated_counts):
                ngrams[index].append(count)

            # DEBUG
            # if np.count_nonzero(ngram_counts[0]) > 1:
            #     for x in np.nditer(np.nonzero(ngram_counts[0])):
            #         print features[x]
            #     print aggregated_counts
            #     print 'Headline:'
            #     print stance['Headline']
            #     print 'Article:'
            #     print self.articles[stance['Body ID']]
            #     print features
            #     print ngram_counts
        # print ngrams

    def _get_refuting_words(self):
        """ Retrieves headlines of the articles and indicates a count of each of the refuting words in the headline.
        Returns a list containing the number of refuting words found (at lease once) in the headline. """

        _refuting_words = [ 'fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not',
        'despite', 'nope', 'doubt', 'doubts', 'bogus', 'debunk', 'pranks', 'retract']

        features = []
        for stance in self.stances:
            #print "[DEBUG] stance ", stance
            count = [1 if refute_word in stance['Headline'] else 0 for refute_word in _refuting_words]
            #print "[DEBUG] count ", count
            features.append(count)
        #print "[DEBUG] features", features
        return features

    def _polarity_feature(self):
        _refuting_words = [ 'fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not',
        'despite', 'nope', 'nowhere', 'doubt', 'doubts', 'bogus', 'debunk', 'pranks',
        'retract', 'nothing', 'never', 'none', 'budge']


        def determine_polarity(text):
            tokens = tokenize_text(text)
            return sum([token in _refuting_words for token in tokens]) % 2

        X = []
        for dict in self.stances:
            features = []
            features.append(determine_polarity(dict['Headline']))
            features.append(determine_polarity(self.articles.get(dict['Body ID'])))
            X.append(features)

        return np.array(X)


if __name__ == '__main__':
    feature_data = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    feature_generator = FeatureGenerator(feature_data.get_clean_articles(), feature_data.get_clean_stances())

    feature_generator._get_ngrams(3)
    feature_generator._get_refuting_words()
    X = feature_generator._polarity_feature()









    #
