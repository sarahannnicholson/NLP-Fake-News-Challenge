"""Methods for generating each feature can be added to the FeatureGenerator class."""

import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tqdm

from FeatureData import FeatureData, tokenize_text

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags


class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, clean_articles, clean_stances):
        self._articles = clean_articles  # dictionary {article ID: body}
        self._stances = clean_stances  # list of dictionaries
        self._max_ngram_size = 3
        self._refuting_words = ['fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt',
                                'doubts', 'bogus', 'debunk', 'pranks', 'retract']

    def get_features(self):
        """Retrieves the full set of features as a matrix (the X matrix for training). You only need to run this
        if the features have been updated since the last time they were output to a file under the features
        directory."""
        feature_names = []
        features = []

        logging.debug('Retrieving headline ngrams...')
        ngrams = np.array(self._get_ngrams())
        features.append(ngrams)
        ngram_headings = [('ngram_' + str(count)) for count in range(1, self._max_ngram_size + 1)]
        feature_names.append(ngram_headings)
        self._feature_to_csv(ngrams, ngram_headings, 'features/ngrams.csv')

        logging.debug('Retrieving refuting words...')
        refuting = np.array(self._get_refuting_words())
        features.append(refuting)
        [feature_names.append(word + '_refuting') for word in self._refuting_words]
        self._feature_to_csv(refuting, self._refuting_words, 'features/refuting.csv')

        logging.debug('Retrieving polarity...')
        polarity = np.array(self._polarity_feature())
        features.append(polarity)
        feature_names.append('headline_polarity')
        feature_names.append('article_polarity')
        self._feature_to_csv(polarity, ['headline_polarity', 'article_polarity'], 'features/polarity.csv')

        logging.debug('Retrieving named entity cosine...')
        named_cosine = np.array(self._named_entity_feature()).reshape(len(self._stances), 1)
        features.append(named_cosine)
        feature_names.append('named_cosine')
        self._feature_to_csv(named_cosine, 'named_cosine', 'features/named_cosine.csv')

        return {'feature_matrix': np.concatenate(features, axis=1), 'feature_names': feature_names}

    def _feature_to_csv(self, feature, feature_headers, output_path):
        """Outputs a feature to a csv file. feature is a 2d numpy matrix containing the feature values and
        feature headers is a list containing the feature's column headings."""
        header = ','.join(feature_headers)
        np.savetxt(fname=output_path, X=feature, delimiter=',', header=header, comments='')

    def _get_ngrams(self):
        """Retrieves counts for ngrams of the article title in the article itself, from one up to size max_ngram_size.
        Returns a list of lists, each containing the counts for a different size of ngram."""
        ngrams = []

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

            aggregated_counts = [0 for _ in range(self._max_ngram_size)]

            # Create a list of the aggregated counts of each ngram size.
            for index in np.nditer(np.nonzero(ngram_counts[0]), ['zerosize_ok']):
                aggregated_counts[len(features[index].split()) - 1] += ngram_counts[0][index]

            ngrams.append(aggregated_counts)

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

        polarities = []
        for stance in tqdm.tqdm(self._stances):
            headline_polarity = determine_polarity(stance['Headline'])
            body_polarity = determine_polarity(self._articles.get(stance['Body ID']))
            polarities.append([headline_polarity, body_polarity])

        return polarities

    def _named_entity_feature(self):
        """ Retrieves a list of Named Entities from the Headline and Body.
        Returns a list containing the cosine simmilarity between the counts of the named entities """

        def determine_named_entities(text):
            named_tags = pos_tag(word_tokenize(text.encode('ascii', 'ignore')))
            return " ".join([name[0] for name in named_tags if name[1].startswith("NN")])

        named_cosine = []
        for stance in tqdm.tqdm(self._stances):
            head = determine_named_entities(stance['Headline'])
            body = determine_named_entities(self._articles.get(stance['Body ID']))
            vect = TfidfVectorizer(min_df=1)
            tfidf = vect.fit_transform([head,body])
            cosine = (tfidf * tfidf.T).todense().tolist()
            if len(cosine) == 2:
                named_cosine.append(cosine[1][0])
            else:
                named_cosine.append(0)

        named_cosine = [item/max(named_cosine) for item in named_cosine]
        return named_cosine


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
