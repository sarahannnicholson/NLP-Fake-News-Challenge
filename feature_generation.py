"""Methods for generating each feature can be added to the FeatureGenerator class."""

import logging
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tqdm

from FeatureData import FeatureData, tokenize_text

from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.chunk import tree2conlltags
from nltk.stem import PorterStemmer


class FeatureGenerator(object):
    """Class responsible for generating each feature used in the X matrix."""

    def __init__(self, clean_articles, clean_stances, original_articles, load_data=True):
        self._articles = clean_articles  # dictionary {article ID: body}
        self._original_articles = original_articles
        self._stances = clean_stances  # list of dictionaries
        self._max_ngram_size = 3
        self._refuting_words = ['fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt',
                                'doubts', 'bogus', 'debunk', 'pranks', 'retract']

    @staticmethod
    def get_features_from_file(features_directory, use=[]):
        """Returns the full set of features as a 2d numpy array by concatenating all of the feature csv files located
        under the features directory."""
        features = []
        for feature_csv in os.listdir(features_directory):
            if np.count_nonzero([feature_csv.startswith(x) for x in use]):
                with open(os.path.join(features_directory, feature_csv)) as f:
                    content = np.loadtxt(fname=f, comments='', delimiter=',', skiprows=1)

                    if len(content.shape) == 1:
                        content = content.reshape(content.shape[0], 1)

                    features.append(content)

        return np.concatenate(features, axis=1)

    def get_features(self, features_directory="features"):
        """Retrieves the full set of features as a matrix (the X matrix for training). You only need to run this
        if the features have been updated since the last time they were output to a file under the features
        directory."""
        feature_names = []
        features = []

        if False:
            logging.debug('Retrieving headline ngrams...')
            ngrams = np.array(self._get_ngrams())
            features.append(ngrams)
            ngram_headings = [('ngram_' + str(count)) for count in range(1, self._max_ngram_size + 1)]
            feature_names.append(ngram_headings)
            self._feature_to_csv(ngrams, ngram_headings, features_directory+'/ngrams.csv')

        if False:
            logging.debug('Retrieving refuting words...')
            refuting = np.array(self._get_refuting_words())
            features.append(refuting)
            [feature_names.append(word + '_refuting') for word in self._refuting_words]
            self._feature_to_csv(refuting, self._refuting_words, features_directory+'/refuting.csv')

        if False:
            logging.debug('Retrieving polarity...')
            polarity = np.array(self._polarity_feature())
            features.append(polarity)
            feature_names.append('headline_polarity')
            feature_names.append('article_polarity')
            self._feature_to_csv(polarity, ['headline_polarity', 'article_polarity'], features_directory+'/polarity.csv')

        if True:
            logging.debug('Retrieving named entity cosine...')
            named_cosine = np.array(self._named_entity_feature()).reshape(len(self._stances), 1)
            features.append(named_cosine)
            feature_names.append('named_cosine')
            self._feature_to_csv(named_cosine, ['named_cosine'], features_directory+'/named_cosine.csv')

        if True:
            logging.debug('Retrieving VADER...')
            vader = np.array(self._vader_feature()).reshape(len(self._stances), 2)
            features.append(vader)
            feature_names.append('vader_pos')
            feature_names.append('vader_neg')
            self._feature_to_csv(vader, ['vader'], features_directory+'/vader.csv')

        if False:
            logging.debug('Retrieving jaccard similarities...')
            jaccard = np.array(self._get_jaccard_similarity()).reshape(len(self._stances), 1)
            features.append(jaccard)
            feature_names.append('jaccard_similarity')
            self._feature_to_csv(jaccard, ['jaccard_similarity'], features_directory+'/jaccard_similarity.csv')

        return {'feature_matrix': np.concatenate(features, axis=1), 'feature_names': feature_names}

    def _feature_to_csv(self, feature, feature_headers, output_path):
        """Outputs a feature to a csv file. feature is a 2d numpy matrix containing the feature values and
        feature headers is a list containing the feature's column headings."""
        header = ','.join(feature_headers)
        np.savetxt(fname=output_path, X=feature, delimiter=',', header=header, comments='')

    @staticmethod
    def combine_train_and_test_features():
        """ Concatenates training and competition features into single files under the 'combined_features'
        directory. """
        for feature in os.listdir('features'):
            with open(os.path.join('features', feature), 'r+') as f_train, open(os.path.join('test_features', feature), 'r+') as f_test:
                f_train.flush()
                f_test.flush()
                os.fsync(f_train.fileno())
                os.fsync(f_test.fileno())

                with open(os.path.join('combined_features', feature), 'wb') as f_combined:
                    for line in f_train:
                        f_combined.write(line)

                    next(f_test)
                    for line in f_test:
                        f_combined.write(line)


    def _get_ngrams(self):
        """Retrieves counts for ngrams of the article title in the article itself, from one up to size max_ngram_size.
        Returns a list of lists, each containing the counts for a different size of ngram."""
        ngrams = []

        for stance in tqdm.tqdm(self._stances):
            # Retrieves the vocabulary of ngrams for the headline.
            stance_vectorizer = CountVectorizer(input=stance['Headline'], ngram_range=(1, self._max_ngram_size),
                                                binary=True)
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

            # attempt to standardize ngram counts across headlines and bodies of varying length by dividing total
            # ngram hits by the length of the headline. These will need to be normalized later so they lie
            # between 0 and 1.
            standardized_counts = [1.0*count/len(stance['Headline'].split()) for count in aggregated_counts]

            ngrams.append(standardized_counts)

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
        stemmer = PorterStemmer()
        def get_tags(text):
            return pos_tag(word_tokenize(text.encode('ascii', 'ignore')))

        def filter_pos(named_tags, tag):
            return " ".join([stemmer.stem(name[0]) for name in named_tags if name[1].startswith(tag)])

        named_cosine = []
        tags = ["NN"]
        for stance in tqdm.tqdm(self._stances):
            stance_cosine = []
            head = get_tags(stance['originalHeadline'])
            body = get_tags(self._original_articles.get(stance['Body ID'])[:255])

            for tag in tags:
                head_f = filter_pos(head, tag)
                body_f = filter_pos(body, tag)

                if head_f and body_f:
                    vect = TfidfVectorizer(min_df=1)
                    tfidf = vect.fit_transform([head_f,body_f])
                    cosine = (tfidf * tfidf.T).todense().tolist()
                    if len(cosine) == 2:
                        stance_cosine.append(cosine[1][0])
                    else:
                        stance_cosine.append(0)
                else:
                    stance_cosine.append(0)
            named_cosine.append(stance_cosine)
        return named_cosine

    def _vader_feature(self):
        sid = SentimentIntensityAnalyzer()
        features = []

        for stance in tqdm.tqdm(self._stances):
            headVader = sid.polarity_scores(stance["Headline"])
            bodyVader = sid.polarity_scores(sent_tokenize(self._original_articles.get(stance['Body ID']))[0])
            features.append(abs(headVader['pos']-bodyVader['pos']))
            features.append(abs(headVader['neg']-bodyVader['neg']))
        return features

    def _get_jaccard_similarity(self):
        """ Get the jaccard similarities for each headline and article body pair. Jaccard similarity is defined as
        J(A, B) = |A intersect B| / |A union B|. Try to normalize by only considering the first"""
        similarities = []
        for stance in tqdm.tqdm(self._stances):
            headline = set(stance['Headline'].split())
            body = set(self._articles.get(stance['Body ID']).split()[:255])
            jaccard = float(len(headline.intersection(body))) / len(headline.union(body))
            similarities.append(jaccard)

        return similarities


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    feature_data = FeatureData('data/competition_test_bodies.csv', 'data/competition_test_stances.csv')
    feature_generator = FeatureGenerator(feature_data.get_clean_articles(), feature_data.get_clean_stances(), feature_data.get_original_articles())
    features = feature_generator.get_features("test_features")

    feature_data = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    feature_generator = FeatureGenerator(feature_data.get_clean_articles(), feature_data.get_clean_stances(), feature_data.get_original_articles())
    features = feature_generator.get_features()

    # Concatenate competition and training features to get combined files
    FeatureGenerator.combine_train_and_test_features()
