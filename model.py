import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from feature_generation import FeatureGenerator
from FeatureData import FeatureData
import scorer

class Model(object):
    def __init__(self, modelType, features):
        self._stance_map = {'unrelated': 0, 'discuss': 1, 'agree': 2, 'disagree': 3}
        self._model_type = modelType
        self._features_for_X1 = features
        self._feature_col_names = []

    def get_data(self, body_file, stance_file, features_directory):
        feature_data = FeatureData(body_file, stance_file)
        X_train, self._feature_col_names = FeatureGenerator.get_features_from_file(use=self._features_for_X1,
                                                        features_directory=features_directory)

        print "X_train shape:", X_train.shape
        y_train = np.asarray([self._stance_map[stance['Stance']] for stance in feature_data.stances])
        print "col names", self._feature_col_names
        # Scale features to range[0, 1] to prevent larger features from dominating smaller ones
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)

        return {'X':X_train, 'y':y_train}

    def related_unrelated(self, y):
        return [x > 0 for x in y]

    def get_trained_classifier(self, X_train, y_train):
        """Trains the model and returns the trained classifier to be used for prediction on test data. Note
        that stances in test data will need to be translated to the numbers shown in self._stance_map."""
        if self._model_type == 'svm':
            classifier = svm.SVC(kernel='linear')
        elif self._model_type == 'nn':
            classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,), random_state=1)

        classifier.fit(X_train, y_train)
        return classifier


    def test_classifier(self, classifier, X_test):
        return classifier.predict(X_test)


def precision(actual, predicted, stance_map):
    pairs = zip(actual, predicted)
    print "Precision"
    scores = {stance: None for stance in stance_map.iterkeys()}
    for stance, index in stance_map.iteritems():
        truePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] == index])
        falsePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] != index])
        try:
            precision = 100 * float(truePositive) / (truePositive + falsePositive + 1)
            scores[stance] = precision
            print stance + ": " + str(precision)
        except ZeroDivisionError:
            print "Zero"

    return scores

def recall(actual, predicted, stance_map):
    print "Recall"
    pairs = zip(actual, predicted)
    scores = {stance: None for stance in stance_map.iterkeys()}
    for stance, index in stance_map.iteritems():
        truePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] == index])
        falseNegative = np.count_nonzero([x[1] != index for x in pairs if x[0] == index])
        try:
            recall = 100 * float(truePositive) / (truePositive + falseNegative + 1)
            scores[stance] = recall
            print stance + ": " + str(recall)
        except ZeroDivisionError:
            print "Zero"

    return scores

def accuracy(actual, predicted, stance_map):
    print "Accuracy"
    pairs = zip(actual, predicted)
    scores = {stance: None for stance in stance_map.iterkeys()}
    for stance, index in stance_map.iteritems():
        accurate = np.count_nonzero([x[1] == index and x[1] == x[0] for x in pairs])
        total = np.count_nonzero([x[0] == index for x in pairs])
        try:
            accuracy = 100 * float(accurate)/total
            scores[stance] = accuracy
            print stance + ": " + str(accuracy)
        except ZeroDivisionError:
            print "Zero"

    return scores

def stratify(X, y):
    """ Returns X and y matrices with an even distribution of each class """
    # Find the indices of each class
    disagree_indices = np.where(y == 3)[0]
    agree_indices = np.where(y == 2)[0]
    discuss_indices = np.where(y == 1)[0]
    unrelated_indices = np.where(y == 0)[0]

    num_disagree = disagree_indices.shape[0]

    # Take the first num_disagrees entries for each class
    reduced_agree_indices = agree_indices[:len(agree_indices)]
    reduced_discuss_indices = discuss_indices[:len(discuss_indices)]
    reduced_unrelated_indices = unrelated_indices[:(num_disagree + len(agree_indices) + len(discuss_indices))]

    # Recombine into stratified X and y matrices
    X_stratified = np.concatenate([X[disagree_indices], X[reduced_agree_indices], X[reduced_discuss_indices],
                                   X[reduced_unrelated_indices]], axis=0)
    y_stratified = np.concatenate([y[disagree_indices], y[reduced_agree_indices], y[reduced_discuss_indices],
                                   y[reduced_unrelated_indices]], axis=0)

    return {'X': X_stratified, 'y': y_stratified}

def score_average(scores):
    """ Used to calculate score averages resulting from kfold validation. """
    # Calculate averages for precision, recall, and accuracy
    score_sums = {stance: 0 for stance in model1._stance_map.iterkeys()}
    invalid_counts = {stance: 0 for stance in
                      model1._stance_map.iterkeys()}  # Count number of zero division errors and exclude from averages

    for result in scores:
        for stance in model1._stance_map.iterkeys():
            if result[stance] != None:
                score_sums[stance] += result[stance]
            else:
                invalid_counts[stance] += 1

    # Dictionary containing average scores for each stance
    return {stance: score_sums[stance]/(len(scores) - invalid_counts[stance]) for stance in model1._stance_map.iterkeys()}

def convert_stance_to_related(y):
    for stance, i in enumerate(y):
        if stance != 0:
            y[i] = 1
    return y

def plot_coefficients(classifier, feature_names, i, k):
     top_features=len(feature_names)/2
     coef = classifier.coef_[0]

     top_positive_coefficients = np.argsort(coef)[-top_features:]
     top_negative_coefficients = np.argsort(coef)[:top_features]
     top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

     # create plot
     plt.figure(figsize=(30, 15))
     colors = ['#cccccc' if c < 0 else 'teal' for c in coef[top_coefficients]]
     plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation='vertical', ha='right')
     plt.savefig("graphs/plot-NN_model" + str(i) + "_kfold" + str(k) + ".png")

def map_stances(y):
    stance_map = {0: 'unrelated', 1: 'discuss', 2: 'agree', 3: 'disagree'}
    return [stance_map.get(key) for key in y]

if __name__ == '__main__':
    startTime = datetime.now()
    features_for_X1 = [
        #'refuting',
        'ngrams',
        #'polarity',
        'named',
        #'vader',
        'jaccard',
        'quote_analysis',
        'lengths',
        #'punctuation_frequency',
        'word2Vec'
    ]
    features_for_X2 = [
        'refuting',
        'ngrams',
        'polarity',
        'named',
        'vader',
        'jaccard',
        'quote_analysis',
        'lengths',
        'punctuation_frequency',
        'word2Vec'
    ]

    model1 = Model('nn', features_for_X1)
    model2 = Model('nn', features_for_X2)

    data = model1.get_data('data/combined_bodies.csv', 'data/combined_stances.csv', 'combined_features')
    data2 = model2.get_data('data/combined_bodies.csv', 'data/combined_stances.csv', 'combined_features')

    X1 = data['X']
    y1 = data['y']
    X2 = data2['X']
    y2 = data2['y']

    stratify_data = False
    if stratify_data:
        stratified = stratify(X1, y1)
        X1 = stratified['X']
        y1 = stratified['y']
        X2 = stratified['X']
        y2 = stratified['y']

    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    competition_scores = []

    kfold = StratifiedKFold(n_splits=10)
    k=0

    for train_indices, test_indices in kfold.split(X1, y1):
        X1_train = X1[train_indices]
        y1_train = [int(s != 0) for s in y1[train_indices]]
        X2_train = X2[train_indices]
        y2_train = y2[train_indices]

        #Save testing data
        X1_test = X1[test_indices]
        X2_test = X2[test_indices]
        y_test = y2[test_indices]

        # remove rows of the unrelated class for X2_train and y2_train
        X2_train_filtered = X2_train[np.nonzero(y1_train)]
        y2_train_filtered = y2_train[np.nonzero(y1_train)]

        # phase 1: Neural Net Classifier for unrelated/related classification
        print "#1 Train"
        print np.bincount(y1_train)
        print np.unique(y1_train)
        NN_clf1 = model1.get_trained_classifier(X1_train, y1_train)
        #plot_coefficients(NN_clf1, model1._feature_col_names, 1, k)

        # phase 2: Neural Net Classifier for agree, disagree, discuss
        print "#2 Train"
        print np.bincount(y2_train_filtered)
        print np.unique(y2_train_filtered)
        NN_clf2 = model2.get_trained_classifier(X2_train_filtered, y2_train_filtered)
        #plot_coefficients(NN_clf2, model2._feature_col_names, 2, k)
       
        y_predicted = model1.test_classifier(NN_clf1, X1_test)
        print "#1 Test"
        print np.bincount(y_predicted)
        print np.unique(y_predicted)

        y2_predicted = model2.test_classifier(NN_clf2, X2_test)
        print "#2 Test"
        print np.bincount(y2_predicted)
        print np.unique(y2_predicted)

        print "Actual Test"
        print np.bincount(y_test)
        print np.unique(y_test)

        # add agree, disagree, discuss results back into y_predicted
        for i, stance in enumerate(y_predicted):
            if stance != 0:
                y_predicted[i] = y2_predicted[i]

        print "Final"
        print np.bincount(y_predicted)
        print np.unique(y_predicted)

        precision_scores.append(precision(y_test, y_predicted, model1._stance_map))
        recall_scores.append(recall(y_test, y_predicted, model1._stance_map))
        accuracy_scores.append(accuracy(y_test, y_predicted, model1._stance_map))

        y_test= map_stances(y_test)
        y_predicted = map_stances(y_predicted)
        competition_score = scorer.report_score(y_test, y_predicted)
        competition_scores.append(competition_score)
        k+=1

    print '\nKfold precision averages: ', score_average(precision_scores)
    print 'Kfold recall averages: ', score_average(recall_scores)
    print 'Kfold accuracy averages: ', score_average(accuracy_scores)
    print 'competition score averages: ', sum(competition_scores) / len(competition_scores)

    print "time to run: ", datetime.now() - startTime
