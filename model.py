import numpy as np
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold

from feature_generation import FeatureGenerator
from FeatureData import FeatureData


class Model(object):
    def __init__(self, modelType):
        self._stance_map = {'unrelated': 0, 'discuss': 1, 'agree': 2, 'disagree': 3}
        self._model_type = modelType
        self._features_for_X1= [
           #'refuting',
           'ngrams',
           #'polarity',
           'named',
           #'vader',
           'jaccard',
           'quote_analysis',
            'lengths'
        ]
        self._features_for_X2 = [
            # 'refuting',
            'ngrams',
            # 'polarity',
            'named',
            # 'vader',
            'jaccard',
            'quote_analysis',
            'lengths'
        ]

    def get_data(self, body_file, stance_file, features_directory):
        feature_data = FeatureData(body_file, stance_file)
        X_train = FeatureGenerator.get_features_from_file(use=self._features_for_X1,
                                                           features_directory=features_directory)
        y_train = np.asarray([self._stance_map[stance['Stance']] for stance in feature_data.stances])

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
            classifier = svm.SVC(decision_function_shape='ovr', cache_size=1000)
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

def recal(actual, predicted, stance_map):
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
    reduced_agree_indices = agree_indices[:num_disagree]
    reduced_discuss_indices = discuss_indices[:num_disagree]
    reduced_unrelated_indices = unrelated_indices[:num_disagree]

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

if __name__ == '__main__':
    # SVM Model
    model1 = Model('svm')

    # NN model
    model2 = Model('nn')

    data = model1.get_data('data/combined_bodies.csv', 'data/combined_stances.csv', 'combined_features')

    X1 = data['X']
    y1 = data['y']

    stratify_data = True
    if stratify_data:
        stratified = stratify(X1, y1)
        X1 = stratified['X']
        y1 = stratified['y']

    ##
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    kfold = StratifiedKFold(n_splits=10)
    for train_indices, test_indices in kfold.split(X1, y1):
        X1_train = X1[train_indices]
        y1_train = y1[train_indices]

        # remove rows of the unrelated class for X2_train and y2_train
        mask = np.ones(len(X1_train), dtype=bool)
        unrelated_rows = []
        for i, stance in enumerate(y1_train):
            if stance == 0:
                unrelated_rows.append(i)
        mask[unrelated_rows] = False

        X2_train = X1_train[mask]
        y2_train = y1_train[mask]
        X_test = X1[test_indices]
        y_test = y1[test_indices]

        # phase 1: SVM classifier for unrelated/related classification
        # phase 2: Neural Net Classifier for agree, disagree, discuss
        SVM_classifier = model1.get_trained_classifier(X1_train, y1_train)
        NN_classifier = model2.get_trained_classifier(X2_train, y2_train)
        y_predicted = model1.test_classifier(SVM_classifier, X_test)

        mask = np.ones(len(y_predicted), dtype=bool)
        related_rows = []
        for i, stance in enumerate(y_predicted):
            if stance == 0:
                related_rows.append(i)
        mask[related_rows] = False

        X2_test = X_test[mask]
        y2_predicted = model2.test_classifier(NN_classifier, X2_test)

        # add agree, disagree, discuss results back into y_predicted
        current_index = 0
        for i, stance in enumerate(y_predicted):
            if stance != 0:
                y_predicted[i] = y2_predicted[current_index]
                current_index += 1


        precision_scores.append(precision(y_test, y_predicted, model1._stance_map))
        recall_scores.append(recal(y_test, y_predicted, model1._stance_map))
        accuracy_scores.append(accuracy(y_test, y_predicted, model1._stance_map))

    print ''
    print 'Kfold precision averages: ' + str(score_average(precision_scores))
    print 'Kfold recall averages: ' + str(score_average(recall_scores))
    print 'Kfold accuracy averages: ' + str(score_average(accuracy_scores))

