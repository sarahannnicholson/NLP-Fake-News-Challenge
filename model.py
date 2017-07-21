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
        self._use_features= [
           'refuting',
           'ngrams',
           'polarity',
           'named',
           #'vader',
           'jaccard'
        ]

    def get_data(self, body_file, stance_file, features_directory):
        feature_data = FeatureData(body_file, stance_file)
        X_train = FeatureGenerator.get_features_from_file(use=self._use_features, features_directory=features_directory)
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


    def test_classifier(self, classifier, X_test, y_test):
        predicted = []
        for i, stance in enumerate(y_test):
            predicted.append(classifier.predict([X_test[i]])[0])

        return predicted

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
    score_sums = {stance: 0 for stance in model._stance_map.iterkeys()}
    invalid_counts = {stance: 0 for stance in
                      model._stance_map.iterkeys()}  # Count number of zero division errors and exclude from averages

    for result in scores:
        for stance in model._stance_map.iterkeys():
            if result[stance] != None:
                score_sums[stance] += result[stance]
            else:
                invalid_counts[stance] += 1

    # Dictionary containing average scores for each stance
    return {stance: score_sums[stance]/(len(scores) - invalid_counts[stance]) for stance in model._stance_map.iterkeys()}

if __name__ == '__main__':
    # SVM Model
    model = Model('svm')

    # NN model
    #model = Model('nn')

    data = model.get_data('data/combined_bodies.csv', 'data/combined_stances.csv', 'combined_features')

    X = data['X']
    y = data['y']

    stratify_data = True
    if stratify_data:
        stratified = stratify(X, y)
        X = stratified['X']
        y = stratified['y']

    kfold = True
    if kfold:
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        kfold = StratifiedKFold(n_splits=10)

        for train_indices, test_indices in kfold.split(X, y):
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            classifier = model.get_trained_classifier(X_train, y_train)
            predicted = model.test_classifier(classifier, X_train, y_train)

            print str(model._use_features)
            precision_scores.append(precision(y, predicted, model._stance_map))
            recall_scores.append(recal(y, predicted, model._stance_map))
            accuracy_scores.append(accuracy(y, predicted, model._stance_map))

        print ''
        print 'Kfold precision averages: ' + str(score_average(precision_scores))
        print 'Kfold recall averages: ' + str(score_average(recall_scores))
        print 'Kfold accuracy averages: ' + str(score_average(accuracy_scores))
    else:
        data = model.get_data('data/combined_bodies.csv', 'data/combined_stances.csv', 'features')
        testNum = 1000

        X_test = data['X'][-testNum:]
        X_train = data['X'][:-testNum]

        Only_R_UR = False
        if Only_R_UR:
            y_test = model.related_unrelated(data['y'][-testNum:])
            y_train = model.related_unrelated(data['y'][:-testNum])
        else:
            y_test = data['y'][-testNum:]
            y_train = data['y'][:-testNum]

        classifier = model.get_trained_classifier(X_train, y_train)
        predicted = model.test_classifier(classifier, X_test, y_test)

        print str(model._use_features)
        precision(y_test, predicted, model._stance_map)
        recal(y_test, predicted, model._stance_map)
        accuracy(y_test, predicted, model._stance_map)
