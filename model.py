import numpy as np
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

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
    for stance, index in stance_map.iteritems():
        truePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] == index])
        falsePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] != index])
        try:
            print stance + ": " + str(100 * float(truePositive) / (truePositive + falsePositive + 1))
        except ZeroDivisionError:
            print "Zero"

def recal(actual, predicted, stance_map):
    print "Recall"
    pairs = zip(actual, predicted)
    for stance, index in stance_map.iteritems():
        truePositive = np.count_nonzero([x[1] == index for x in pairs if x[0] == index])
        falseNegative = np.count_nonzero([x[1] != index for x in pairs if x[0] == index])
        try:
            print stance + ": " + str(100 * float(truePositive) / (truePositive + falseNegative + 1))
        except ZeroDivisionError:
            print "Zero"

def accuracy(actual, predicted, stance_map):
    print "Accuracy"
    pairs = zip(actual, predicted)
    for stance, index in stance_map.iteritems():
        accurate = np.count_nonzero([x[1] == index and x[1] == x[0] for x in pairs])
        total = np.count_nonzero([x[0] == index for x in pairs])
        try:
            print stance + ": " + str(100 * float(accurate)/total)
        except ZeroDivisionError:
            print "Zero"

if __name__ == '__main__':
    # SVM Model
    model = Model('svm')

    # NN model
    #model = Model('nn')

    data = model.get_data('data/train_bodies.csv', 'data/train_stances.csv', 'features')
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
