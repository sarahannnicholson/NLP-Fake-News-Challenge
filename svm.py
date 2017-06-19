import numpy as np
from sklearn import svm, preprocessing

from feature_generation import FeatureGenerator
from FeatureData import FeatureData


class SVMModel(object):
    def __init__(self):
        self._stance_map = {'unrelated': 0, 'discuss': 2, 'agree': 3, 'disagree': 4}
        self._use_features= [
#            'refuting',
            'ngrams',
#            'polarity',
            'named'
        ]

    def get_data(self, body_file, stance_file):
        feature_data = FeatureData(body_file, stance_file)
        X_train = FeatureGenerator.get_features_from_file(self._use_features)
        y_train = np.asarray([self._stance_map[stance['Stance']] for stance in feature_data.stances])

        # Scale features to range[0, 1] to prevent larger features from dominating smaller ones
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)

        return {'X':X_train, 'y':y_train}

    def related_unrelated(self, y):
        return [x > 0 for x in y]

    def get_trained_classifier(self, X_train, y_train):
        """Trains the svm classifier and returns the trained classifier to be used for prediction on test data. Note
        that stances in test data will need to be translated to the numbers shown in self._stance_map."""
        svm_classifier = svm.SVC(decision_function_shape='ovr', cache_size=1000)
        svm_classifier.fit(X_train, y_train)
        return svm_classifier

    def test_classifier(self, svm_classifier, X_test, y_test):
        predicted = []
        for i, stance in enumerate(y_test):
            predicted.append(svm_classifier.predict([X_test[i]])[0])

        return predicted

    def precision(self, actual, predicted):
        pairs = zip(actual, predicted)
        truePositive = np.count_nonzero([x[1] > 0 for x in pairs if x[0] > 0])
        falsePositive = np.count_nonzero([x[1] > 0 for x in pairs if x[0] == 0])
        return float(truePositive) / (truePositive + falsePositive + 1)

    def recal(self, actual, predicted):
        pairs = zip(actual, predicted)
        truePositive = np.count_nonzero([x[1] > 0 for x in pairs if x[0] > 0])
        falseNegative = np.count_nonzero([x[1] == 0 for x in pairs if x[0] > 0])
        return float(truePositive) / (truePositive + falseNegative + 1)

    def accuracy(self, actual, predicted):
        pairs = zip(actual, predicted)
        accurate = np.count_nonzero([x[1] == x[0] for x in pairs])
        return float(accurate)/len(pairs)

if __name__ == '__main__':
    model = SVMModel()
    data = model.get_data('data/train_bodies.csv', 'data/train_stances.csv')
    testNum = 1000

    X_test = data['X'][-testNum:]
    X_train = data['X'][:-testNum]

    Only_R_UR = True
    if Only_R_UR:
        y_test = model.related_unrelated(data['y'][-testNum:])
        y_train = model.related_unrelated(data['y'][:-testNum])
    else:
        y_test = data['y'][-testNum:]
        y_train = data['y'][:-testNum]

    classifier = model.get_trained_classifier(X_train, y_train)
    predicted = model.test_classifier(classifier, X_test, y_test)

    print str(model._use_features)
    print "Precision %f" % model.precision(y_test, predicted)
    print "Recal %f" % model.recal(y_test, predicted)
    print "Accuracy %f" % model.accuracy(y_test, predicted)
