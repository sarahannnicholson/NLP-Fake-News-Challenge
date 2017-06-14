import numpy as np
from sklearn import svm, preprocessing

from feature_generation import FeatureGenerator
from FeatureData import FeatureData


class SVMModel(object):
    def __init__(self):
        self._stance_map = {'unrelated': 0, 'discuss': 2, 'agree': 3, 'disagree': 4}

    def get_trained_classifier(self):
        """Trains the svm classifier and returns the trained classifier to be used for prediction on test data. Note
        that stances in test data will need to be translated to the numbers shown in self._stance_map."""
        feature_data = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')

        X_train = FeatureGenerator.get_features_from_file()
        y_train = np.asarray([self._stance_map[stance['Stance']] for stance in feature_data.stances])

        # Scale features to range[0, 1] to prevent larger features from dominating smaller ones
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)

        svm_classifier = svm.SVC(decision_function_shape='ovo')
        svm_classifier.fit(X_train, y_train)

        return svm_classifier


if __name__ == '__main__':
    model = SVMModel()
    classifier = model.get_trained_classifier()
