from svm import SVMModel
import scorer

if __name__ == '__main__':
    model = SVMModel()
    train_data = model.get_data('data/train_bodies.csv', 'data/train_stances.csv', 'features')
    test_data = model.get_data('data/competition_test_bodies.csv', 'data/competition_test_stances.csv', 'test_features')

    X_test = test_data['X']
    X_train = train_data['X']

    Only_R_UR = False
    if Only_R_UR is True:
        y_test = model.related_unrelated(test_data['y'])
        y_train = model.related_unrelated(train_data['y'])
    else:
        y_test = test_data['y']
        y_train = train_data['y']

    classifier = model.get_trained_classifier(X_train, y_train)
    predicted = model.test_classifier(classifier, X_test, y_test)

    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

    # TODO this is temporary (I hope)
    score_map = {0: 3, 1: 2, 2: 0, 3: 1}  # Maps our values to what the scorer expects

    scorer.report_score([LABELS[score_map[e]] for e in y_test],[LABELS[score_map[e]] for e in predicted])

    print str(model._use_features)
    model.precision(y_test, predicted)
    model.recal(y_test, predicted)
    model.accuracy(y_test, predicted)
