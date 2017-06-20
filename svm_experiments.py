from svm import SVMModel

if __name__ == '__main__':
    features = [
       # 'refuting',
       'ngrams',
       # 'polarity',
       'named',
       # 'jaccard'
    ]

    model = SVMModel()
    train_data = model.get_data('data/train_bodies.csv', 'data/train_stances.csv')
    test_data = model.get_data('data/competition_test_bodies.csv', 'data/competition_test_stances.csv')

    X_test = test_data['X']
    X_train = train_data['X']

    Only_R_UR = True
    if Only_R_UR is True:
        y_test = model.related_unrelated(test_data['y'])
        y_train = model.related_unrelated(train_data['y'])
    else:
        y_test = test_data['y']
        y_train = train_data['y']

    classifier = model.get_trained_classifier(X_train, y_train)
    predicted = model.test_classifier(classifier, X_test, y_test)

    print str(model._use_features)
    print "Precision %f" % model.precision(y_test, predicted)
    print "Recal %f" % model.recal(y_test, predicted)
    print "Accuracy %f" % model.accuracy(y_test, predicted)
