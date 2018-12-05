from read_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    data = execute()
    print(data.shape)
    X = data[:, :16]  # 16 features
    y = data[:, 16]
    print(X.shape)
    print(y.shape)
    print(y)
    train_features, test_features, train_labels, test_labels = train_test_split(X, y,
                                                                                test_size=0.25)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True)
    clf.fit(X, y)
    print(clf.feature_importances_)
    # print(clf.oob_decision_function_)
    print(clf.oob_score_)
    predictions = clf.predict(test_features)
    errors = abs(predictions - test_labels)
    print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(test_labels))
    print("Accuracy:", np.count_nonzero(errors)/len(test_labels))
