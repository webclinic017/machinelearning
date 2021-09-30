import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
from sklearn import datasets, preprocessing
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from utilities import visualize_classifier


def evaluate_performance(classifier, names, datasets):
    X_train, X_test, y_train, y_test = datasets

    # train
    classifier.fit(X_train, y_train)
    # important step -------
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')

    # Compute the classifier output on the test dataset and visualize it
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')

    # Evaluate classifier performance
    class_names = names
    print("\n" + "#"*40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                target_names=class_names))
    print("#"*40 + "\n")
    print("#"*40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names))
    print("#"*40 + "\n")


def building_decision_tree():
    ''' Building a Decision Tree classifier '''
    # Load input data
    input_file = 'data\\data_decision_trees.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Separate input data into two classes based on labels
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])

    # Visualize input data
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
                edgecolors='black', linewidth=1, marker='x')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
    plt.title('Input data')

    # Split data into training and testing datasets
    splited_datasets = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Decision Trees classifier
    params = {'random_state': 0, 'max_depth': 4}
    classifier = DecisionTreeClassifier(**params)

    # Train and evaluate classifier performance
    class_names = ['Class-0', 'Class-1']
    evaluate_performance(classifier, class_names, splited_datasets)
    plt.show()


def measure_confidence(classifier, test_datapoints):
    print("\nConfidence measure:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = f'Class-{np.argmax(probabilities)}'
        print('\nDatapoint:', datapoint)
        print('Probabilities:', probabilities)
        print('Predicted class:', predicted_class)


def random_forest(option=1):
    ''' Building Random Forest '''
    # Load input data
    input_file = 'data\\data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Separate input data into 3 classes based on labels
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    # Visualize input data
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='x')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='^')
    plt.title('Input data')

    # Split data into training and testing datasets
    splited_datasets = train_test_split(X, y, test_size=0.25, random_state=5)
    # Ensemble learning classifier
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    if option == 1:
        ''' RandomForestClassifier '''
        classifier = RandomForestClassifier(**params)
    elif option == 2:
        ''' ExtraTreesClassifier '''
        classifier = ExtraTreesClassifier(**params)

    # Train and evaluate classifier performance
    class_names = ['Class-0', 'Class-1', 'Class-2']
    evaluate_performance(classifier, class_names, splited_datasets)

    # Estimating the confidence measure of the predictions
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4],
                                [5, 2]])

    measure_confidence(classifier, test_datapoints)
    # Visualize the datapoints
    visualize_classifier(classifier, test_datapoints,
                         [0]*len(test_datapoints),
                         'Test datapoints')
    plt.show()


def deal_imbalance(option=1):
    # Load input data
    input_file = 'data\\data_imbalance.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Separate input data into two classes based on labels
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    # Visualize input data
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
                edgecolors='black', linewidth=1, marker='x')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
    plt.title('Input data')

    # # Split data into training and testing datasets
    splited_datasets = train_test_split(X, y, test_size=0.25, random_state=5)

    # Extremely Random Forests classifier: let add "balance" param
    if option == 1:
        ''' no balanced '''
        params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    else:
        ''' balanced '''
        params = {'n_estimators': 100, 'max_depth': 4,
                  'random_state': 0, 'class_weight': 'balanced'}
    classifier = ExtraTreesClassifier(**params)

    # Train and evaluate classifier performance
    class_names = ['Class-0', 'Class-1']
    evaluate_performance(classifier, class_names, splited_datasets)
    plt.show()


def find_optimal_params():
    # Load input data
    input_file = 'data\\data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Separate input data into two classes based on labels
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Define the parameter grid
    parameter_grid = [{'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                      {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}]
    metrics = ['precision_weighted', 'recall_weighted']
    for metric in metrics:
        print("\n##### Searching optimal parameters for", metric)
        classifier = GridSearchCV(ExtraTreesClassifier(random_state=0),
                                  parameter_grid, cv=5, scoring=metric)
        classifier.fit(X_train, y_train)

        # score for each parameter combination
        print("\nGrid scores for the parameter grid:")

        # for parameter, score in classifier.cv_results_.items():
        #     print(parameter, score, sep=' --> ', end='\n-----------------\n')

        print("\nBest parameters:", classifier.best_params_)

        # performance report
        y_pred = classifier.predict(X_test)
        print("\nPerformance report:\n")
        print(classification_report(y_test, y_pred))


def compute_feature_importance():
    # Load housing data
    housing_data = datasets.load_boston()

    # Shuffle the data
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    # Devide and train an AdaBoostRegressor using DecisionTreeRegressor
    regressor = AdaBoostRegressor(DecisionTreeRegressor(
        max_depth=4), n_estimators=400, random_state=7)
    regressor.fit(X_train, y_train)

    # Evaluate performance of AdaBoost regressor
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\nADABOOST REGRESSOR")
    print(f"Mean squared error = {round(mse, 2)}")
    print(f"Explained variance score = {round(evs, 2)}")

    # Extract feature importances
    feature_importances = regressor.feature_importances_
    feature_names = housing_data.feature_names

    # Normalize the importance values
    feature_importances = 100.0 * \
        (feature_importances/max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title('Feature importance using AdaBoost regressor')
    plt.show()


def predict_traffic():
    # Load input data
    input_file = 'data\\traffic_data.txt'
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = line[:-1].split(',')
            data.append(items)
    data = np.array(data)

    # Convert string data to numerical data
    label_encoder = []
    X_encoded = np.empty(data.shape)
    for i, item in enumerate(data[0]):
        if item.isdigit():
            X_encoded[:, i] = data[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            # data = data.reshape(-1, 1)
            X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Extremely Random Forests regressor
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    regressor = ExtraTreesRegressor(**params)
    regressor.fit(X_train, y_train)

    # Compute the regressor performance on test data
    y_pred = regressor.predict(X_test)
    print(
        f"Mean absolute error: {round(mean_absolute_error(y_test, y_pred),2)}")

    # Testing encoding on single data instance
    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
    test_datapoint_encoded = [-1] * len(test_datapoint)
    count = 0
    for i, item in enumerate(test_datapoint):
        if item.isdigit():
            test_datapoint_encoded[i] = int(test_datapoint[i])
        else:
            test_datapoint = np.reshape(test_datapoint, (-1, 1))
            test_datapoint_encoded[i] = int(
                label_encoder[count].transform(test_datapoint[i]))
            count = count + 1
    test_datapoint_encoded = np.array(test_datapoint_encoded)

    # Predict the output for the test datapoint
    pred_traffic = int(regressor.predict([test_datapoint_encoded])[0])
    print(f"Predicted traffic: {pred_traffic}")


def main():
    # building_decision_tree()
    # random_forest(int(sys.argv[1]))
    # deal_imbalance(int(sys.argv[1]))
    # find_optimal_params()
    compute_feature_importance()
    # predict_traffic()


if __name__ == "__main__":
    main()
