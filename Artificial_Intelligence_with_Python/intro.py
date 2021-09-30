import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter
from pickle import dump, load
from sklearn import linear_model, datasets
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score, r2_score
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC, SVR
from utilities import visualize_classifier, handling_simple_data

import warnings
warnings.filterwarnings('ignore')


def preprocessing_data(option=1):
    input_data = np.array([[5.1, -2.9, 3.3],
                           [-1.2, 7.8, -6.1],
                           [3.9, 0.4, 2.1],
                           [7.3, -9.9, -4.5]])
    if option == 1:
        ''' Loading datasets '''
        print("Example: I say Hello world")
        house_prices = datasets.load_boston()
        print(house_prices.data)

        digits = datasets.load_digits()
        print(digits.images[4])
        # print(digits)
    elif option == 2:
        ''' np array: input_data '''
        print(input_data)
    elif option == 3:
        ''' Binarize data '''
        data_binarized = preprocessing.Binarizer(
            threshold=2.1).transform(input_data)
        print("\nBinarized data:\n", data_binarized)
    elif option == 4:
        ''' Mean removal data
        taking "mean" very close to 0 and std near or equal 1.
        '''
        print("\nBEFORE:")
        print("Mean =", input_data.mean(axis=0))
        print("Std deviation =", input_data.std(axis=0))
        # Remove mean: ??? what the "scale" fucking
        data_scaled = preprocessing.scale(input_data)
        print("\nAFTER:")
        print("Mean =", data_scaled.mean(axis=0))
        print("Std deviation =", data_scaled.std(axis=0))
    elif option == 5:
        ''' Scaling in a range
        maximum value is 1 and all the other values are relative to
        this value
        '''
        data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
        print("\nMin max scaled data:\n", data_scaled_minmax)
    elif option == 6:
        ''' Normalization '''
        data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
        data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
        print("\nL1 normalized data:\n", data_normalized_l1)
        print("\nL2 normalized data:\n", data_normalized_l2)
    else:
        ''' option_purpose '''
        pass


def label_encoding(option=1):
    input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
    if option == 1:
        ''' first example '''
        # Create label encoder and fit the labels
        encoder = preprocessing.LabelEncoder()
        encoder.fit(input_labels)

        # Print the mapping
        print("\nLabel mapping:")
        for i, item in enumerate(encoder.classes_):
            print(item, '-->', i)

        # Encode a set of labels using the encoder
        test_labels = ['green', 'red', 'black']
        encoded_values = encoder.transform(test_labels)
        print("\nLabels =", test_labels)
        print("Encoded values =", list(encoded_values))

        # Decode a set of values using the encoder
        encoded_values = [3, 0, 4, 1]
        decoded_list = encoder.inverse_transform(encoded_values)
        print("Encoded values =", encoded_values)
        print("Decoded labels =", list(decoded_list))
    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def logistic_regression_classifier(option=1):
    # Define sample input data with two-dimensional
    # vectors and corresponding labels
    X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5],
                  [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9],
                  [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    if option == 1:
        ''' simple Logistic Regression classifier '''
        # Create the logistic regression classifier
        classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

        # Train the classifier
        classifier.fit(X, y)

        # Visualize the performance of the classifier
        visualize_classifier(classifier, X, y)
        plt.show()

    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def naive_bayes_classifier(option=1):
    input_file = 'data\\data_multivar_nb.txt'
    # Load data from input file
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    if option == 1:
        ''' Naïve Bayes classifier '''
        # Create Naïve Bayes classifier
        classifier = GaussianNB()

        # Train the classifier
        classifier.fit(X, y)

        # Predict the values for training data
        y_pred = classifier.predict(X)

        # Compute accuracy
        accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
        print(f"Accuracy of Naïve Bayes classifier = {round(accuracy, 2)}%")

        # Visualize the performance of the classifier
        visualize_classifier(classifier, X, y)

        # calculate the accuracy, precision, and recall values
        # based on threefold cross validation
        num_folds = 3
        accuracy_values = cross_val_score(
            classifier, X, y, scoring='accuracy', cv=num_folds)
        print(f"Accuracy: {round(100*accuracy_values.mean(), 2)}%")

        precision_values = cross_val_score(
            classifier, X, y, scoring='precision_weighted', cv=num_folds)
        print(f"Precision: {round(100*precision_values.mean(), 2)}%")

        recall_values = cross_val_score(
            classifier, X, y, scoring='recall_weighted', cv=num_folds)
        print(f"Recall: {round(100*recall_values.mean(), 2)}%")

        f1_values = cross_val_score(
            classifier, X, y, scoring='f1_weighted', cv=num_folds)
        print(f"F1: {round(100*f1_values.mean(), 2)}%")
        plt.show()

    elif option == 2:
        ''' new Naïve Bayes classifier '''
        # Split data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3)
        classifier_new = GaussianNB()
        classifier_new.fit(X_train, y_train)
        y_test_pred = classifier_new.predict(X_test)

        # compute accuracy of the classifier
        accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
        print(f"Accuracy of new classifier = {round(accuracy, 2)}%")

        # Visualize the performance of the classifier
        visualize_classifier(classifier_new, X_test, y_test)
        plt.show()

    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def confusion_matrix_creation(option=1):
    # Define sample labels
    true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
    pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]
    if option == 1:
        ''' Confusion matrix first example '''
        # Create confusion matrix
        confusion_mat = confusion_matrix(true_labels, pred_labels)

        # Visualize confusion matrix
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
        plt.title('Confusion matrix')
        plt.colorbar()
        ticks = np.arange(5)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.show()

        # Classification report
        targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
        print('\n', classification_report(true_labels, pred_labels,
                                          target_names=targets))
    elif option == 2:
        ''' option_purpose '''
        pass
    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def score_pred(classifier, X, y, label_encoder):

    # y = y.ravel()

    # Compute the F1 score of the SVM classifier
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print(f"F1 score: {round(100*f1.mean(), 2)}%")

    # Predict output for a test datapoint
    input_data = ['38', 'Private', '215646', 'HS-grad', '9', 'Divorced',
                  'Handlers-cleaners', 'Not-in-family', 'White',
                  'Male', '0', '0', '40', 'United-States']
    # Encode test datapoint
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            # input_data = np.reshape(input_data, (-1, 1))
            input_data_encoded[i] = int(
                label_encoder[count].transform(input_data[i]))
            count += 1
    input_data_encoded = np.array(input_data_encoded)
    # Run classifier on encoded datapoint and print output
    predicted_class = classifier.predict(input_data_encoded)
    print(label_encoder[-1].inverse_transform(predicted_class)[0])


def take_svm(option=1):
    ''' References
    https://www.districtdatalabs.com/building-a-classifier-from-census-data
    https://docs.bcbi.brown.edu/bidss/programming/unix/census/
    '''
    input_file = 'data\\income_data.txt'
    # Read the data
    X = []
    y = []
    count_class1 = 0
    count_class2 = 0
    max_datapoints = 25000
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and \
                    count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue
            data = line[:-1].split(', ')
            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1
    # Convert to numpy array
    X = np.array(X)

    # Convert string data to numerical data
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    if option == 1:
        ''' No Cross validation '''
        # Create SVM classifier
        classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        # Train the classifier
        classifier.fit(X, y)

        # predict the output using the classifier
        # print(y.shape)
        score_pred(classifier, X, y, label_encoder)
    elif option == 2:
        ''' Cross validation '''
        # Split data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=5)
        classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)
        # predict the output using the classifier
        score_pred(classifier, X_test, y_test_pred, label_encoder)
    else:
        print("Ejemplo: digo hola mundo")
        pass


def linear_regressior(option=1):
    if option == 1:
        ''' single variable '''
        # Input file containing data
        input_file = 'data\\data_singlevar_regr.txt'

        X_train, X_test, y_train, y_test = handling_simple_data(input_file)
        # '''
        #  : NOT NEED IF READ PICKLE SAVED DATA
        # Create linear regressor object
        regressor = linear_model.LinearRegression()

        # Train the model using the training sets
        regressor.fit(X_train, y_train)

        # Predict the output
        y_test_pred = regressor.predict(X_test)
        # '''
        # # Plot outputs
        # plt.scatter(X_test, y_test, color='green')
        # plt.plot(X_test, y_test_pred, color='black', linewidth=4)
        # plt.xticks(())
        # plt.yticks(())
        # plt.show()

        # Compute performance metrics
        print("Linear regressor performance:")
        print(f"MeanAE={round(mean_absolute_error(y_test,y_test_pred), 2)}")
        print(f"MSE={round(mean_squared_error(y_test,y_test_pred), 2)}")
        print(f"MedAE={round(median_absolute_error(y_test,y_test_pred), 2)}")
        print(f"EVS={round(explained_variance_score(y_test,y_test_pred), 2)}")
        print(f"R2 score = {round(r2_score(y_test, y_test_pred), 2)}")

        output_model_file = 'data\\regressor_model.pickle'
        # # dump to pickle file
        # with open(output_model_file, 'wb') as f:
        #     dump(regressor, f)

        # Load the model
        with open(output_model_file, 'rb') as f:
            regressor_model = load(f)

        # Perform prediction on test data
        y_test_pred_new = regressor_model.predict(X_test)
        print("\nNew meanAE=", round(
            mean_absolute_error(y_test, y_test_pred_new), 2))

    elif option == 2:
        ''' multivariable '''
        # Input file containing data
        input_file = 'data\\data_multivar_regr.txt'

        X_train, X_test, y_train, y_test = handling_simple_data(input_file)

        # Create the linear regressor model
        linear_regressor = linear_model.LinearRegression()
        # Train the model using the training sets
        linear_regressor.fit(X_train, y_train)
        # Predict the output
        y_test_pred = linear_regressor.predict(X_test)

        # Compute performance metrics
        print("Linear regressor performance:")
        print(f"MeanAE={round(mean_absolute_error(y_test,y_test_pred), 2)}")
        print(f"MSE={round(mean_squared_error(y_test,y_test_pred), 2)}")
        print(f"MedAE={round(median_absolute_error(y_test,y_test_pred), 2)}")
        print(f"EVS={round(explained_variance_score(y_test,y_test_pred), 2)}")
        print(f"R2 score = {round(r2_score(y_test, y_test_pred), 2)}")

        # Polynomial regression
        polynomial = PolynomialFeatures(degree=10)
        X_train_transformed = polynomial.fit_transform(X_train)
        datapoint = [[7.75, 6.35, 5.56]]
        poly_datapoint = polynomial.fit_transform(datapoint)

        poly_linear_model = linear_model.LinearRegression()
        poly_linear_model.fit(X_train_transformed, y_train)
        print("\nLinear regression:\n", linear_regressor.predict(datapoint))

        print("\nPolynomial regression:\n",
              poly_linear_model.predict(poly_datapoint))
    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def take_svr(option=1):
    # Load housing data
    data = datasets.load_boston()

    # Shuffle the data
    X, y = shuffle(data.data, data.target, random_state=7)

    # Split the data into training and testing datasets
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    if option == 1:
        ''' option_purpose '''
        # Create Support Vector Regression model
        sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

        # Train Support Vector Regressor
        sv_regressor.fit(X_train, y_train)

        # Evaluate performance of SVR
        y_test_pred = sv_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        evs = explained_variance_score(y_test, y_test_pred)
        print("\n#### Performance ####")
        print(f"Mean squared error = {round(mse, 2)}")
        print(f"Explained variance score = {round(evs, 2)}")

        # Test the regressor on test datapoint
        test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052,
                     26, 666, 20.2, 351.34, 15.27]
        print("\nPredicted price:", sv_regressor.predict([test_data])[0])
    else:
        ''' option_purpose '''
        print("Ejemplo: digo hola mundo")
        pass


def main():
    # preprocessing_data(int(sys.argv[1]))

    # label_encoding(int(sys.argv[1]))

    # logistic_regression_classifier(int(sys.argv[1]))

    # naive_bayes_classifier(int(sys.argv[1]))

    # confusion_matrix_creation(int(sys.argv[1]))

    take_svm(int(sys.argv[1]))

    # linear_regressior(int(sys.argv[1]))

    # take_svr(int(sys.argv[1]))


if __name__ == "__main__":
    main()
