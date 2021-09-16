import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
from pickle import dump, load


def train_knn(file_path):
    df = pd.read_csv(file_path)
    df.replace('?', -99999, inplace=True)
    df.columns = ['id', 'Clump Thickness', 'Cell Size', 'Cell Shape',
                  'Marginal Adhesion', 'Single Epithelial', 'Bare Nuclei',
                  'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class']
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    with open('data\\training\\knn.pickle', 'wb') as f:
        dump(clf, f)


def read_knn(file_path, example):
    with open(file_path, 'rb') as f:
        clf = load(f)
        prediction = clf.predict(example.reshape(len(example), -1))
        print(prediction)


def knn(option=1):
    if option == 1:
        ''' option_purpose '''
        file_path = 'data\\breast-cancer-wisconsin.data'
        train_knn(file_path)
    elif option == 2:
        ''' option_purpose '''
        file_path = 'data\\training\\knn.pickle'
        example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
        read_knn(file_path, example_measures)

    elif option == 3:
        ''' option_purpose '''
        pass
    elif option == 4:
        ''' option_purpose '''
        pass
    elif option == 5:
        ''' option_purpose '''
        pass
    else:
        ''' option_purpose '''
        print("Example: I say Hello world")
        pass
    pass


def main():
    knn(int(sys.argv[1]))


if __name__ == "__main__":
    main()
