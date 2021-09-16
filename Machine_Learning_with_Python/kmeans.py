import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')


# ORIGINAL:

def original():
    X = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8],
                  [8, 8],
                  [1, 0.6],
                  [9, 11]])

    # plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5, zorder=10)
    # plt.show()

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    centroids = clf.cluster_centers_
    labels = clf.labels_

    colors = ["g.", "r.", "c.", "y."]
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker="x", s=150, linewidths=5, zorder=10)
    plt.show()
    return


# def non_numerical_handling(*args, **kwargs):
def non_numerical_handling():
    df = pd.read_excel('data\\origin\\titanic.xls')

    df.drop(['body', 'name'], 1, inplace=True)
    # df = pd.to_numeric(df, errors='coerce')
    df = df.apply(pd.to_numeric, axis=0)
    df.fillna(0, inplace=True)

    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    print(df.head())
    # return df


def funcname(option=1):
    if option == 1:
        ''' option_purpose '''
        print("Example: I say Hello world")
        original()
    elif option == 2:
        ''' option_purpose '''
        non_numerical_handling()
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
        pass
    pass


def main():
    funcname(int(sys.argv[1]))


if __name__ == "__main__":
    main()
