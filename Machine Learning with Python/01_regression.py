from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import investpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import quandl
import math
import datetime
from matplotlib import style
import pickle

# quandl.ApiConfig.api_key = 'isu4pbfFzpfUnowC-k-R'
style.use('ggplot')
"""
startdate = '01/01/2010'
enddate = datetime.date.today().strftime("%d/%m/%Y")
# df = quandl.get('WIKI/NVDA')
df = investpy.stocks.get_stock_historical_data(
    'nvda', 'united states', startdate, enddate)
df.to_csv('data/NVDA.csv')
# print(df.head())
# print(df.tail())
"""

# ############## Intro and Data ##############


if __name__ == '__main__':
    # raw data
    df = pd.read_csv('data/NVDA.csv', header=0,
                     index_col='Date', parse_dates=True)
    # get needed columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # volatility analysis: high minus low % change or daily percent change
    # Close, Spread/Volatility, %change
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    #  % spread based on the closing price
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# ############## Features and Labels ##############
    forecast_col = 'Close'
    # fill is better than drop
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01*len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    """
    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])
    # Standardize a dataset along any axis
    # range of -1 to 1. speeds up. accuracy
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Choosing the right estimator
    # https://scikit-learn.org/stable/tutorial/machine_learning_map/

    # old code
    # clf = svm.SVR()
    # clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    # 'precomputed': ValueError: Precomputed matrix must be a square matrix
    # https://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(k, accuracy)
    """

# ############## Forecasting and Predicting ##############
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    '''
    # comment after finish model training
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    # saving python object to pickle:
    with open('data/nvdatraining.pickle', 'wb') as f:
        pickle.dump(clf, f)
    '''
    # loading time
    pickle_in = open('data/nvdatraining.pickle', 'rb')
    clf = pickle.load(pickle_in)

    forecast_set = clf.predict(X_lately)
    # print(forecast_set, accuracy, forecast_out)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    df.to_csv('data/NVDA_predict.csv')
    df['Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
