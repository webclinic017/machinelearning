import sys
import numpy as np
import pandas as pd
import investpy as iv
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import style
from datetime import date
from math import ceil
from pickle import dump, load
from statistics import mean
import random
# import seaborn as sns
style.use('fivethirtyeight')

intervals = ['Daily', 'Weekly', 'Monthly']


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def get_data(quote):
    '''
    # start, end = find_days_in_month()   # need changing this
    quote, interval, day_range = inputs
    start, end = find_days_in_arange(day_range, interval)
    start, end = convert_date(start), convert_date(end)  # can reuse
    print(start, end)
    '''
    start, end = '01/01/2018', date.today().strftime("%d/%m/%Y")
    # quote, interval = inputs
    for interval in intervals:
        df = iv.commodities.get_commodity_historical_data(
            quote, start, end, interval=interval)
        # print(df.tail(10))
        # print(df)
        df.to_csv(f"data\\origin\\{quote}_{interval}.csv")
    return


def read_data(quote, interval):
    df = pd.read_csv(f"data\\origin\\{quote}_{interval}.csv")
    df.drop('Currency', axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # print(df.describe())
    # print(df.tail())
    return df


def fry_data(quote, interval):
    df = read_data(quote, interval)
    df['HL_PCT'] = (df['High'] - df['Low'])/df['Low']
    df['PCT_change'] = (df['Close'] - df['Open'])/df['Open']
    df['PCT_Close'] = df['Close'].pct_change()
    df['PCT_Vol'] = df['Volume'].pct_change()
    # print(df.describe())
    # print(df.tail())
    return df


def label_data(*inputs):

    return df, forecast_out


def train_test(*inputs):
    df, fore_col, useSVM, size, is_train = inputs
    forecast_out = int(ceil(size*len(df)))
    df['label'] = df[fore_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))  # feature
    X = preprocessing.scale(X)

    last_dates = df.iloc[-forecast_out:].index

    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    types = 'SVR' if useSVM else 'LR'
    if is_train:
        # define classifier
        if useSVM:
            # lack of kernel handle
            clf = svm.SVR()
        else:
            clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        with open(f'data\\training\\reg_{fore_col}_{types}.pickle', 'wb') as f:
            dump(clf, f)
    else:
        with open(f'data\\training\\reg_{fore_col}_{types}.pickle', 'rb') as f:
            clf = load(f)
    # test
    confidence = clf.score(X_test, y_test)
    print(confidence)
    forecast_set = clf.predict(X_lately)
    return pd.DataFrame(forecast_set, index=last_dates,
                        columns=['ForeCast '+fore_col])


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


def create_dataset(hm, variance, step=3, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def regression(option=1):
    quote, interval = "Gold", "Daily"
    if option == 1:
        ''' get_data '''
        print("Example: I say Hello world")
        # inputs = ("Gold", "Daily")
        # get_data(*inputs)
        get_data(quote)
    elif option == 2:
        ''' read_data '''
        df = read_data(quote, interval)
        print(df.tail())
    elif option == 3:
        ''' fry_data '''
        df = fry_data(quote, interval)
        print(df.tail())
    elif option == 4:
        ''' train_test '''
        # handle ouliner data
        df = fry_data(quote, interval)
        df.drop(['PCT_Close', 'PCT_Vol'], axis=1, inplace=True)
        df = clean_dataset(df)
        inputs = [df, 'Close', False, 0.01, False]
        df = train_test(*inputs)
        # print(df.tail(15))
    elif option == 5:
        ''' regression scrath '''
        xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)
        m, b = best_fit_slope_and_intercept(xs, ys)
        # print(m, b)
        regression_line = [(m*x)+b for x in xs]
        r_squared = coefficient_of_determination(ys, regression_line)

        predict_x = 7
        predict_y = (m*predict_x)+b
        print(r_squared, predict_y)

        plt.scatter(predict_x, predict_y, color='green', label='predict')
        plt.scatter(xs, ys, color='#003F72', label='data')
        plt.plot(xs, regression_line, label='regression line')
        plt.show()
    elif option == 6:
        ''' create_dataset '''
        xs, ys = create_dataset(40, 40, 4, correlation='pos')
        m, b = best_fit_slope_and_intercept(xs, ys)
        regression_line = [(m*x)+b for x in xs]
        r_squared = coefficient_of_determination(ys, regression_line)
        print(r_squared)
        plt.scatter(xs, ys, color='blue', label='data')
        plt.plot(xs, regression_line, label='regression line')
        plt.show()
    else:
        ''' option_purpose '''
        pass
    pass


def main():
    regression(int(sys.argv[1]))
    pass


if __name__ == "__main__":
    main()
