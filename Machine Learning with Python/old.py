# # ############## Intro and Data ##############
# from sklearn import preprocessing, svm
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# import investpy
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# # import quandl
# import math
# import datetime
# from matplotlib import style
# import pickle

# # quandl.ApiConfig.api_key = 'isu4pbfFzpfUnowC-k-R'
# style.use('ggplot')
# """
# startdate = '01/01/2010'
# enddate = datetime.date.today().strftime("%d/%m/%Y")
# # df = quandl.get('WIKI/NVDA')
# df = investpy.stocks.get_stock_historical_data(
#     'nvda', 'united states', startdate, enddate)
# df.to_csv('data/NVDA.csv')
# # print(df.head())
# # print(df.tail())
# """

# # ############## Intro and Data ##############


# if __name__ == '__main__':
#     # raw data
#     df = pd.read_csv('data/NVDA.csv', header=0,
#                      index_col='Date', parse_dates=True)
#     # get needed columns
#     df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
#     # volatility analysis: high minus low % change or daily percent change
#     # Close, Spread/Volatility, %change
#     df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
#     #  % spread based on the closing price
#     df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
#     df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# # ############## Features and Labels ##############
#     forecast_col = 'Close'
#     # fill is better than drop
#     df.fillna(value=-99999, inplace=True)
#     forecast_out = int(math.ceil(0.01*len(df)))
#     df['label'] = df[forecast_col].shift(-forecast_out)
#     """
#     X = np.array(df.drop(['label'], 1))
#     y = np.array(df['label'])
#     # Standardize a dataset along any axis
#     # range of -1 to 1. speeds up. accuracy
#     X = preprocessing.scale(X)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     # Choosing the right estimator
#     # https://scikit-learn.org/stable/tutorial/machine_learning_map/

#     # old code
#     # clf = svm.SVR()
#     # clf = LinearRegression(n_jobs=-1)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
#     print(accuracy)

#     # 'precomputed': ValueError: Precomputed matrix must be a square matrix
#     # https://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
#     for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#         clf = svm.SVR(kernel=k)
#         clf.fit(X_train, y_train)
#         accuracy = clf.score(X_test, y_test)
#         print(k, accuracy)
#     """

# # ############## Forecasting and Predicting ##############
#     X = np.array(df.drop(['label'], 1))
#     X = preprocessing.scale(X)
#     X_lately = X[-forecast_out:]
#     # move backward forecast_out days
#     X = X[:-forecast_out]
#     df.dropna(inplace=True)
#     y = np.array(df['label'])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     '''
#     # comment after finish model training
#     clf = LinearRegression(n_jobs=-1)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)

#     # saving python object to pickle:
#     with open('data/nvdatraining.pickle', 'wb') as f:
#     # with open('data/tslatraining.pickle', 'wb') as f:
#         pickle.dump(clf, f)
#     '''
#     # loading time
#     # pickle_in = open('data/nvdatraining.pickle', 'rb')
#     pickle_in = open('data/tslatraining.pickle', 'rb')
#     clf = pickle.load(pickle_in)

#     forecast_set = clf.predict(X_lately)
#     # print(forecast_set, accuracy, forecast_out)
#     df['Forecast'] = np.nan
#     last_date = df.iloc[-1].name
#     last_unix = last_date.timestamp()
#     one_day = 86400
#     next_unix = last_unix + one_day
#     for price in forecast_set:
#         next_date = datetime.datetime.fromtimestamp(next_unix)
#         next_unix += 86400
#         # print row with all columns = Nan except the last is predict price
#         df.loc[next_date] = [np.nan for _ in range(
#             len(df.columns)-1)] + [price]
#     # df.to_csv('data/TSLA_predict.csv')
#     # move back, compare real label to predict label
#     # ??? what about today
#     df['Close'].plot()
#     df['Forecast'].plot()
#     plt.legend(loc=4)
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.show()

#     # 2 solutions:
#     # 1. combine, find algo for the last day with old mechanic
#     # 2. fill date time in future then continue run old mechanic (FALSE)
#     # https://medium.com/swlh/predict-gold-prices-with-scikit-learn-d3eb07496d3e#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjE3MTllYjk1N2Y2OTU2YjU4MThjMTk2OGZmMTZkZmY3NzRlNzA4ZGUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MjI0ODUwNjksImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwMTUyNTI3OTcwOTMyODc4MTQ4OCIsImVtYWlsIjoibWFuaGh1bmcuZHQ2QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiTmd1eWVuIEh1bmciLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2lXRTlIZnZ4cXcwazRnV2FkbTlNMnVHc1l3UFIweTlOdjJSMmE0c1E9czk2LWMiLCJnaXZlbl9uYW1lIjoiTmd1eWVuIiwiZmFtaWx5X25hbWUiOiJIdW5nIiwiaWF0IjoxNjIyNDg1MzY5LCJleHAiOjE2MjI0ODg5NjksImp0aSI6IjRmYzZlN2ViOWNlY2IxMGEwYTE4ZThjZGUwMGQ3ODgzODUwYWJhYjQifQ.hiqty1r177rKIzyOy2hnQBAa9n59pLZVPyDREut_ggYYUfazg22wzmzmern8wzB-ZCvb8t5W3PZl_88Uh_UmJFDzVcZv7iuCMeclobkPO4ghF-73_w9yDtMGN3V0UvFv29VN9bZwGrHBUdpqedvK6FAgwQBzVsLTwCUckPnW3dxl2ZpVFLBo5rRuvAhcHpuqE884DApux1lJvd9j6-zZ6zkwWgTP0gJZr9M7Diegyuj7zfRm06ybyX71kv1AqjoVRkwcZAcUmEOCP0OzfD0_CLRfOl1twaOnh07kcjM7CSDGOTpkntJ9Pgf8QLCyk-MJqBr7reO5ocDZ1R6Db7xrfw


# # ---------------------------------------------------------------------
# # ############## Regression - Theory ##############
# #  linear algebra objective: calculate points relationships in vector space
# # dataset isn't continuous ???

# ############## Best Fit Slope ##############
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# xs = [1, 2, 3, 4, 5]
# ys = [5, 4, 6, 5, 6]

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)


def bet_fit_slope(xs, ys):
    m = (mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)**2)-mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b


m, b = bet_fit_slope(xs, ys)
# print(m, b)
# ############## Best Fit Line ##############
regression_line = [(m*x)+b for x in xs]

predict_x = 7
predict_y = (m*predict_x) + b

plt.scatter(xs, ys, color='b', label='data')
plt.scatter(predict_x, predict_y, color='g', label='predict')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()

# # https://scikit-learn.org/stable/tutorial/machine_learning_map/
