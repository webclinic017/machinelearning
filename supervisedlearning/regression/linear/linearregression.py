# https://machinelearningcoban.com/2016/12/28/linearregression/

# __future__ : to use new language features
from __future__ import division, print_function, unicode_literals
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # # -------------Sp500 example-------------------------
pairs = ['SP500', 'US10Yeild']


def combine_data():
    main_df = pd.DataFrame()
    for count, ticker in enumerate(pairs):
        df = pd.read_csv(f'data/{ticker}.csv')
        df.set_index('Date', inplace=True)

        # df.dropna(axis=1, inplace=True)
        df.rename(columns={'Price': f'{ticker} Price',
                  'Change %': f'{ticker} Change %'}, inplace=True)
        df.drop(['Open', 'High', 'Low'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='left')
        # print('{} and {}'.format(count, ticker))
    main_df.drop(['Vol.'], 1, inplace=True)
    main_df.dropna(axis=0, how='any', inplace=True)
    # main_df.to_csv('data/pairs_joined_closes.csv')
    # print(main_df.head())
    return main_df


# combine_data()
df = pd.read_csv('data/pairs_joined_closes.csv')
# reverse order
df = df.iloc[::-1]
# df.set_index('Date', inplace=True)
# tmp = df['Price'].to_numpy()
print(df.head())


# add Correlation Score


# # # -----------------------------------------------------

# # input data
# # transpose feature array
# X = np.array(
#     [[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# # outcome
# y = np.array([[49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# # (13, 1): column, row
# # print(X.shape[0])
# # ------------------ cach thu cong --------------------
# # create matrix One
# one = np.ones((X.shape[0], 1))

# # Join a sequence of arrays along an existing axis.
# # return concatenated array

# Xbar = np.concatenate((one, X), axis=1)

# # A va b (outcome???)
# # Dot product of two arrays (nhan ma tran)
# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, y)
# # pseudo inverse
# w = np.dot(np.linalg.pinv(A), b)
# # print('w = ', w)

# # ------------------ dung sklearn --------------------

# # fit_intercept = False for calculating the bias
# # the intercept is forced to the origin (0, 0).
# regr = linear_model.LinearRegression(fit_intercept=False)
# regr.fit(Xbar, y)

# # ------------------ So sanh --------------------
# # Compare two results
# print('Solution found by scikit-learn  : ', regr.coef_)
# print('Solution found by (5): ', w.T)


# # # -----------------------------------------------------
# # Preparing the fitting line
# w_0 = w[0][0]
# w_1 = w[1][0]

# x0 = np.linspace(145, 185, 2)
# y0 = w_0 + w_1*x0

# y1 = w_1*155 + w_0
# y2 = w_1*160 + w_0

# print(f'Predict weight of person with height 155 cm: {y1:.2f} (kg),\
# real number: 52 (kg)')
# print(f'Predict weight of person with height 160 cm: {y2:.2f} (kg),\
# real number: 56 (kg)')
# # # -----------------------------------------------------

# draw plot----------------------------------

# # 'ro' stand for red circle
# plt.plot(X.T, y.T, 'ro')    # data
# plt.plot(x0, y0)
# # x, y corrdinate
# plt.axis([140, 190, 45, 75])
# # label name
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (cm)')
# plt.show()


# # # -----------------------------------------------------

# pip install prophet

# # --------------------- sklearn ----------------------------
