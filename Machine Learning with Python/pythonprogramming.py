import pandas as pd
import investpy as iv
import numpy as np
from sklearn import preprocessing, svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statistics import mean

import random
from datetime import date
from math import ceil, sqrt
from pickle import load, dump
from collections import Counter
import warnings

import matplotlib.pyplot as plt
from matplotlib import style

# style.use('ggplot')
style.use('fivethirtyeight')

if __name__ == "__main__":

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def get_data(quote, start, end, interval):
        df = iv.commodities.get_commodity_historical_data(
            quote, start, end, interval=interval)

        # df = iv.crypto.get_crypto_historical_data(
        #     quote, start, end, interval=interval)

        # df = iv.stocks.get_stock_historical_data(
        #     quote, 'united states', start, end, interval=interval)

        df.drop('Currency', axis=1, inplace=True)
        df['HL_PCT'] = (df['High'] - df['Low'])/df['Low']
        df['PCT_change'] = (df['Close'] - df['Open'])/df['Open']
        df['PCT_Close'] = df['Close'].pct_change()
        df['PCT_Vol'] = df['Volume'].pct_change()
        # print(df.tail(20))
        df.to_csv(f'data/source/{quote}_{interval}.csv')

    def read_data(quote, interval):
        df = pd.read_csv(f'data/source/{quote}_{interval}.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # # main stream
        # df = df[['Close', 'Volume', 'HL_PCT',
        #          'PCT_change', 'PCT_Close', 'PCT_Vol']]

        df = df[['High', 'Low', 'Close', 'Open', 'Volume', 'HL_PCT',
                 'PCT_change', 'PCT_Close', 'PCT_Vol']]

        # # treated NaN as an outlier feature
        # df.fillna(value=-99999, inplace=True)
        return df

    # label : forecast_col
    def reg_proc(df, forecast_col, useSVM=True, size=0.01):
        # process: sample vs population
        forecast_out = int(ceil(size * len(df)))    # sample number
        df['label'] = df[forecast_col].shift(-forecast_out)

        # forecast_col = 'Close'
        # forecast_out = int(ceil(size * len(df)))
        # df['label'] = df[forecast_col].shift(-forecast_out)

        # df = df['2020': '2021']
        # print(df.info())
        # print(df.tail(20))

        # Scikit-learn actually fundamentally requires numpy arrays
        # speeds up processing and increase accuracy
        X = np.array(df.drop(['label'], 1))  # feature
        X = preprocessing.scale(X)  # want features range of -1 to 1

        # extract date range
        last_dates = df.iloc[-forecast_out:].index

        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        df.dropna(inplace=True)

        y = np.array(df['label'])   # label

        # training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        '''
        # define classifier
        if useSVM:
            clf = svm.SVR()
        else:
            clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)   # self : returns an instance of self.
        # test
        confidence = clf.score(X_test, y_test)
        with open(f'reg_proc_{forecast_col}.pickle', 'wb') as f:
            dump(clf, f)
        '''
        pickle_in = open(f'reg_proc_{forecast_col}.pickle', 'rb')
        clf = load(pickle_in)

        # array of forecasts
        forecast_set = clf.predict(X_lately)
        return pd.DataFrame(forecast_set, index=last_dates,
                            columns=['ForeCast '+forecast_col])

    def regression():
        # start = '01/01/2010'
        # today = date.today().strftime("%d/%m/%Y")

        # get_data('Gold', start, today, 'Daily')
        # get_data('Gold', start, today, 'Weekly')
        # get_data('Gold', start, today, 'Monthly')

        reg_df = read_data('Gold', 'Daily')
        reg_df.drop(['PCT_Close', 'PCT_Vol'], axis=1, inplace=True)

        # handle missing data
        reg_df = clean_dataset(reg_df)  # outliner: nan, zero, ...
        print(reg_df.tail())

        # # main close
        # forecast_col = 'Close'
        # forecast_out = int(ceil(size * len(reg_df)))
        # reg_df['label'] = reg_df[forecast_col].shift(-forecast_out)

        dfcopy_h = reg_df.copy()
        high_df = reg_proc(dfcopy_h, 'High', False, 0.00125)

        dfcopy_l = reg_df.copy()
        low_df = reg_proc(dfcopy_l, 'Low', False, 0.00125)

        # com_df = pd.concat([reg_df, high_df, low_df], axis=1)
        com_df = pd.concat([high_df, low_df], axis=1)
        '''
        # com_df
        Constraint: High ko dc < Close hqua
        '''
        print(com_df.tail(15))

    def best_fit_slope_and_intercept(xs, ys):
        m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2-mean(xs**2))
        b = mean(ys) - m*mean(xs)
        return b, m

    def squared_error(ys_orig, ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))

    def coefficient_of_determination(ys_orig, ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        # print(y_mean_line)  # ???
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        r_squared = 1 - squared_error_regr/squared_error_y_mean
        return r_squared

    def create_dataset(hm, variance, step=2, correlation=False):
        val = 10
        ys = []
        for i in range(hm):
            y = val + random.randrange(-variance, variance)
            ys.append(y)
            if correlation and correlation == 'pos':
                val += step
            elif correlation and correlation == 'neg':
                val -= step
        # xs ~ time: add(shift)/ multiply(scale)
        xs = [i for i in range(len(ys))]
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    def best_func():
        # # own old code
        # xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        # ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

        # xs, ys = create_dataset(40, 40, 2, correlation='pos')
        xs, ys = create_dataset(40, 40, 2, correlation='neg')
        b, m = best_fit_slope_and_intercept(xs, ys)
        regression_line = [(m*x)+b for x in xs]
        r_squared = coefficient_of_determination(ys, regression_line)
        print(b, m, r_squared)

        # # own point
        # tmp_x = 7.3
        # tmp_y = (m*tmp_x)+b
        # plt.scatter(tmp_x, tmp_y, color='red')

        plt.scatter(xs, ys, color='green', label='data')  # handles with labels
        plt.plot(xs, regression_line)
        plt.legend(loc=4)

        plt.show()

    def cancer_ds():
        df = pd.read_csv('data/breast-cancer-wisconsin.data')
        df.replace('?', -99999, inplace=True)

        df.columns = ['id', 'Clump Thickness', 'Cell Size', 'Cell Shape',
                      'Marginal Adhesion', 'Single Epithelial', 'Bare Nuclei',
                      'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class']
        df.drop(['id'], 1, inplace=True)
        return df

    def classifier_task(task):
        df = cancer_ds()

        X = np.array(df.drop(['class'], 1))
        y = np.array(df['class'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        # changing
        clf = task()

        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

        example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
        # example_measures = np.array(
        #     [[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])

        # single example vs single feature
        print(clf.predict(example_measures.reshape(len(example_measures), -1)))

    def euclidean_distance(plot1, plot2):
        # # numpy sqrt: quite a bit faster
        # return np.sqrt(np.sum((np.array(plot1)-np.array(plot2))**2))

        # # not using numpy: simple measuring the length of a line
        # return sqrt(
        #     (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

        # numpy norm: same calculation: measures the magnitude of a vector
        return np.linalg.norm(np.array(plot1)-np.array(plot2))

    def test_euclidean():
        plot1 = [1, 3]
        plot2 = [2, 5]
        ed = euclidean_distance(plot1, plot2)
        print(ed)

    def knn2():
        # create dataset
        dataset = {'k': [[1, 2], [2, 3], [3, 1]],
                   'r': [[6, 5], [7, 7], [8, 6]]}
        new_features = [5, 7]
        [[plt.scatter(ii[0], ii[1], s=100, color=i)
          for ii in dataset[i]] for i in dataset]
        plt.scatter(new_features[0], new_features[1], s=100)

        result = k_nearest_neighbors(dataset, new_features)
        plt.scatter(new_features[0], new_features[1], s=100, color=result)
        plt.show()

    def k_nearest_neighbors(data, predict, k=3):
        # voting groups!/ total vote
        if len(data) >= k:
            warnings.warn('K is set to a value less than total voting groups!')
        distances = []
        for group in data:
            for features in data[group]:    # value of dict (corrdinate)
                ed = euclidean_distance(features, predict)
                distances.append([ed, group])
        # print(sorted(distances)[:k])    # get 3 nearest point
        votes = [i[1] for i in sorted(distances)[:k]]
        # extract feature
        vote_result = Counter(votes).most_common(1)[0][0]

        return vote_result

    def breast_cancer_ds():
        df = cancer_ds()

        full_data = df.astype(float).values.tolist()
        random.shuffle(full_data)

        test_size = 0.2

        # preparing label dataset ~ 2: no sick, 4: sick
        train_set = {2: [], 4: []}
        test_set = {2: [], 4: []}

        # split dataset
        train_data = full_data[:-int(test_size*len(full_data))]
        test_data = full_data[-int(test_size*len(full_data)):]
        # print(len(full_data), len(train_data), len(test_data))

        for i in train_data:
            # print(i[-1], i[:-1])
            train_set[i[-1]].append(i[:-1])

        for i in test_data:
            # print(i[-1], i[:-1])
            test_set[i[-1]].append(i[:-1])

        correct = 0
        total = 0

        # ~ 2: no sick, 4: sick
        for group in test_set:
            for data in test_set[group]:
                # data is feature: attribute columns in data set
                vote = k_nearest_neighbors(train_set, data, k=5)
                if group == vote:
                    correct += 1
                total += 1
        print('Accuracy:', correct/total)

    '''
    start = '01/01/2010'
    end = '23/08/2021'
    get_data('Gold', start, end, 'Daily')
    get_data('Gold', start, end, 'Weekly')
    get_data('Gold', start, end, 'Monthly')

    # regression()
    # best_func()
    # test_euclidean()
    # knn2()
    # breast_cancer_ds()

    # # knn() changing to task name
    # classifier_task(neighbors.KNeighborsClassifier)

    # # svm 4.6x faster than knn
    # classifier_task(svm.SVC)
    '''

    # building our Support Vector Machine class
    class SVM():
        def __init__(self, visualization=True):
            self.visualization = visualization
            self.colors = {1: 'r', -1: 'b'}
            if self.visualization:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(1, 1, 1)

        # train
        def fit(self, data):
            self.data = data  # data is a dict
            opt_dict = {}   # optimization dict
            transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
            all_data = []
            for k, v in self.data.items():
                for features in v:
                    for feature in features:
                        all_data.append(feature)
            self.max_feature_val = max(all_data)
            self.min_feature_val = min(all_data)
            all_data = None
            # print(self.max_feature_val, self.min_feature_val)

            step_sizes = [self.max_feature_val * 1]
            # step_sizes = [self.max_feature_val * 0.1,
            #               self.max_feature_val * 0.01,
            #               # point of expense:
            #               self.max_feature_val * 0.001]

            # extremely expensive
            b_range_mul = 1
            b_mul = 3
            latest_optimum = self.max_feature_val*10

            # # fixed range, but minimize step
            # for step in step_sizes:
            #     print(-1*(self.max_feature_val*b_range_mul),
            #           self.max_feature_val*b_range_mul,
            #           step*b_mul)

            for step in step_sizes:
                w = np.array([latest_optimum, latest_optimum])
                # print(w)    # [80 80]
                # we can do this because convex
                optimized = False

                # fucking kho hieu
                while not optimized:
                    for b in np.arange(-1*(self.max_feature_val*b_range_mul),
                                       self.max_feature_val*b_range_mul,
                                       step*b_mul):
                        for transformation in transforms:
                            w_t = w*transformation
                            # # current is [80, 80] then transform to
                            # # another axis [-80  80] [-80 -80] [ 80 -80]
                            # print(w_t)

                            found_opt = True
                            for k, v in self.data.items():
                                for xi in v:
                                    if not k*(np.dot(w_t, xi)+b) >= 1:
                                        found_opt = False
                            if found_opt:
                                # np.linalg.norm([80, -80]) vs
                                # array([ 80, -80]), -8 ~ plane vs bias
                                opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                    if w[0] < 0:
                        optimized = True
                        print('Optimized a step.')
                    else:
                        w = w - step
                # print(opt_dict)

                # # sort ascending norms keys
                # print(norms)
                norms = sorted([n for n in opt_dict])
                # { ||w||: [w,b] }
                opt_choice = opt_dict[norms[0]]  # get the smallest
                self.w = opt_choice[0]  # smallest point plane
                self.b = opt_choice[1]  # smallest point bias
                latest_optimum = opt_choice[0][0]+step*2  # hardcode
            '''
            '''

        # predict
        def predict(self, features):
            # sign( x.w+b )
            classification = np.sign(np.dot(np.array(features), self.w)+self.b)
            # if it isn't zero
            if classification != 0 and self.visualization:
                self.ax.scatter(
                    features[0], features[1], s=200, marker='*',
                    c=self.colors[classification])
            else:
                print('featureset', features, 'is on the decision boundary')
            return classification

        def visualize(self):
            # scattering known featuresets.
            [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
              for x in self.data[i]] for i in self.data]
            # hyperplane = x.w+b
            # v = x.w+b
            # psv = 1
            # nsv = -1
            # dec = 0

            def hyperplane(x, w, b, v):
                # v = (w.x+b)
                return (-w[0]*x-b+v) / w[1]

            datarange = (self.min_feature_val*0.9, self.max_feature_val*1.1)
            hyp_x_min = datarange[0]
            hyp_x_max = datarange[1]

            # (w.x+b) = 1
            # positive support vector hyperplane
            psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
            psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
            self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

            # (w.x+b) = -1
            # negative support vector hyperplane
            nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
            nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
            self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

            # (w.x+b) = 0
            # positive support vector hyperplane
            db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
            db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
            self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

            plt.show()

    def svm_scratch():
        # create object
        svm = SVM()
        # sample data
        data_dict = {-1: np.array([[1, 7],
                                   [2, 8],
                                   [3, 8]]),

                     1: np.array([[5, 1],
                                  [6, -1],
                                  [7, 3]])}
        # train
        svm.fit(data=data_dict)

        # # new income data
        # predict_us = [[0, 10],
        #               [1, 3],
        #               [3, 4],
        #               [3, 5],
        #               [5, 5],
        #               [5, 6],
        #               [6, -5],
        #               [5, 8]]
        # # predict all of them
        # for p in predict_us:
        #     svm.predict(p)
        # svm.visualize()

    # svm_scratch()
