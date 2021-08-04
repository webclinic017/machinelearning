import pandas as pd
# import quandl
import investpy as iv
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# quandl.ApiConfig.api_key = 'isu4pbfFzpfUnowC-k-R'


if __name__ == "__main__":
    start = '01/01/2010'
    today = date.today().strftime("%d/%m/%Y")
    # analysis_path = 'data/analysis/'
    # ''' # T.B.D

    # def inner_volatility(source='data', quotes='GOOGL',
    #                      interval='Monthly'):   # , periods=6
    def inner_volatility(df):
        # df = pd.read_csv(source + f'/{quotes}_{interval}.csv')
        # df = df.iloc[-periods-1:]
        # length change
        df['COO'] = (df['Close']-df['Open'])/df['Open']*100
        # entire lenght
        df['HLL'] = (df['High']-df['Low'])/df['Low']*100
        # short edge
        df['HCC'] = (df['High']-df['Close'])/df['Close']*100
        # df['HLC'] = (df['High']-df['Low'])/df['Close']*100
        # long edge
        df['CLL'] = (df['Close']-df['Low'])/df['Low']*100
        # bull/ bear domination all around
        df['HOL'] = (df['High']-df['Open'])/df['Low']*100
        # bull/ bear Adj domination
        df['CLO'] = (df['Close']-df['Low'])/df['Open']*100
        # real body
        df['COL'] = (df['Close']-df['Open'])/df['Low']*100
        # strength volitality
        df['HLO'] = (df['High']-df['Low'])/df['Open']*100
        # pct chage day previous close
        df['PCT'] = (df['Close']-df['Open'])/df['Close']*100
        # df = df[-periods:]
        # df.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
        df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
        # df.set_index('Date', inplace=True)
        # df.to_csv(analysis_path + f'{quotes}_{interval}_vols.csv')
        return df

    # inner_volatility()
    # '''
    # ------------------------------------------------------------

    def get_data(source='data/source', quotes='GOOGL',
                        interval='Daily', samplesize=0.01):
        ''' # example download data
        # Collapse can be "daily","weekly",
        # "monthly", "quarterly" or "annual"
        '''
        # df = quandl.get(f"WIKI/{quotes}", collapse=interval)

        # df = iv.stocks.get_stock_historical_data(quotes,
        #                                          'united states', start,
        #                                          today, interval=interval)
        # df.to_csv(f'{source}/{quotes}_{interval}.csv')
        ''' # read saving data
        '''
        df = pd.read_csv(f'{source}/{quotes}_{interval}.csv', header=0,
                         index_col='Date', parse_dates=True)
        ''' # check percent of values in price are missing
        '''
        # print(df.isnull().sum()/len(df))
        ''' # get needed columns
        '''
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        ''' # simple example: high-low change and percent change
        '''
        # df['HL_PCT'] = (df['High']-df['Low'])/df['Close']*100
        # df['PCT'] = (df['Close']-df['Open'])/df['Close']*100

        ''' # test with others params
        '''
        df = inner_volatility(df)
        # df.to_csv(f'data/analysis/{quotes}_{interval}_vols.csv')

        # # maybe encounter GAP, so not using this formula
        # df['PCT'] = df['Close'].pct_change()*100
        ''' # new df
        '''
        # COO, HLL, HCC, CLL, HOL, CLO, COL, HLO, PCT
        df = df[['Close', 'COO', 'PCT', 'Volume']]
        # print(df.tail())
        ''' # Features and Labels
        '''
        forecast_col = 'Close'
        df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(samplesize*len(df)))
        df['label'] = df[forecast_col].shift(-forecast_out)
        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)

        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        df.dropna(inplace=True)

        y = np.array(df['label'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        ''' # apply kernel types
        for k in ['linear', 'poly', 'rbf']:
            clf = svm.SVR(kernel=k)
            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            print(k, confidence)
        '''
        # --------------------------------------
        ''' # apply LinearRegression
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print(confidence)
        '''
        # --------------------------------------
        clf = svm.SVR(kernel='linear')
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        forecast_set = clf.predict(X_lately)
        # print(forecast_set, confidence, forecast_out)
        df['Forecast'] = np.nan
        last_date = df.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day
        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += 86400
            df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        df.to_csv(f'data/predict/{quotes}_{interval}_forecast.csv')
        df['Close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Forecast {quotes} {interval} prices')
        print(quotes, last_date, round(confidence*100, 2), round(
            df['Forecast'].min(), 2), round(df['Forecast'].max(), 2))
        plt.show()

    def get_faang():
        ''' # FAANG stocks example
        '''
        items = ['FB', 'AMZN', 'MSFT', 'NFLX', 'GOOGL']
        for item in items:
            get_data(quotes=item, interval='Daily')
        ''' # only 1 stock
        '''
        # get_data(quotes='GOOGL')

    get_faang()
