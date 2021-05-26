import sklearn
import quandl
import pandas as pd
import os

quandl.ApiConfig.api_key = 'isu4pbfFzpfUnowC-k-R'

# df = quandl.get('USTREASURY/YIELD')
# df = quandl.get('WIKI/TSLA')
# df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# df.to_csv('data/TSLA.csv')

df = pd.read_csv('data/TSLA.csv')

print(df.tail())
