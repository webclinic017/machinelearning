import investpy
import datetime
import pandas as pd
import xlrd
import requests as rq
from bs4 import BeautifulSoup

startdate = '01/01/2010'
enddate = datetime.date.today().strftime('%d/%m/%Y')
if __name__ == "__main__":
    '''
    # download data then dump
    df = investpy.currency_crosses.get_currency_cross_historical_data(
        'GBP/USD', startdate, enddate)
    # drop last columns
    # df = df.iloc[:, :-1]
    # df.to_pickle('gu_investpy.pickle')
    df.to_csv('gu.csv')
    '''
    # ------------------- Pickling -------------------
    # df = pd.read_pickle('gu_investpy.pickle')
    # print(df.tail())
    # ------------------- Flat file -------------------
    # df = pd.read_fwf('gu.csv')
    # df = pd.read_table('gu.csv')
    # df = pd.read_csv('gu.csv')
    # print(df.tail())

    # # ------------------- Clipboard -------------------

    # # df = pd.read_clipboard(header=None, squeeze=True)
    # # df.to_csv('com_performance.csv')
    # df = pd.read_csv('com_performance.csv')
    # # drop last unused row
    # df.drop(df.tail(1).index, inplace=True)
    # # set index to 'Commodity' col
    # df.set_index('Commodity', inplace=True)
    # # drop unused col
    # # df.drop(df.columns[[0]], axis=1, inplace=True)
    # df.to_csv('com_performance.csv')
    # print(df)
    # # # ------------------- Excel -------------------
    # # Error when read xls extension --------
    # # https://finance.vietstock.vn/HNG/thong-ke-giao-dich.htm

    # # xls but xml fucking
    # # xl_file = pd.ExcelFile('HNG.xls')
    # df = pd.read_html('HNG.xls')
    # # df = {sheet_name: xl_file.parse(sheet_name)
    # #       for sheet_name in xl_file.sheet_names}
    # # print(type(df))
    # df = df[1]
    # print(df.tail())
    # # df[new columns]

    # # Error when read xlsx extension --------
    # xl = pd.ExcelFile('usstocks.xlsx')
    # for name in xl.sheet_names:
    #     df = pd.read_excel(xl, name, engine='openpyxl')
    #     print(df)

    # # # ------------------- JSON -------------------
    # df = pd.read_csv('gu.csv')
    # df.to_json('gu.json')
    # df = pd.read_json('gu.json')
    # print(df)
    # # # ------------------- HTML -------------------
    # to_html vs read_html

    def pairs_extract(soup=soup):
        pairs = ['EURUSD', 'EURJPY', 'EURCAD', 'EURGBP', 'EURAUD', 'EURNZD',
                 'EURCHF', 'GBPUSD', 'GBPJPY', 'GBPCAD', 'GBPAUD', 'GBPNZD',
                 'GBPCHF', 'XAUUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD']
        r = rq.get('https://www.myfxbook.com/community/outlook')
        soup = BeautifulSoup(r.content, "html.parser")
        tmp_tbl = soup.find(id="outlookSymbolsTable")
        df = pd.read_html(tmp_tbl.prettify(), index_col=0)[0]
        drop_cols = ['Community Trend (Shorts vs Longs)',
                     'Symbol Popularity', 'Unnamed: 6']
        df.drop(drop_cols, 1, inplace=True)
        for item in df.index.tolist():
            if item not in pairs:
                df.drop(item, inplace=True)
        return df

    # print(pairs_extract().tail())
    # # # ------------------- XML -------------------
    # to_xml vs read_xml
    # # # ------------------- SQL -------------------
    # ...
