import numpy as np
import pandas as pd
from os import path
# import sys

# window explain:
'''
https://towardsdatascience.com/window-functions-in-pandas-eaece0421f7
'''


# rolling
def rolling_count(df, periods=20, window=3):
    s = pd.Series([2, 3, np.nan, 10])
    # print(s)
    print(type(s.rolling(window).count()))

    print(s.rolling(window).count())
    # 2, [2, 3], [2, 3, nan], [3, nan, 10]
    # print(s.rolling(4).count())
    # # 2, [2, 3], [2, 3, nan], [2, 3, nan, 10]
    '''
    '''

    return


def rolling_sum(df, periods=20, window=3):
    # series part ------------------
    roll_series = df.tail(periods).PCT
    print(roll_series)
    print(roll_series.rolling(window).sum())
    # df part ------------------


def rolling_window(option=1):
    df = read_data()
    if option == 1:
        ''' option_purpose '''
        rolling_count(df)
    elif option == 2:
        ''' option_purpose '''
        rolling_sum(df, periods=5, window=2)
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


# expanding

# ExponentialMovingWindow
def main():
    rolling_window(int(sys.argv[1]))


if __name__ == "__main__":
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
    from commonfunc.inputdata import read_data
    main()
    exit(main())
