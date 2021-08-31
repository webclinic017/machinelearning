import pandas as pd
import numpy as np
import sys


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def preparing_data(num):
    df = pd.read_csv('data/Silver_Weekly.csv')
    df.drop('Currency', axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[-num:]
    return df

# ###########################################################
# ######### Data manipulations ##########
# ###########################################################
# ---------------- pandas.melt -----------------------


def melt(option=1):
    '''Unpivot a DataFrame'''
    tmp = {'A': {0: 'a', 1: 'b', 2: 'c'},
           'B': {0: 1, 1: 3, 2: 5},
           'C': {0: 2, 1: 4, 2: 6}}
    df = pd.DataFrame(tmp)
    if option == 1:
        print(df)
        print(pd.melt(df, id_vars=['A'], value_vars=['B']))
        print(pd.melt(df, id_vars=['A'], value_vars=['B', 'C']))
    elif option == 2:
        # preparing data
        df = preparing_data(3)
        print(df)

        # relocate DataFrame
        df = pd.melt(df, id_vars=['Open'], value_vars=[
            'High', 'Low', 'Close'], ignore_index=False,
            var_name='Cols', value_name='Values')
        print(df)
    elif option == 3:
        print(df)
        # by default: variable  value
        df_copy = df.copy()
        df_copy = pd.melt(df_copy, id_vars=['A'], value_vars=['B'])
        print(df_copy)

        # but you can set var_name | value_name
        df = pd.melt(df, id_vars=['A'], value_vars=['B'],
                     var_name='Hung', value_name='Nguyen')
        print(df)
    else:
        # If you have "multi-index columns"
        df.columns = [list('ABC'), list('DEF')]
        print(df)

        # clone df
        df_c = df.copy()
        df_c = pd.melt(df_c, col_level=0, id_vars=['A'], value_vars=['B'])
        print(df_c)

        df = pd.melt(df, id_vars=[('B', 'E')], value_vars=[('C', 'F')])
        print(df)


# ---------------- pandas.pivot IMPORTANT -----------------------
def pivot(option=1):
    ''' reshaped DataFrame '''
    if option == 1:
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                                   'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           #    'bar': ['A', 'B', 'C', 'D', 'E', 'F'],
                           'baz': [1, 2, 3, 4, 5, 6],
                           'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        print(df)
        print("----------------------------------")
        df_c1 = df.copy()
        df_c1 = df_c1.pivot(index='foo', columns='bar', values='baz')
        print(df_c1)

        print("----------------------------------")
        # same with above
        df_c2 = df.copy()
        df_c2 = df_c2.pivot(index='foo', columns='bar')['baz']
        print(df_c2)

        print("----------------------------------")
        df = df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
        print(df)

    elif option == 2:
        # preparing data
        df = preparing_data(3)
        df = df.pivot(index='Open', columns=['High', 'Low'], values='Close')
        print(df)

        # ideas: mean columns, percent change columns
        '''
        # 1. fill nan value ???
        # 2. using shift value
        # 3. diagonal matrix (features and labels)
        gg: "diagonal matrix features machine learning"
        '''
    else:
        df = pd.DataFrame({
            "lev1": [1, 1, 1, 2, 2, 2],
            "lev2": [1, 1, 2, 1, 1, 2],
            "lev3": [1, 2, 1, 2, 1, 2],
            "lev4": [1, 2, 3, 4, 5, 6],
            "values": [0, 1, 2, 3, 4, 5]})
        print(df)
        print("----------------------------------")
        df_c1 = df.copy()
        df_c1 = df_c1.pivot(index="lev1", columns=[
                            "lev2", "lev3"], values="values")
        print(df_c1)
        print("----------------------------------")

        df_c2 = df.copy()
        df_c2 = df_c2.pivot(index=["lev1", "lev2"], columns=[
            "lev3"], values="values")
        print(df_c2)
        print("----------------------------------")

        # df = pd.DataFrame({"foo": ['one', 'one', 'two', 'two'],
        #                    # Index contains duplicate entries -> Error
        #                    "bar": ['A', 'A', 'B', 'C'],
        #                    "baz": [1, 2, 3, 4]})
        # df = df.pivot(index='foo', columns='bar', values='baz')
        # print(df)


# ---------------- pandas.pivot_table IMPORTANT -----------------------
def pivot_table(option=1):
    ''' Create a spreadsheet-style pivot table | grouping ... '''
    if option == 1:
        # Create a spreadsheet-style pivot table
        # https://www.youtube.com/watch?v=tQRRfIG9UIY
        df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                                 "bar", "bar", "bar", "bar"],
                           "B": ["one", "one", "one", "two", "two",
                                 "one", "one", "two", "two"],
                           "C": ["small", "large", "large", "small",
                                 "small", "large", "small", "small",
                                 "large"],
                           "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                           "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        print(df)
        print("----------------------------------")
        df_c1 = df.copy()
        table = pd.pivot_table(df_c1, values='D', index=['A', 'B'],
                               # Sum of array elements over a given axis.
                               # , fill_value=0.0
                               columns=['C'], aggfunc=np.sum)
        print(table)
        print("----------------------------------")
        # taking the mean across multiple columns
        df_c2 = df.copy()
        table = pd.pivot_table(df_c2, values=['D', 'E'], index=['A', 'C'],
                               aggfunc={'D': np.mean,
                                        'E': np.mean})
        print(table)
        print("----------------------------------")
        # calculate multiple types of aggregations for any given value column
        table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                               aggfunc={'D': np.mean,
                                        'E': [min, max, np.mean]})
        print(table)

    elif option == 2:
        '''
        # 1. label the data
        # 2. ~ reshape data by pivot table
        # 3. using aggfunc
        # 4. fill nan | zero | inf values
        '''
        # preparing data
        # Index contains duplicate entries will be removed
        # auto sort values in columns 'High'
        df = preparing_data(3)
        print(df)
        print("----------------------------------")
        df_c1 = df.copy()
        table = pd.pivot_table(df_c1, index='Open', columns=['High', 'Low'],
                               values=['Close'], aggfunc=np.sum,
                               fill_value=0.0)

        print(table)
        print("----------------------------------")
        df_c2 = df.copy()
        table = pd.pivot_table(df_c2, index=['High', 'Low'],
                               values=['Open', 'Close', 'Volume'],
                               aggfunc={'Open': np.mean, 'Close': np.mean,
                                        'Volume': np.mean})
        print(table)
# ---------------- pandas.crosstab -----------------------
# use to quantitatively analyze
#     the relationship between multiple variables. ... By showing
#         how correlations change:
#             one group of variables to another
#     allows for the identification:
#         patterns
#         trends
#         probabilities within data sets


def crosstab(option=1):
    # look like label name
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
                  "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one",
                  "one", "two", "two", "two", "one"], dtype=object)
    c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
                  "shiny", "dull", "shiny", "shiny", "shiny"],
                 dtype=object)
    if option == 1:
        df = pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
        print(df)
        pass
    elif option == 2:
        df = preparing_data(3)
        print(df)
        print("----------------------------------")
        pass
    else:
        foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
        bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
        df = pd.crosstab(foo, bar)
        print(df)
        df = pd.crosstab(foo, bar, dropna=False)
        print(df)
        pass


# ---------------- pandas.cut IMPORTANT -----------------------
# Bin values into discrete intervals
'''
Use cut when you need to segment and sort data values into bins.
This function is also useful for going from a continuous variable
    to a categorical variable. For example, cut could convert
        ages to groups of age ranges.
Supports binning into an equal number of bins, or
    a pre-specified array of bins.
'''


def cut(option=1):
    if option == 1:
        # Discretize into three equal-sized bins
        # df = pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
        df = pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
        print(df)
        pass
    elif option == 2:
        df = preparing_data(3)
        print(df)
        print("----------------------------------")
    elif option == 3:
        '''
        Custome cut range???
        1. convert to numpy array
        2. Cut ...
            intervals??? labeled
        '''
        # assign specific labels
        df = pd.cut(np.array([1, 7, 5, 4, 6, 3]),
                    3, labels=["bad", "medium", "good"])
        print(df)
        print("----------------------------------")

        # unordered categories | allow non-unique labels
        # df = pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
        #             # accept duplicate
        #             labels=["B", "A", "B"], ordered=False)
        # print(df)
        # print("----------------------------------")

        # labels=False implies you just want the bins back
        df = pd.cut([0, 1, 1, 2], bins=4, labels=False)
        print(df)
        print("----------------------------------")
    elif option == 4:
        # Passing a Series as an input returns a Series with categorical dtype
        s = pd.Series(np.array([2, 4, 6, 8, 10]),
                      index=['a', 'b', 'c', 'd', 'e'])
        print(pd.cut(s, 3))

    elif option == 5:
        # Passing a Series as an input returns a Series with "mapping value".
        # used to "map numerically" to "intervals" based on "bins"
        s = pd.Series(np.array([2, 4, 6, 8, 10]),
                      index=['a', 'b', 'c', 'd', 'e'])
        print(pd.cut(s, [0, 2, 4, 6, 8, 12],
              labels=False, retbins=True, right=False))  # labels? right?
        pass
    elif option == 6:
        s = pd.Series(np.array([2, 4, 6, 8, 10]),
                      index=['a', 'b', 'c', 'd', 'e'])
        # Use drop optional when bins is not unique
        print(pd.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,
                     right=False, duplicates='drop'))
    else:
        bins = pd.IntervalIndex.from_tuples(
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        print(pd.cut([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], bins))
    pass


# ---------------- pandas.qcut -----------------------
# "Quantile"-based "discretization" function
# https://pbpython.com/pandas-qcut-cut.html
'''
Discretize variable into
    equal-sized buckets
        based on rank
        based on sample quantiles
produce a "Categorical" object
    indicating quantile membership for each data point
"ndarray or Series" | "Number of quantiles" | return bins (maybe scalar bins)
return : "Series of type category" or "Categorical"
'''


def qcut(option=1):
    if option == 1:
        print(pd.qcut(range(5), 4))
    elif option == 2:
        print(pd.qcut(range(5), 3, labels=['good', 'medium', 'bad']))
        # print(pd.qcut(range(5), 3, labels=[
        #       'good', 'medium', 'bad'], retbins=True))

    else:
        df = pd.qcut(range(10), 5, labels=False, retbins=True)
        print(df)
        for item in df:
            print(len(item), type(item))
    pass


# ----------- pandas.merge IMPORTANT NEED_PRACTICE -----------
# Merge DataFrame or named Series objects with a database-style join
'''
Series object ~ DataFrame single named column
how:
    left/right ~ keys from left/right frame
    | outer ~ union of keys from both frames
    | inner ~ intersection of keys from both frames
    | cross ~ cartesian product from both frames
'''


def merge(option=1):
    if option == 1:
        df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 5]})
        print(df1)
        print("----------------------------------")
        df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [5, 6, 7, 8]})
        print(df2)
        print("----------------------------------")
        # Merge df1 and df2 on the lkey and rkey columns
        df = df1.merge(df2, left_on='lkey', right_on='rkey')
        print(df)
        print("----------------------------------")
        # version has suffix
        df3 = df1.merge(df2, left_on='lkey', right_on='rkey',
                        suffixes=('_left', '_right'))
        print(df3)
        print("----------------------------------")
    elif option == 2:
        df1 = pd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        print(df1)
        df2 = pd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        print(df2)
        df = df1.merge(df2, how='outer', on='a')
        print(df)
        print("----------------------------------")
        df3 = df1.merge(df2, how='inner', on='a')
        print(df3)
        print("----------------------------------")
        df4 = df1.merge(df2, how='left', on='a')
        print(df4)
        print("----------------------------------")
        df5 = df1.merge(df2, how='right', on='a')
        print(df5)
        print("----------------------------------")
    else:
        df1 = pd.DataFrame({'left': ['foo', 'bar']})
        df2 = pd.DataFrame({'right': [7, 8]})
        df = df1.merge(df2, how='cross')
        print(df)
        # print("----------------------------------")
    pass


# ---------------- pandas.merge_ordered CONSIDER --------------
# Perform merge with optional filling/interpolation
'''
Designed for ordered data:
    time series data
    group-wise merge
fill_method : Interpolation method
The merged DataFrame output type will the be same as ‘left’
'''


def merge_ordered(option=1):
    if option == 1:
        df1 = pd.DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"]
            }
        )
        df2 = pd.DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
        df = pd.merge_ordered(df1, df2, fill_method="ffill", left_by="group")
        print(df)
    else:
        pass
    pass


# -------------- pandas.merge_asof IMPORTANT MUST CONSIDER --------------
# similar left-join except matching on nearest key rather than equal keys
'''
For each row in the left DataFrame:
    A “backward” search
    A “forward” search
    A “nearest” search

'''


def merge_asof(option=1):
    if option == 1:
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame(
            {"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})
        print(pd.merge_asof(left, right, on="a"))
        print(pd.merge_asof(left, right, on="a", allow_exact_matches=False))
        print(pd.merge_asof(left, right, on="a", direction="backward"))
        print(pd.merge_asof(left, right, on="a", direction="forward"))
        print(pd.merge_asof(left, right, on="a", direction="nearest"))
        pass
    elif option == 2:
        # use indexed DataFrames
        left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
        right = pd.DataFrame(
            {"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
        print(pd.merge_asof(left, right, left_index=True, right_index=True))
    else:
        # real-world times-series example
        quotes = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.030"),
                    pd.Timestamp("2016-05-25 13:30:00.041"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.049"),
                    pd.Timestamp("2016-05-25 13:30:00.072"),
                    pd.Timestamp("2016-05-25 13:30:00.075")
                ],
                "ticker": [
                    "GOOG",
                    "MSFT",
                    "MSFT",
                    "MSFT",
                    "GOOG",
                    "AAPL",
                    "GOOG",
                    "MSFT"
                ],
                "bid": [720.50, 51.95, 51.97, 51.99,
                        720.50, 97.99, 720.50, 52.01],
                "ask": [720.93, 51.96, 51.98, 52.00,
                        720.93, 98.01, 720.88, 52.03]
            }
        )
        print(quotes)
        print("----------------------------------")
        trades = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.038"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048")
                ],
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.0],
                "quantity": [75, 155, 100, 100, 100]
            }
        )
        print(trades)
        print("----------------------------------")
        # asof of the quotes
        print(pd.merge_asof(trades, quotes, on="time", by="ticker"))
        print("----------------------------------")
        # asof within 2ms between the quote time and the trade time
        df = pd.merge_asof(
            trades, quotes, on="time", by="ticker",
            tolerance=pd.Timedelta("2ms")
        )
        print(df)
        print("----------------------------------")
        # asof within 10ms
        df2 = pd.merge_asof(
            trades,
            quotes,
            on="time",
            by="ticker",
            tolerance=pd.Timedelta("10ms"),
            allow_exact_matches=False
        )
        print(df2)

# ---------------- pandas.concat -----------------------
# Concatenate pandas objects
'''

'''
# ---------------- pandas.get_dummies -----------------------
# ---------------- pandas.factorize -----------------------
# ---------------- pandas.unique -----------------------
# ---------------- pandas.wide_to_long -----------------------

# ###########################################################
# ######### Top-level missing data ##########
# ###########################################################


# ###########################################################
# ######### Top-level conversions ##########
# ###########################################################

# ###########################################################
# ######### Top-level dealing with datetimelike ##########
# ###########################################################

# ###########################################################
# ######### Top-level dealing with intervals ##########
# ###########################################################

# ###########################################################
# ######### Top-level evaluation ##########
# ###########################################################

# ###########################################################
# ######### Hashing ##########
# ###########################################################

# ###########################################################
# ######### Testing ##########
# ###########################################################


def main():
    a = int(sys.argv[1])
    # melt(a)
    # pivot(a)
    # pivot_table(a)
    # crosstab(a)
    # cut(a)
    # qcut(a)
    # merge(a)
    # merge_ordered(a)
    merge_asof(a)
    pass


if __name__ == "__main__":
    main()
    pass
